# Chapter 3: Data Matching

Welcome back to the `crest` tutorial! In the [previous chapter](02_data_abstractions_.md), we explored how `crest` uses **Data Abstractions** like `Datafile`, `Dataset`, `Block`, `Blockset`, and `Sample` to organize and represent your potentially massive datasets. We saw how the `Dataset.generate_samples()` method is the crucial step that orchestrates the creation of individual `Sample`s ready for processing.

Now, let's zoom in on a fundamental challenge that `Dataset.generate_samples()` must overcome, especially when dealing with data from multiple sources, like our satellite and weather data use case: **Data Matching**.

## What is Data Matching in CREST? (The Matchmaking Analogy Revisited)

Imagine you have two maps of the same city, but one is a satellite photo and the other shows weather station locations. To understand the weather at each satellite observation point, you need to find which weather stations are "close enough" to each satellite point.

Data Matching in `crest` is precisely this: a sophisticated **matchmaking service** for data points from different grids or blocks. It finds corresponding data points (neighbors) across different datasets or blocks based on their coordinates (like latitude, longitude, time) and a defined tolerance.

This is crucial because:

*   Data sources often have different spatial resolutions (e.g., 1km satellite pixels vs. 10km weather model grid cells).
*   Time steps might not align perfectly (e.g., satellite snapshot vs. hourly weather forecast).
*   Some data points might be missing or invalid in certain datasets.

Data Matching ensures that when you try to extract a `Sample` (a little data window) for a specific location and time, you only do so if valid data is available *and* corresponds meaningfully across *all* the `Datafile`s you've included in your `Dataset`.

## Your Use Case: Aligning Satellite and Weather Data

In our example, we have:

*   Satellite data (`soil_moisture`) with one grid.
*   Weather data (`temperature`, `precipitation`) with another grid.

These grids likely have different sets of latitude, longitude, and time points. Before we can create a `Sample` that combines `soil_moisture`, `temperature`, and `precipitation` for a specific event (e.g., a 3x3 pixel window over 7 time steps), `crest` needs to figure out which satellite grid points correspond to which weather grid points within a certain "distance" or tolerance.

This is where Data Matching comes into play, working *within* each `Blockset`.

## How Data Matching Fits into `generate_samples()`

You, as the user, don't typically call a "match data" function directly. The Data Matching process is an internal step performed by the `Blockset` when it's asked to generate `Sample`s.

Let's revisit the `generate_samples()` call:

```python
# Example: Generating samples (revisited)
# Assuming my_dataset is already created from Chapter 2

print("\nGenerating samples...")

# Calling this method triggers the entire process, including data matching
lazy_samples_array = my_dataset.generate_samples(compute=False, verbose=True)

print(f"Generated a lazy dask array of samples: {lazy_samples_array}")
```

When `my_dataset.generate_samples()` is called, `crest` first breaks down the large datasets into smaller `Block`s and groups corresponding `Block`s into `Blockset`s (as discussed in Chapter 2).

Then, for *each* `Blockset`, the following happens:

1.  The `Blockset` loads the actual coordinate data for the `Block`s it contains.
2.  It passes these coordinates and their associated "resolutions" (effectively, defining the size of each grid cell or the expected spacing) to the Data Matching subsystem.
3.  The Data Matching subsystem finds all pairs (or triplets, etc., depending on the number of `Datafile`s) of indices from the different blocks that fall within the defined tolerance of each other.
4.  The `Blockset` uses these matching indices to know exactly which locations have valid, corresponding data across all its blocks.
5.  For each set of matching indices, the `Blockset` extracts the necessary data windows (`window_depth`) from its blocks to assemble a `Sample`.

So, Data Matching is the engine *inside* the `Blockset` that determines *which* specific locations yield valid `Sample`s.

## Behind the Scenes: The Matchmaking Engine

How does `crest` find these corresponding points efficiently, especially for large grids? This is where the optimized algorithms come in.

Let's trace the process within a single `Blockset`:

```{mermaid}
sequenceDiagram
    participant Blockset as Blockset Object
    participant Blocks as Blocks (from Datafiles)
    participant MatchupLogic as Matching Subsystem (e.g., brute.py)
    participant NumbaFunc as Numba Function (bruteforce_double)
    participant ValidIndices as Matching Indices

    Blockset->>Blocks: Load coordinate data for my Blocks
    Blocks-->>Blockset: Return coordinates & resolutions
    Blockset->>MatchupLogic: Call find_neighbors/brute(...)
    MatchupLogic->>MatchupLogic: Organize coordinates into Grid/Pair objects
    MatchupLogic->>NumbaFunc: Call bruteforce_double(...) with data, resolutions, tolerance
    NumbaFunc->>NumbaFunc: Efficiently iterate and check point pairs within tolerance
    NumbaFunc-->>MatchupLogic: Return arrays of matching indices (e.g., satellite_indices, weather_indices)
    MatchupLogic-->>Blockset: Return arrays of matching indices
    Blockset->>Blockset: Filter out indices with invalid data (using valid masks)
    Blockset->>Blocks: For each final matching index set, extract data windows
    Blocks-->>Blockset: Return data windows for each Sample
    Blockset-->>Blockset: Assemble Sample objects
    Blockset-->>DaskArray: Return array of Samples/SampleSets (as a task result)
```

This diagram shows that the `Blockset` delegates the core matching task to a specialized subsystem (like the `brute.py` module). This subsystem uses highly optimized, often Numba-accelerated, functions to perform the comparison.

Let's look at some simplified code concepts related to the matching logic, primarily found in `utils/matchup/bruteforce/`.

First, the data for matching is often wrapped in `Grid` objects:

```python
# From utils/matchup/bruteforce/Grid.py (simplified)
class Grid:
    def __init__(self,
        coordinates : np.ndarray | Callable, # The point locations
        resolutions : np.ndarray | Callable, # Defines the 'size' of points/spacing
        table : None | np.ndarray = None, # Original indices (important for later)
        # ... other metadata ...
    ):
        self._C = coordinates # Lazy access to coordinates
        self._R = resolutions # Lazy access to resolutions
        self._T = table # Keep track of original row indices
        # ... init other attributes ...

    @cached_property
    def coordinates(self) -> np.ndarray:
        # Loads or gets coordinates, ensures correct shape
        return self._C() if callable(self._C) else self._C

    @cached_property
    def resolutions(self) -> np.ndarray:
        # Loads or gets resolutions, formats them for matching
        R = np.atleast_1d(self._R() if callable(self._R) else self._R)
        return full_resolutions(self.coordinates, R) # Helper to handle different resolution formats
```

**Explanation:** The `Grid` class is a simple container for the coordinate data and resolution information from one source (or a combination of sources as matching progresses). Crucially, it keeps track of the `table` – the *original* indices from the source data. The matching process finds indices *within* the temporary coordinate arrays, but the final output needs to map back to the original data locations using this `table`. The `resolutions` are formatted by `full_resolutions` to represent the tolerance around each point in each dimension.

Next, pairs of `Grid` objects are matched using a `Pair` object:

```python
# From utils/matchup/bruteforce/Pair.py (simplified)
class Pair:
    def __init__(self, G1: Grid, G2: Grid):
        self.G1 = G1 # The first Grid
        self.G2 = G2 # The second Grid

    def match_indices(self, ...):
        # This method orchestrates the matching between G1 and G2

        # 1. Align grids if needed (e.g., tile for broadcast compatibility)
        g1_aligned = self.G1.tiled_align(self.G2)
        g2_aligned = self.G2.tiled_align(self.G1)

        # 2. Call the core Numba-accelerated brute force function
        # This function does the heavy lifting of comparing points
        ix1, ix2 = bruteforce_double(
            g1_aligned.coordinates, g2_aligned.coordinates,
            *g1_aligned.resolutions, *g2_aligned.resolutions,
            # ... other args like skip_dims, progress_bar ...
        )

        # 3. Map indices back to original data using the Grid's table
        # (Simplified: actual logic is handled when creating combined Grid)
        # original_ix1 = g1_aligned.table[ix1]
        # original_ix2 = g2_aligned.table[ix2]

        return ix1, ix2 # Indices within the *aligned* grids
```

**Explanation:** The `Pair` class represents the task of finding matches between two `Grid`s. Its `match_indices` method is the main entry point to the matching *algorithm*. It prepares the data and then calls the core Numba function.

The real speed comes from the Numba-accelerated brute force functions, like `bruteforce_double`, which perform the point-by-point comparisons. Let's look at the core check performed *inside* that Numba code:

```python
# From utils/matchup/bruteforce/utils/bruteforce_numba.py (simplified Numba function)
@nb.njit(...) # Numba decorator for performance
def check(x, xrl, xrr, y, yrl, yrr):
    """ Checks if point 'y' is within the tolerance range of point 'x'
        along all dimensions.

        x, y: The coordinates of the two points (vectors)
        xrl, xrr: Left/Right tolerance for point x (vectors)
        yrl, yrr: Left/Right tolerance for point y (vectors)
    """
    for i in range(len(x)): # Iterate through each dimension (lat, lon, time, etc.)
        # Check if y[i] is outside the range [x[i] - xrl[i], x[i] + xrr[i]]
        if (y[i] - yrl[i]) > x[i]: # Is y's lower bound above x's coordinate?
            # If y is strictly 'above' x, check if x is within y's tolerance
            if (x[i] - xrl[i]) > y[i]: # Is x's lower bound above y's coordinate? (Failure case 0)
                return i, 0 # Not a match, y is too low or x too high relative to tolerances
            if y[i] > (x[i] + xrr[i]): # Is y's coordinate above x's upper bound tolerance? (Failure case 1)
                return i, 1 # Not a match, y is too high relative to x's tolerance
        else:
            # If y[i] is not above x[i], check the other side
            if x[i] > (y[i] + yrr[i]): # Is x's coordinate above y's upper bound tolerance?
                 if (x[i] - xrl[i]) > y[i]: # Is x's lower bound above y's coordinate? (Failure case 0)
                    return i, 0 # Not a match
                 if y[i] > (x[i] + xrr[i]): # Is y's coordinate above x's upper bound tolerance? (Failure case 1)
                    return i, 1 # Not a match

    return -1, -1 # If the loop finishes, they match in all dimensions!
```

**Explanation:** This simplified Numba function `check` is the heart of the point comparison. It takes two points (`x` and `y`) and their respective left/right tolerances (`xrl`/`xrr`, `yrl`/`yrr`) in each dimension. It iterates through the dimensions (like latitude, then longitude, then time). For each dimension, it checks if `y` is within the acceptable range defined by `x`'s coordinate and `x`'s tolerance, and symmetrically if `x` is within `y`'s tolerance. If it finds *any* dimension where the points are too far apart according to the tolerances, it returns the dimension index and a flag indicating which side the failure occurred, allowing the main brute force loop to potentially skip ahead. If it checks all dimensions and finds no such failure, it returns `-1, -1`, indicating a match.

The `bruteforce_double` function (not shown in full here due to complexity) uses this `check` function within nested loops, efficiently iterating through potential pairs of points from the two grids (`G1` and `G2`). It leverages the fact that the grids are sorted (lexicographically) to avoid redundant checks by skipping large portions of the second grid whenever a mismatch is found in an earlier dimension. This is a key optimization that makes the brute force approach feasible for large datasets. It also adaptively switches which grid's points it iterates through based on which is smaller or offers better skipping opportunities.

After `bruteforce_double` returns the indices of matching points between `G1` and `G2`, the `Pair` object returns these indices. The `brute.py` module combines these indices and repeats the process if there are more than two grids, iteratively building up a table of indices that match across all grids in the `Blockset`.

Finally, the `Blockset` receives this table of matching indices and uses it to extract the windows of data (`Sample`s) that will be passed along for further processing.

The process of Data Matching is complex under the hood, relying on careful data organization (the `Grid` and `Pair` classes) and highly optimized algorithms (the Numba brute force functions) to efficiently find correspondences across your gridded data sources based on your defined tolerances.

## Conclusion

In this chapter, we learned about **Data Matching**, a critical subsystem in `crest` for aligning gridded data from multiple sources. We saw how it finds corresponding points based on coordinates and tolerances, essential for building meaningful `Sample`s that combine data from different origins. You now know that this complex matching process happens automatically within each `Blockset` when you call `Dataset.generate_samples()`, leveraging optimized algorithms like Numba-accelerated brute force search.

With your data organized into `Sample`s based on valid matches, the next step is to process these samples efficiently, often in batches for training or inference.

Ready to learn how `crest` handles processing chunks of samples? Let's move on to the next chapter: [Batching](04_batching_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)