# Chapter 2: Data Abstractions

Welcome back to the `crest` tutorial! In the [previous chapter](01_configuration_.md), we learned how to set up the foundational "recipe book" for our `crest` project using the `Config` component. This recipe tells our project where to find things and how to behave.

Now that we know *where* to look for our data (thanks to the `Config`), the next big question is: how do we *represent* and *handle* that data within our project?

Imagine you're working with data that's massive, perhaps spread across multiple files or even different types of data sources (like satellite images, weather model outputs, or sensor readings). You can't load it all into your computer's memory at once. You need a structured way to:

1.  Know *what* data sources you have.
2.  Load just the parts you need.
3.  Organize data from different sources that correspond to the same time and place.
4.  Break the data into smaller, manageable pieces for processing or training a model.
5.  Prepare those pieces into the exact format your model expects.

This is where `crest`'s **Data Abstractions** come in.

## What are Data Abstractions in CREST? (The Library Analogy)

Think of your data as a vast **library**. This library contains many **books**, each covering a specific topic or containing data from one source. To make this library useful, it needs organization!

`crest` uses several classes to represent your data at different levels of organization, much like a library system:

*   **`Datafile`:** Represents a single "book" in your library. It's one specific source of data (like one Zarr file, one satellite product).
*   **`Dataset`:** Represents the "library catalog" or a collection of related "books". It knows about multiple `Datafile`s and helps coordinate them.
*   **`Block`:** Represents a "chapter" within a single "book". It's a spatial/temporal chunk of data from *one* `Datafile`.
*   **`Blockset`:** Represents a collection of "chapters", one from *each* relevant "book" in the catalog, all related to the same spatial/temporal area.
*   **`Sample`:** Represents a single "paragraph" within a "chapter", ready to be read (used by a model). It's a specific window of data from a `Blockset` representing one training example.
*   **`SampleSet`:** A collection of "paragraphs" or `Sample`s.

These classes provide different "views" of your data, allowing `crest` to handle complexity, manage memory, and enable parallel processing efficiently.

## Your Use Case: Combining Satellite and Weather Data

Let's consider our simple use case: we have satellite data (e.g., soil moisture) in one file and weather data (e.g., temperature, precipitation) in another file. Both cover the same region and time period, but they might have different spatial resolutions or slight differences in their time stamps.

We want to:

1.  Load both data sources.
2.  Align them so we can get satellite and weather data for the *same* location and time.
3.  Break the combined data into chunks.
4.  Extract small windows of data (like a 3x3 grid of pixels over 7 days) around specific locations to use as input for a model.

This is a common task in Earth observation and climate science, and `crest`'s data abstractions are built for this.

## Level 1: The `Datafile` (Your Book)

The `Datafile` class is your entry point for a single data source. You tell it where your data lives, what variables (features) you're interested in, and any initial filtering or windowing requirements.

Imagine your satellite data is in a Zarr store at `/data/satellite/soil_moisture.zarr` and your weather data is at `/data/weather/era5_surface.zarr`.

Here's how you'd represent them as `Datafile` objects:

```python
# Example: Creating Datafile objects
from crest.data.loading import Datafile
from pathlib import Path

# Define paths (replace with your actual paths or use config)
sat_path = Path('/data/satellite/soil_moisture.zarr')
weather_path = Path('/data/weather/era5_surface.zarr')

# Create a Datafile for satellite data
satellite_datafile = Datafile(
    location=sat_path,
    features=['soil_moisture'], # We only want the soil moisture variable
    extent={'latitude': [10, 20], 'longitude': [5, 15]}, # Limit to a specific region
    window_depth={'time': 3}, # Include 3 steps before and 3 after the center time
    # ... other optional parameters like invalid_value, preprocessors
)

# Create a Datafile for weather data
weather_datafile = Datafile(
    location=weather_path,
    features=['temperature', 'precipitation'], # We want these variables
    # We can use the same extent if needed, or let Dataset align them
    window_depth={'time': (5, 1)}, # 5 steps before, 1 after the center time
    # ... other parameters
)

print(f"Created Datafile: {satellite_datafile}")
print(f"Created Datafile: {weather_datafile}")
```

**Explanation:**

*   We import the `Datafile` class.
*   We create two `Datafile` instances, one for each data source.
*   `location`: Tells the `Datafile` where to find the data (e.g., a path to a Zarr store).
*   `features`: Specifies which variables (like 'soil\_moisture', 'temperature') from that source we care about.
*   `extent`: Lets us specify a geographic or temporal box to limit the data loaded.
*   `window_depth`: Crucially, this tells the `Datafile` what shape of "paragraph" (sample window) it should expect for its data along each dimension. `{'time': 3}` means `(3, 3)` (3 elements left, 3 right) for a total window size of 7 along the 'time' dimension. `{'time': (5, 1)}` is more specific, requesting 5 elements to the left and 1 to the right, for a total of 7. This is configured *per Datafile*.

The `Datafile` doesn't load all the data into memory immediately. It uses libraries like `xarray` and `dask` under the hood to keep things "lazy" – it only loads data when it's absolutely needed.

## Level 2: The `Dataset` (The Library Catalog)

Once you have your individual `Datafile`s (books), you group them together into a `Dataset` (the catalog). The `Dataset` understands that these `Datafile`s are related and need to be used together.

```python
# Example: Creating a Dataset
from crest.data.loading import Dataset
# Assuming satellite_datafile and weather_datafile are already created

# Create a Dataset containing our two Datafiles
my_dataset = Dataset([satellite_datafile, weather_datafile])

print(f"\nCreated Dataset containing {len(my_dataset)} Datafiles.")
# You can access the individual datafiles like items in a list
# print(my_dataset[0])
# print(my_dataset[1])
```

**Explanation:**

*   We import the `Dataset` class.
*   We create a `Dataset` instance, passing a list of our `Datafile` objects to it.
*   The `Dataset` now manages this collection. It knows their individual configurations, features, and windowing requirements.

The `Dataset` is where the coordination happens. If your `Datafile`s have slightly different coordinate systems or resolutions, the `Dataset` will figure out how they relate to each other. It's also responsible for deciding how to break the combined data into manageable chunks for parallel processing.

## Generating Samples: The `generate_samples` Method

The core task of the `Dataset` is to find and prepare the individual data "paragraphs" (Samples) that your model will use. This is done with the `generate_samples` method. This single method orchestrates the entire process of:

1.  Making sure all `Datafile`s agree on common dimensions.
2.  Calculating an efficient way to chunk the data into `Block`s.
3.  Creating `Blockset`s (groups of corresponding `Block`s).
4.  Within each `Blockset`, finding the exact spatial/temporal locations where data exists and is valid *across all contained Block*s (this is the "matching" step, covered in more detail in the next chapter).
5.  For each matching location, extracting the specified window of data from each `Block` to form a `Sample`.
6.  Collecting these `Sample`s into a `dask` array, ready for processing or batching.

```python
# Example: Generating samples from the Dataset
# Assuming my_dataset is already created

print("\nGenerating samples...")

# Call generate_samples. This kicks off the heavy lifting.
# compute=False means it returns a Dask graph, not the actual data yet.
# verbose=True gives you feedback on the process.
lazy_samples_array = my_dataset.generate_samples(compute=False, verbose=True)

print(f"Generated a lazy dask array of samples: {lazy_samples_array}")
print(f"This array represents {lazy_samples_array.npartitions} partitions (tasks).")

# To actually get the data, you would call .compute() on the array
# print("\nComputing samples (this might take a while)...")
# all_my_samples = lazy_samples_array.compute()
# print(f"Finished computing. Total samples found: {len(all_my_samples)}")
```

**Explanation:**

*   Calling `generate_samples` is the key action.
*   `compute=False` is typical in `crest` workflows, as it returns a `dask` array. This array represents the *plan* to compute the samples, not the samples themselves. The actual computation happens later, often handled by the [Batching](04_batching_.md) component.
*   `verbose=True` helps you see what the `Dataset` is doing behind the scenes (like figuring out chunking).

The result is a `dask` array. Each element in this array, when computed, will be a `Sample` object.

## Levels 3 & 4: `Block` and `Blockset` (Chapters and Grouped Chapters)

When `generate_samples` is called, the `Dataset` first figures out how to split each `Datafile` (book) into `Block`s (chapters). It tries to find a common chunking strategy that works well for all `Datafile`s in the `Dataset`.

```{mermaid}
graph TD
    A[Dataset] --> B{Decide Chunking};
    B --> C[Datafile 1];
    B --> D[Datafile 2];
    C --> C1[Block 1.1];
    C --> C2[Block 1.2];
    C --> C3[Block 1.3];
    D --> D1[Block 2.1];
    D --> D2[Block 2.2];
    D --> D3[Block 2.3];
    C1 --> E{Group Blocks};
    D1 --> E;
    C2 --> F{Group Blocks};
    D2 --> F;
    C3 --> G{Group Blocks};
    D3 --> G;
    E --> E1[Blockset 1];
    F --> F1[Blockset 2];
    G --> G1[Blockset 3];
```

*   Each `Block` holds a specific chunk of a single `Datafile`'s data, potentially with some overlap at the edges to ensure samples near boundaries can still be formed correctly.
*   `crest` then groups these `Block`s into `Blockset`s. A `Blockset` contains one `Block` from each `Datafile` that corresponds to the same general area. This is the unit that parallel processing will work on.

The main job of a `Blockset` is to figure out, *within its limited spatial/temporal area*, which exact locations have valid data available from *all* its contained `Block`s. These locations are where samples can be generated. The process of finding these matching locations is central to `crest` and happens inside the `Blockset`.

## Levels 5 & 6: `Sample` and `SampleSet` (Paragraphs and Collections of Paragraphs)

Once a `Blockset` has identified the valid matching locations, it extracts the specified "window" (`window_depth`) of data from each of its `Block`s for every matching location. Each such extracted window becomes part of a single `Sample` object.

```{mermaid}
flowchart LR
    Blockset -- Finds Matches --> MatchedLocations(Matched Locations)
    MatchedLocations -- For each location --> ExtractWindow1(Extract Window from Block 1)
    MatchedLocations -- For each location --> ExtractWindow2(Extract Window from Block 2)
    ExtractWindow1 & ExtractWindow2 -- Combine --> Sample(Sample Object)
    Sample -- Collect --> SampleSet(SampleSet)
```

*   A `Sample` object is lightweight; it holds the data windows for one spatial/temporal point, pulled from all the relevant `Datafile`s. This is typically the format expected by a machine learning model.
*   A `SampleSet` is simply a collection of these `Sample` objects, usually the output of processing a single `Blockset`.

When you compute the `dask` array returned by `generate_samples`, `crest` works through the `Blockset`s in parallel. Each `Blockset` computes its `SampleSet`, and these `SampleSet`s are concatenated into the final array of `Sample`s.

## Behind the Scenes: How Abstractions Work Together

Let's look at a simplified flow of how `Dataset.generate_samples` uses these abstractions.

```{mermaid}
sequenceDiagram
    participant You as Your Code
    participant Dataset as crest.Dataset
    participant Datafile as crest.Datafile
    participant Blockset as crest.Blockset
    participant Block as crest.Block
    participant Sample as crest.Sample
    participant SampleSet as crest.SampleSet

    You->>Dataset: Create Dataset([Datafile 1, Datafile 2])
    You->>Dataset: generate_samples()
    Dataset->>Datafile: Load metadata / Initial checks (e.g., .data property)
    Datafile-->>Dataset: Provide metadata (dims, shapes, etc.)
    Dataset->>Dataset: Calculate optimal chunking/blocking strategy
    Dataset->>Datafile: Apply new chunking (e.g., update_blocks)
    Datafile-->>Dataset: Return rechunked Datafile objects
    Dataset->>Datafile: Create Block dask arrays (apply_overlap)
    Datafile-->>Dataset: Return lazy Block dask arrays
    Dataset->>Dataset: Group Block arrays into Blockset tasks
    Dataset->>Blockset: Delayed: Create Blockset()
    Blockset->>Block: Delayed: Create Block()
    Block-->>Blockset: Provide computed Block data/metadata (on demand)
    Blockset->>Blockset: Find matching valid locations within Blocks (find_neighbors)
    Blockset->>Block: Extract data window for each match (extract)
    Block-->>Blockset: Provide extracted data windows
    Blockset->>SampleSet: Create SampleSet([windows])
    SampleSet->>Sample: Create Sample() (on demand when accessed/computed)
    Sample-->>SampleSet: Provide Sample object
    SampleSet-->>Blockset: Provide SampleSet (or array of Samples)
    Blockset-->>Dataset: Return SampleSet task result
    Dataset-->>You: Return lazy dask Array of SampleSets/Samples
```

This diagram illustrates the flow. `Dataset` orchestrates the initial setup and chunking, creating `Blockset`s. Each `Blockset` then operates somewhat independently (often in parallel), pulling data from its component `Block`s to find matches and create `Sample`s.

Let's peek at some simplified code snippets (referencing provided files):

**`Datafile` Initial Loading:**

```python
# From data/loading/Datafile.py (simplified)
class Datafile:
    def __init__(self, location, features=[], ...):
        self.location = location
        self.features = sorted(features)
        # ... store other config ...
        self._config = self.__getstate__() # Store config for hashing/pickling

    @cached_property
    def _raw_data(self):
        """ Loads the data lazily using xarray.open_zarr or similar """
        print(f"Loading data from {self.location}...")
        if isinstance(self.location, xr.Dataset):
            return self.location
        # Uses fsspec/zarr/xarray to open the data source
        return xr.open_zarr(self.location, **self._kwargs)

    @cached_property
    def data(self) -> xr.DataArray:
        """ Applies extent, features, preprocessors, creates valid_mask etc. """
        raw = self._raw_data # Accessing this triggers the lazy load if needed
        # ... apply extent, select features, apply preprocessors ...
        # ... create/process 'valid_mask' ...
        return processed_data_array # Simplified return
```

**Explanation:** The `Datafile` stores its configuration in `__init__`. The actual loading (`_raw_data`) and initial processing (`data`) are done using `cached_property`, meaning they only happen the *first* time you access `.data`. This keeps things fast if you just create the object but don't need the data immediately.

**`Dataset` Initialization:**

```python
# From data/loading/Dataset.py (simplified)
class Dataset(BaseSet): # Inherits from BaseSet for apply/map functionality
    def __init__(self, locations, **kwargs):
        # Create Datafile objects for each location
        self.container = list(map(partial(Datafile.load, **kwargs), locations))
        # Assign an index to each datafile
        for i, datafile in enumerate(self): datafile.dataset_index = i
        # ... other setup ...
```

**Explanation:** The `Dataset`'s `__init__` is simple. It takes a list of locations (which can be paths or existing `Datafile` objects) and ensures they are all converted into `Datafile` instances, storing them in its `container` list.

**`Dataset.generate_samples` High-Level Steps:**

```python
# From data/loading/Dataset.py (simplified)
class Dataset:
    # ... __init__ and other methods ...

    def generate_samples(self, ..., compute=True, ...):
        # 1. Ensure all Datafiles are aware of all dimensions
        self.ensure_dims(...) # Adds 'virtual' dimensions if needed

        # 2. Determine number of blocks per dimension and element overlaps
        # Calculates chunking strategy (autochunk) and overlap needed

        # 3. Rechunk data based on the required number of blocks
        # Calls update_blocks or update_chunks on each Datafile

        # 4. Apply overlap, create Blocksets with one Block per Datafile
        prepped = self.apply_overlap(...) # Creates lazy Block objects
        create_set = partial(Blockset, ...)
        # Groups Blocks into Blocksets (delayed tasks)
        blocksets = map(dask.delayed(create_set), zip(*prepped, strict=True))

        # Find all samples in parallel across the created blocksets
        get_return = lambda bs, obj=return_objs: bs if obj else bs.find_matches
        # Return list of delayed tasks (either Blockset objects or their find_matches method)
        tasks = list(map(get_return, blocksets))

        if compute:
            # Trigger dask computation
            samples = da.hstack( da.compute(*[t() for t in tasks]) )
            return samples
        else:
            return tasks # Return the list of delayed tasks
```

**Explanation:** This snippet shows the sequence of internal calls within `generate_samples`. It coordinates across all `Datafile`s (`ensure\_dims`), calculates how to chunk them, tells the `Datafile`s to rechunk, creates the necessary `Block`s with overlap, groups them into `Blockset` tasks, and finally defines the dask graph (`tasks`) that, when computed, will produce the samples.

## Conclusion

In this chapter, we explored `crest`'s core **Data Abstractions**. We learned how:

*   A `Datafile` represents a single data source with its own configuration.
*   A `Dataset` manages a collection of `Datafile`s and coordinates their use.
*   `Dataset.generate_samples` is the key method to orchestrate the process of finding valid samples.
*   Data is internally broken down into `Block`s (chunks of `Datafile`s) and grouped into `Blockset`s for parallel processing.
*   Finally, `Sample`s (individual data examples) are extracted from `Blockset`s and collected into `SampleSet`s, typically represented as a `dask` array.

These abstractions provide a powerful and flexible way to handle complex, large-scale datasets. Now that we understand how `crest` represents and organizes data, the next crucial step is to understand *how* it finds the exact spatial and temporal points that are valid across *all* our different data sources.

Let's move on to the next chapter: [Data Matching](03_data_matching_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)