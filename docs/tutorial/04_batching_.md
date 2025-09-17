# Chapter 4: Batching

Welcome back to the `crest` tutorial! In the [previous chapter](03_data_matching_.md), we delved into **Data Matching**, understanding how `crest` efficiently finds corresponding data points across different sources within a `Blockset` to create valid `Sample`s. By the end of that chapter, we had a `dask` array representing a collection of `Sample`s, ready to be used.

Now that we have our perfectly matched and prepared `Sample`s, how do we efficiently feed them into a machine learning model for training or inference? You can't just load millions or billions of samples into memory all at once!

Imagine you have a factory processing raw materials (your data) into finished products (your samples). You don't send each product individually to the assembly line (your model). Instead, you group them into standard packages of a fixed size and send those packages down a conveyor belt.

This is the concept of **Batching** in `crest`.

## What is Batching in CREST? (The Conveyor Belt Analogy)

The **Batcher** component in `crest` is like the **conveyor belt system** in your factory. It takes the collection of prepared `Sample`s and organizes them into fixed-size groups called **batches**. These batches are then efficiently delivered for processing.

Why do we need this?

1.  **Memory Efficiency:** Your computer's memory is limited. Batching allows you to process data in smaller, manageable chunks.
2.  **Computational Efficiency:** GPUs and modern processors are designed to perform parallel operations on chunks of data simultaneously. Processing a batch at once is much faster than processing individual samples one by one.
3.  **Model Training:** Machine learning models, especially deep learning models, are typically trained using batches of data to compute gradients and update weights efficiently.
4.  **Parallelism:** The `Batcher` can use multiple workers (processes or threads) to prepare these batches in parallel, significantly speeding up the data loading and preparation pipeline.

The `Batcher` takes the lazy data representation (like the `dask` array of samples from a `Dataset`) and manages the workload of computing those samples, assembling them into batches, and making them available to your main program loop.

## Your Use Case: Getting Batches for Training

Continuing our satellite and weather data example from Chapter 2 and 3:

*   We have satellite data (`soil_moisture`) and weather data (`temperature`, `precipitation`).
*   We configured a `Dataset` (Chapter 2) telling `crest` about these sources and the shape of the `Sample` windows we want (e.g., 3x3 pixels over 7 time steps).
*   `Dataset.generate_samples()` (orchestrating `Blockset`s and [Data Matching](03_data_matching_.md)) created a `dask` array, a *plan* to get samples that are valid across both data sources.

Now, we want to train a model that takes a batch of these combined satellite/weather samples as input. We need a way to:

1.  Tell `crest` how many samples should be in each batch (`batch_size`).
2.  Optionally, specify which features (`soil_moisture`, `temperature`, `precipitation`) should be extracted and organized into arrays within the batch, rather than getting raw `Sample` objects.
3.  Get batches one by one until all samples are processed (or forever if `repeat` is true).
4.  Ideally, have this process happen in the background using multiple workers so the model training isn't waiting for data.

This is exactly what the `Batcher` does.

## Using the Batcher Component

The `Batcher` class is the main tool you'll interact with. You create an instance of it, passing your `Dataset` (or list of `Dataset`s), the desired `batch_size`, and other configurations.

Here's how you'd create a `Batcher` for our use case, assuming `my_dataset` is set up as in Chapter 2:

```python
# Example: Creating a Batcher object
from crest.data.batching import Batcher
# Assume my_dataset is defined from Chapter 2

# Define how many samples per batch
my_batch_size = 64

# Define which features to extract and put into arrays in the batch
# This should match features you care about from your Datafiles
my_features = ['soil_moisture', 'temperature', 'precipitation']

# Create the Batcher
batcher = Batcher(
    dataset=my_dataset,
    batch_size=my_batch_size,
    features=my_features, # Tell it to structure batches with these features
    workers=4,          # Use 4 worker processes to prepare batches
    shuffle=True,       # Randomize the order of samples
    repeat=True,        # Keep yielding batches indefinitely (good for training)
    log_file='batcher.log', # Log Batcher activity
)

print(f"Created Batcher for dataset: {batcher.dataset}")
print(f"Batch size: {batcher.batch_size}")
print(f"Workers: {batcher.workers}")
```

**Explanation:**

*   We import the `Batcher` class.
*   We define our `batch_size` (64 samples per batch) and the list of `features` we want in the output batches.
*   We create a `Batcher` instance, passing `my_dataset`.
*   `batch_size`: The number of `Sample`s in each batch.
*   `features`: A list of feature names (strings). If provided, the batch will be structured as a dictionary (or nested structure matching the `features` list) where keys are feature names and values are NumPy arrays containing the data for that feature across all samples in the batch. If `features` is `None`, the batch will be a list of raw `Sample` objects. Using `features` is common for feeding data directly into models.
*   `workers`: The number of worker processes that will run in the background to compute samples and assemble batches. More workers can speed things up if your data loading/processing is a bottleneck and you have CPU cores available.
*   `shuffle`: Whether to randomize the order of samples before creating batches. Important for model training to avoid bias.
*   `repeat`: If `True`, the batcher will cycle through the entire dataset repeatedly, yielding batches indefinitely. Useful for training loops that run for many epochs.
*   `log_file`: Specifies a file to write detailed logs about the batching process, which can be helpful for debugging performance.

Once you have the `Batcher` object, you can iterate over it (or use `next()`) to get batches. The `Batcher` is designed to be used as a context manager, which ensures that its worker processes and resources are properly started and cleaned up.

```python
# Example: Iterating over the Batcher to get batches
# Assume 'batcher' is created as in the previous snippet

print("\nStarting to iterate batches...")

# Use the Batcher as a context manager
with batcher:
    # Loop to get batches
    for i, batch in enumerate(batcher):
        print(f"Received Batch {i+1}")

        # What does a batch look like?
        # If features was provided (like 'soil_moisture', 'temperature'):
        # batch will be a dictionary { 'soil_moisture': array([...]), 'temperature': array([...]), ... }
        # print(f"  Batch keys: {list(batch.keys())}")
        # print(f"  Shape of soil_moisture data: {batch['soil_moisture'].shape}") # Should be (batch_size, ...)

        # If features was None:
        # batch will be a list of Sample objects
        # print(f"  Batch is a list of {len(batch)} Sample objects.")

        # ... Feed this batch into your model ...

        # For demonstration, let's stop after a few batches
        if i >= 4: # Get 5 batches
            break

print("\nFinished getting batches.")

# The 'with' statement automatically calls batcher.close() when exiting
# This gracefully shuts down the worker processes and cleans up resources.
```

**Explanation:**

*   The `with batcher:` statement is important. It starts the worker processes when you enter the block and ensures they are stopped when you exit.
*   The `for batch in batcher:` loop pulls batches from the `Batcher`'s internal queue. The `Batcher` works in the background, using its workers to compute samples and assemble batches *ahead of time*. When you ask for a batch in the loop, it's usually already ready, minimizing waiting time.
*   The structure of `batch` depends on whether you provided the `features` parameter during `Batcher` initialization. If `features` was a list of strings, you get a dictionary mapping feature names to NumPy arrays. Each array has the shape `(batch_size, ...)`, where `...` is the spatial and temporal window shape you defined in your `Datafile`s (Chapter 2).
*   Inside the loop is where you would typically add your model's training or prediction step, using the `batch` data as input.

You can also use `Batcher.load_cached()` to pre-compute a fixed number of batches and save them to disk, loading them later. This is useful if data loading is slow and you want to avoid repeating it every time you run your code (e.g., during model development or hyperparameter tuning).

```python
# Example: Caching batches
# Assume 'dataset' is defined and you want to cache 1000 batches
from crest.data.batching import Batcher
from pathlib import Path

cache_file = Path('my_cached_batches.pkl')

# If the cache file doesn't exist, this will run the batcher and save 1000 batches
# If it exists, it will load the batches directly from the file
with Batcher.load_cached(
    dataset=my_dataset,
    cache_path=cache_file,
    batch_size=64,
    features=['soil_moisture', 'temperature'],
    n_samples=1000 * 64, # Cache enough samples for 1000 batches
    # Other Batcher args like workers, shuffle, repeat won't apply when loading
) as cached_batches:
    print(f"\nLoaded or created {len(cached_batches)} cached batches.")
    # Now you can iterate over the loaded list of batches
    for i, batch in enumerate(cached_batches):
        print(f"Using cached Batch {i+1}")
        # ... use the cached batch ...
        if i >= 4: # Just show a few cached batches
            break

print("Finished using cached batches.")
```

**Explanation:**

*   `Batcher.load_cached` acts as a factory. It checks if `cache_path` exists.
*   If `cache_path` does NOT exist, it creates a `Batcher` internally with the provided arguments (like `dataset`, `batch_size`, `features`, `workers`, `shuffle`, etc.), runs it to generate `n_samples`, saves the resulting list of batches to `cache_path` using `pickle`.
*   If `cache_path` DOES exist, it simply loads the list of batches from the file.
*   The `with ... as cached_batches:` block then gives you a standard Python list of batches (not a live `Batcher`), which you can iterate over. This is useful for quickly accessing pre-computed data.

## Behind the Scenes: How Batching Works

The `Batcher` coordinates a pipeline of data processing using multiple processes and threads.

Here's a simplified flow:

```{mermaid}
sequenceDiagram
    participant You as Your Code
    participant Batcher as crest.Batcher
    participant MPQueue as Multiprocessing Queue
    participant WorkerProcess1 as Worker Process 1
    participant WorkerProcessN as Worker Process N
    participant Dataset as crest.Dataset
    participant Blockset as crest.Blockset
    participant Sample as crest.Sample
    participant Batch as Batch Data

    You->>Batcher: Create Batcher(...)
    You->>Batcher: Enter context (`with batcher:`)
    Batcher->>MPQueue: Create Queue
    Batcher->>WorkerProcess1: Spawn Process (_queue_batches entry)
    Batcher->>WorkerProcessN: Spawn Process (_queue_batches entry)
    You->>Batcher: Iterate (`for batch in batcher:`)
    Batcher->>MPQueue: Get batch (waits if empty)

    loop While iterating
        WorkerProcess1->>Batcher: Get shared state (config, queue, flags)
        WorkerProcess1->>WorkerProcess1: Pick next Blockset to compute (using BlockConfig logic)
        alt Blockset available
            WorkerProcess1->>Dataset: Access lazy Dask graph (from generate_samples)
            WorkerProcess1->>Blockset: Trigger compute() on assigned Blockset task (Data Matching happens here!)
            Blockset-->>WorkerProcess1: Return computed Samples (NumPy array)
            WorkerProcess1->>WorkerProcess1: Extract features & format Samples
            WorkerProcess1->>WorkerProcess1: Group Samples into Batches
            WorkerProcess1->>MPQueue: Put Batch in Queue (waits if full)
        else No Blockset available
            WorkerProcess1->>WorkerProcess1: Wait / Check for exit signal
        end
        MPQueue-->>Batcher: Provide Batch when available
        Batcher-->>You: Yield Batch
    end

    You->>Batcher: Exit context (`with` finishes)
    Batcher->>WorkerProcess1: Signal exit (_exit_flag)
    Batcher->>WorkerProcessN: Signal exit (_exit_flag)
    WorkerProcess1->>WorkerProcess1: Clean up resources
    WorkerProcessN->>WorkerProcessN: Clean up resources
    Batcher->>Batcher: Join Worker Processes
    Batcher->>Batcher: Clean up queue
```

Key internal components and concepts:

*   **`_processes`**: When `workers > 0`, the `Batcher` creates a pool of independent Python processes. Each process runs the `_queue_batches` function.
*   **`_queue`**: A `multiprocessing.Queue` used for safe communication between the worker processes and the main process. Workers put completed batches into the queue, and the main process gets them from the queue.
*   **`_exit_flag`**: A `multiprocessing.Event` used to signal to all worker processes that they should shut down gracefully. Set when the main process exits the `with batcher:` block or encounters an error.
*   **`_generator`**: In the main process (or if `workers <= 0`), this cached property holds the main generator function that orchestrates the process. It calls `_generate_batches`.
*   **`_generate_batches`**: This function manages the loop over the available `Blockset`s (or tasks representing them). It uses `ThreadedFunction`s (`_block_tasks` and `_batch_tasks`) to parallelize computation and batch formation, even within a single process.
*   **`_block_tasks`**: A `ThreadedFunction` responsible for computing the *samples* from the lazy `Blockset` tasks returned by `Dataset.generate_samples()`. This is where the Dask computation is triggered, which in turn performs the [Data Matching](03_data_matching_.md).
*   **`_batch_tasks`**: A `ThreadedFunction` responsible for taking the computed samples, extracting features (if specified), shuffling them, and assembling them into batches.
*   **`_queue_batches`**: The entry point for each worker process. It continuously pulls tasks (representing groups of `Blockset`s) from an internal queue (managed within the worker by `_block_tasks`), processes them using its own threads (`_batch_tasks`), and puts the resulting batches into the main `_queue`.
*   **`BlockConfig`**: (Used internally, especially with `MultiBatcher` or complex configurations) These objects track the status of different configurations for sample generation (e.g., how many samples are queued, how many workers are using this config). Workers use this information to decide which type of block (or configuration) is "most needed" to ensure balanced batch generation, especially when dealing with configurations that might yield different numbers of samples or take varying amounts of time. (See `data/batching/BlockConfig.py`)

Let's look at a simplified view of the worker process entry point (`_queue_batches` in `data/batching/Batcher.py`):

```python
# From data/batching/Batcher.py (simplified _queue_batches)
def _queue_batches(self, process_ix, shared_attrs):
    # This code runs inside a worker process

    try:
        # 1. Store shared attributes (like the queue, exit flag, etc.)
        self.__dict__.update(shared_attrs | {'_pidx': process_ix})
        queue = shared_attrs['_queue'] # Access the main queue

        # 2. Initialize this worker's generator
        # This calls _generator(), which sets up _block_tasks and _batch_tasks
        for batch in self._generator:
            # 3. Put generated batches into the main queue
            while not self._exit: # Check exit flag while waiting for queue space
                try:
                    queue.put_nowait(batch) # Put batch without blocking
                    break # Successfully put batch, continue
                except Full:
                    # Queue is full, wait a bit and check exit flag again
                    time.sleep(WAIT_TIME)

            if self._exit: break # Exit loop if signaled

    except Exception as e:
        # Log any unexpected errors in the worker process
        self.error(f'{self.process_name} exception: {e}\n' +
                   f'{traceback.format_exc()}')
        raise # Re-raise to make the process exit with an error code

    finally:
        # Ensure graceful shutdown even if loop finishes or error occurs
        if self._exit:
             self.close(0, '_queue_batches') # Clean up worker resources
```

**Explanation:** This is a highly simplified view, but it shows the core loop of a worker process. It sets itself up with necessary shared state, starts its internal generator (`self._generator`), and then continuously pulls batches from that generator (which uses its own threads to compute Blocksets and form batches) and pushes them onto the main `_queue`. It includes logic to wait if the queue is full and to check the `_exit_flag` to know when to shut down.

The actual work of computing the samples and forming batches happens within the `_generator` method and the `_block_tasks` / `_batch_tasks` `ThreadedFunction`s, which trigger the Dask computation graph (including the [Data Matching](03_data_matching_.md)) on the assigned Blocksets.

## Conclusion

In this chapter, you learned about the **Batching** component in `crest` and its central role in preparing `Sample`s for efficient processing. You saw how the `Batcher` takes the lazy sample representation from a `Dataset` (produced after [Data Abstractions](02_data_abstractions_.md) and [Data Matching](03_data_matching_.md)) and delivers them as fixed-size batches. You learned how to create a `Batcher`, configure its size, features, and parallelism, and iterate over it to get batches, leveraging the context manager for resource management. We also touched on the caching feature for saving pre-computed batches.

With your data now flowing in nicely structured batches, you're ready to introduce the component that will consume this data: the model.

Let's move on to the next chapter: [Model](05_model_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)