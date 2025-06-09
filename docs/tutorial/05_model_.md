# Chapter 5: Model

Welcome back to the `crest` tutorial! In the [previous chapter](04_batching_.md), we explored the **Batching** component and learned how it efficiently prepares and delivers chunks of your carefully organized and matched data ([Data Abstractions](02_data_abstractions_.md), [Data Matching](03_data_matching_.md)) in fixed-size batches, ready for processing.

Now that the data is flowing smoothly in manageable packages, we need something to actually *do* something with those packages! This is where the **Model** component comes in.

## What is a Model in CREST? (The Conductor Analogy)

Think back to our orchestra analogy. We have the sheet music ([Model Graph (HTG)](06_model_graph__htg__.md) - coming in the next chapter!), the individual musicians and sections (the layers or operations within the model), and the audience providing data (our batches). But who brings it all together? Who interprets the score and directs the musicians to play in harmony and at the right time?

That's the role of the **Model** component in `crest`.

The `crest.Model` class is like the **conductor of the orchestra**. It takes the musical score (a computational graph, specifically a [HierarchalTensorGraph (HTG)](06_model_graph__htg__.md)), understands how all the pieces fit together, and then directs the performance. In the context of machine learning or data processing, the "performance" is running the computation defined by the graph on your data batches.

Specifically, the `Model` class in `crest`:

1.  **Wraps the Graph (HTG):** It holds the blueprint ([Model Graph (HTG)](06_model_graph__htg__.md)) that defines the sequence of operations (like neural network layers, simple math operations, etc.).
2.  **Builds a Backend Model:** It translates this abstract graph into a concrete computational model using a specific library or "backend" (like TensorFlow/Keras).
3.  **Provides Standard Methods:** It gives you simple methods like `compile`, `fit` (for training), and `predict` (for making inferences), hiding the complexity of the backend framework.
4.  **Manages Data Flow:** It works seamlessly with the [Batching](04_batching_.md) component to feed data batches into the backend model efficiently.

This separation means you can design your data processing or machine learning architecture using `crest`'s graph abstractions and then use the `Model` class to run it with a powerful backend without getting bogged down in backend-specific data pipeline details.

## Your Use Case: Training and Predicting with a Batcher

Let's continue our satellite and weather data example. We have:

*   A `Dataset` ([Data Abstractions](02_data_abstractions_.md)) set up to provide samples combining satellite soil moisture and weather temperature/precipitation.
*   A `Batcher` ([Batching](04_batching_.md)) configured to deliver batches of these samples, formatted nicely as NumPy arrays for the different features (`soil_moisture`, `temperature`, `precipitation`).
*   A conceptual model architecture (the [Model Graph (HTG)](06_model_graph__htg__.md)) that defines how to take these input features and produce an output (e.g., predict soil moisture for the next day).

Now, we need the `Model` component to:

1.  Load or create the model based on the defined graph.
2.  Prepare it for training (e.g., specify the optimizer and loss function).
3.  Connect it to the `Batcher` and train it using the data batches.
4.  Connect it to a `Batcher` with new data and generate predictions.

## Using the Model Component

The primary way to interact with the `Model` component is through the `crest.model.Model` class. You typically load a model defined by a [Model Graph (HTG)](06_model_graph__htg__.md), compile it, and then use its `fit` or `predict` methods.

First, you need to load or define your model's graph. In this chapter, we'll assume the graph is already defined (details on creating it are in [Chapter 6: Model Graph (HTG)](06_model_graph__htg__.md)). The `Model` class has a `load` method for this.

```python
# Example: Loading a CREST Model
from crest.model import Model
from pathlib import Path

# Assume you have a directory 'my_model_files'
# containing 'htg.graph.json' and backend model files
model_directory = 'my_model_files'
model_backend_type = 'keras' # Or 'pickle' etc.

# Load the CREST Model object
# This reads the graph and loads the backend model structure
try:
    my_crest_model = Model.load(model_directory, model_backend_type)
    print(f"Successfully loaded model: {my_crest_model.name}")
except Exception as e:
    print(f"Could not load model: {e}")
    # In a real script, handle this error
```

**Explanation:**

*   We import the `Model` class.
*   `Model.load()` is a static method that takes the path to where your model files are saved and the type of backend model saved there (`'keras'` is common).
*   It reads the graph definition (`htg.graph.json`) and loads the corresponding backend model.
*   The result is a `crest.model.Model` object, `my_crest_model`.

Once you have a `Model` object, you need to `compile` it. This step is specific to the backend (like Keras) and involves setting up things like the optimizer, loss function, and metrics that will be used during training.

```python
# Example: Compiling the Model
# Assume my_crest_model is loaded

print("\nCompiling the model...")

# The compile method takes backend-specific arguments (like Keras compile args)
my_crest_model.compile(
    optimizer='adam',
    loss='mse',        # Mean Squared Error
    metrics=['mae']    # Mean Absolute Error
)

print("Model compiled.")
# The underlying Keras model is now configured for training/evaluation
```

**Explanation:**

*   The `compile` method prepares the underlying backend model for training or evaluation.
*   The arguments you pass (`optimizer`, `loss`, `metrics`) are typically passed directly to the backend's compile function (e.g., `tf.keras.Model.compile`).
*   Calling `compile` also implicitly calls the internal `build` method if the model hasn't been built yet, which creates the backend model structure from the graph.

Now, let's train the model using a `Batcher` created from your dataset ([Chapter 4: Batching](04_batching_.md)). The `fit` method handles this.

```python
# Example: Training the Model using a Batcher
# Assume my_crest_model is compiled
# Assume my_dataset is created from Chapter 2
# Assume my_batch_size and my_features are defined from Chapter 4

print("\nStarting model training...")

# The fit method can take a Dataset or a Batcher
# It will create a Batcher internally if you pass a Dataset
# We'll pass our dataset config and let fit make the Batcher
# Note: In practice, you might pass a pre-configured Batcher if you need more control
my_crest_model.fit(
    dataset=my_dataset, # Or a pre-made Batcher instance
    epochs=5,           # Train for 5 epochs
    batch_size=my_batch_size, # Use the specified batch size
    # steps_per_epoch=... # Often needed with Batchers/generators
    # validation_data=... # Optional validation dataset/batcher
    # callbacks=[...]     # Optional Keras callbacks
)

print("Model training finished.")
```

**Explanation:**

*   The `fit` method takes your `dataset` (which can be a `Dataset`, `StructuredDataset`, `Batcher`, or dictionary) and training parameters.
*   It calls its internal `_make_batcher` method to ensure it's working with a `Batcher`. It configures this internal batcher based on `batch_size`, required `features` (inputs and outputs of your model's graph), `shuffle`, and `repeat=True` (typical for training).
*   Then, it calls the underlying backend model's `fit` method (e.g., `self.model.fit(training_batcher, ...)`) using the generated batches.

Finally, let's use the trained model to make predictions on new data using a `Batcher`. The `predict` method is used here.

```python
# Example: Making predictions
# Assume my_crest_model is trained
# Assume my_prediction_dataset is another Dataset instance
# Assume my_batch_size is defined

print("\nStarting model prediction...")

# The predict method can also take a Dataset or a Batcher
# We'll pass our prediction dataset config
# We specify the features needed for prediction (model inputs)
my_predictions = my_crest_model.predict(
    dataset=my_prediction_dataset, # Or a pre-made Batcher
    batch_size=my_batch_size,      # Use the specified batch size
    # steps=... # Number of batches to predict on (use exhaust=True to predict on all)
    # coords=['latitude', 'longitude', 'datetime'] # Optional: Include these features in the output
)

print("Prediction complete.")
# my_predictions is likely a dictionary containing the model's output features
# and optionally any requested 'coords'
# print(f"Prediction output keys: {list(my_predictions.keys())}")
```

**Explanation:**

*   Similar to `fit`, `predict` takes your `dataset` and prediction parameters.
*   It uses `_make_batcher` to get a `Batcher`. This batcher is configured to provide the model's *input* features and any additional `coords` you requested. `shuffle` is typically `False` for prediction to maintain order. `repeat` is `False` by default.
*   It then iterates through batches from this batcher and calls the underlying backend model's `predict_on_batch` method for each batch, collecting the results. There's also a `predict_exhaust` which collects all batches first and then calls `predict` once, useful if your backend benefits from seeing all data at once or if you need the results concatenated. The `predict` method shown above iterates batch by batch.
*   The result `my_predictions` is typically a dictionary mapping output feature names to arrays of predictions, potentially including the requested coordinate data.

The `GriddedModel` class (seen in the provided code snippets) is an example of a higher-level component that *uses* the `Model` class, specifically its `predict_on_batch` method, within a loop that also handles iterating through time steps (`get_ts_extents`), initializing datasets (`init_dataset`), and archiving results using the [Archiver](09_archiver_.md). This shows how the core `Model`'s methods can be integrated into more complex workflows.

## Behind the Scenes: How the Model Works

The `crest.model.Model` class primarily acts as an interface layer between your `crest` workflow, the abstract [Model Graph (HTG)](06_model_graph__htg__.md), and the chosen backend library (like TensorFlow/Keras).

Here's a simplified flow for how the `Model` orchestrates prediction:

```{mermaid}
sequenceDiagram
    participant You as Your Code
    participant CREST_Model as crest.Model
    participant CREST_Batcher as crest.Batcher (Chapter 4)
    participant Backend_Model as Backend Model (e.g., tf.keras.Model)
    participant Dataset as Dataset (Chapter 2)

    You->>CREST_Model: Create Model(graph) OR Model.load(...)
    You->>CREST_Model: compile(...)
    CREST_Model->>CREST_Model: Build Backend Model structure (from graph)
    CREST_Model->>Backend_Model: Compile Backend Model (optimizer, loss, etc.)
    CREST_Model-->>You: Model compiled

    You->>CREST_Model: predict(dataset, ...)
    CREST_Model->>CREST_Model: _make_batcher(dataset)
    CREST_Model->>CREST_Batcher: Create/Configure Batcher(dataset, ...)
    CREST_Batcher-->>CREST_Model: Return Batcher instance
    CREST_Model->>CREST_Batcher: Enter context (`with batcher:`)
    loop For each batch from Batcher
        CREST_Batcher->>CREST_Batcher: Prepare next batch (uses Dask, Data Matching etc.)
        CREST_Batcher-->>CREST_Model: Yield Batch (dictionary of arrays)
        CREST_Model->>Backend_Model: predict_on_batch(batch_inputs)
        Backend_Model-->>CREST_Model: Return prediction results
        CREST_Model->>CREST_Model: Collect results, maybe add coords
    end
    CREST_Batcher->>CREST_Batcher: Exit context (`with batcher:`) - cleans up workers
    CREST_Model-->>You: Return collected predictions
```

Key internal aspects of the `Model` class (`model/Model.py`):

*   **`__init__`:** This is where the `graph` (a [HierarchalTensorGraph](06_model_graph__htg__.md)) is stored. It also takes the `inputs` and `outputs` defined in the graph and translates them into backend-specific input specifications (like Keras `Input` layers).

    ```python
    # From model/Model.py (simplified __init__)
    class Model(BaseModel):
        def __init__(self, graph: TensorGraph, **kwargs):
            self.graph = graph # Store the HTG graph
            self.model = None # The backend model (e.g., Keras) is built later
            self.name = graph.name

            # Translate graph inputs (TensorSpec) into backend inputs (Keras Input layers)
            self.inputs = {}
            for k, v in self.graph.inputs.items():
                if (not v is None):
                    # Check if it's a TensorSpec or already backend specific
                    # Create Keras Input layer matching shape/dtype
                    self.inputs[k] = tf.keras.Input(shape=v.shape[1:], dtype=v.dtype, name=k)
                else:
                    self.inputs[k] = None

            # Call the graph with the inputs to get the output structure
            # This connects the layers/nodes defined in the graph
            self.outputs = self.graph(self.inputs)
            # ... rest of init ...
    ```

    **Explanation:** The initializer stores the `graph` and prepares the necessary inputs based on the graph's definition. The line `self.outputs = self.graph(self.inputs)` is key: it passes the backend input objects through the structure defined by the `HierarchalTensorGraph`, creating the computational flow.

*   **`build`/`compile`:** The `build` method creates the actual backend `Model` instance (e.g., `tf.keras.Model`) using the inputs and outputs defined during `__init__`. The `compile` method calls `build` and then calls the backend model's own `compile` method with the specified optimizer, loss, etc.

    ```python
    # From model/Model.py (simplified build/compile)
    class Model: # ... other methods ...
        def build(self, _internal=False, **kwargs):
            # Creates the backend model instance (e.g., Keras Model)
            self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        def compile(self, show_summary: bool = False, **kwargs):
            # Calls build to create the backend model
            self.build(_internal=True)
            # Calls the backend model's compile method
            self.model.compile(**kwargs)
            # ... optional summary and checks ...
    ```

    **Explanation:** `build` is where `self.model` (the Keras model) is assigned. `compile` is the user-facing method that triggers the setup for training/prediction.

*   **`_make_batcher`:** This internal helper method is used by `fit` and `predict` to standardize the data input. It checks if the provided `dataset` is already a `Batcher`, `Dataset`, `StructuredDataset`, or dictionary, and creates a `Batcher` instance accordingly ([Chapter 4: Batching](04_batching_.md)).

    ```python
    # From model/Model.py (simplified _make_batcher)
    class Model: # ... other methods ...
        def _make_batcher(self, dataset, **kwargs) -> Batcher:
            if isinstance(dataset, Batcher):
                return dataset # Already a batcher, just return it
            if isinstance(dataset, dict):
                # Handle dict input (create StructuredDataset then Batcher)
                # ... simplified ...
                return Batcher(StructuredDataset(...), **kwargs)
            if isinstance(dataset, (Dataset, StructuredDataset)):
                # Handle Dataset/StructuredDataset input
                return Batcher(dataset, **kwargs)
            raise ImproperModelError(...) # Raise error for unsupported types
    ```

    **Explanation:** This method simplifies the user interface for `fit` and `predict`, allowing them to accept various data representations, while ensuring that internally they always work with a `Batcher`.

*   **`fit`/`predict`:** These methods primarily configure the necessary `Batcher` (via `_make_batcher`), handle backend-specific parameters (`kwargs`), and then call the corresponding method on the underlying backend model (`self.model.fit` or `self.model.predict`/`self.model.predict_on_batch`). They also manage the context of the `Batcher` using the `with` statement for proper resource cleanup.

    ```python
    # From model/Model.py (simplified fit/predict)
    class Model: # ... other methods ...
        def fit(self, dataset: Dataset | Batcher | StructuredDataset, **kwargs):
            # Configure Batcher for training (features=inputs+outputs, repeat=True)
            train_kwargs = { 'features': [list(self.inputs), list(self.outputs)], 'repeat': True, ... }
            training_batcher = self._make_batcher(dataset, **train_kwargs)

            # Use the batcher in a context and call backend model's fit
            with training_batcher as data:
                self.model.fit(data, **kwargs)

        def predict(self, dataset: Dataset | StructuredDataset | Batcher | dict, coords=[], **kwargs) -> dict:
            # Configure Batcher for prediction (features=inputs + coords, repeat=False)
            batch_kwargs = { 'features': coords + list(self.inputs), 'shuffle': False, 'repeat': False, ... }
            batcher = self._make_batcher(dataset, **batch_kwargs)

            pred = [] # List to collect predictions
            lbls = None # To store requested coordinate data

            # Iterate through batches from the batcher
            with batcher as data:
                # ... loop through batches ...
                batch = next(data) # Get a batch
                # ... optional: extract coords from batch into lbls ...
                # Call backend model's predict_on_batch (or predict)
                pred_batch = self.model.predict_on_batch(batch) # Or self.model.predict(batch)
                # ... optional: re-attach coords to pred_batch ...
                # ... append pred_batch to pred list ...
            # ... concatenate results from pred list ...
            return concatenated_predictions # Return dictionary of results
    ```

    **Explanation:** `fit` and `predict` demonstrate how the `Model` class connects the data pipeline (the `Batcher`) to the backend computation engine (`self.model`). They manage the flow of data and results, handling details like extracting the correct features from the batch for the backend model.

The `Model` class thus provides a clean, backend-agnostic interface for compiling, training, and predicting with models defined by the [Model Graph (HTG)](06_model_graph__htg__.md).

## Conclusion

In this chapter, you learned about the **Model** component in `crest`. You saw how this class acts as the conductor, wrapping a computational graph ([Model Graph (HTG)](06_model_graph__htg__.md)) and translating it into a backend-specific model (like Keras). You learned how to load, compile, train (using `fit`), and make predictions (using `predict`) with a `crest.model.Model` object, leveraging the [Batching](04_batching_.md) component for efficient data handling.

You now have a solid understanding of how `crest` manages your data and prepares it for processing, and how the `Model` component brings the computational architecture to life using that data. But where does the design of the model itself come from? What is this "graph" that the `Model` wraps?

Ready to dive into the musical score? Let's move on to the next chapter: [Model Graph (HTG)](06_model_graph__htg__.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)