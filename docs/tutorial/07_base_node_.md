# Chapter 7: Base Node

Welcome back to the `crest` tutorial! In the [previous chapter](06_model_graph__htg__.md), we learned about the `HierarchalTensorGraph` (HTG), which acts as the **master flowchart** or **blueprint** for your computational workflow. It defines how different steps (nodes) connect and how data flows between them (edges).

But what exactly are these "steps" or "nodes" that make up the flowchart? What are the fundamental building blocks?

Imagine our house-building analogy again. The HTG is the overall architectural plan showing where the walls, windows, doors, and roof go. But you still need individual blueprints for making a standard wall section, a standard window frame, etc. These smaller, standardized pieces are what you actually build or buy and then assemble according to the master plan.

In `crest`, these standardized individual building blocks are called **Base Nodes**.

## What is a Base Node in CREST? (The Standard Building Block Analogy)

A `BaseNode` in `crest` is like a **standard blueprint for a single, specific building block** or operation within your workflow. It represents one distinct step, whether it's a simple mathematical calculation, a data transformation, or a complex machine learning layer.

The core idea behind `BaseNode` is to provide a **common interface** or template. This template ensures that any operation you define, regardless of whether it's a pure Python function, a Keras neural network layer, or something else, can be understood and used by the `crest` [Model Graph (HTG)](06_model_graph__htg__.md) structure.

Think of it this way: the HTG needs to know how to pass data *into* a node and how to get data *out*. The `BaseNode` template guarantees that any node adhering to this template will have defined "inlets" (inputs) and "outlets" (outputs), and a standard way to trigger its operation (its `call` method).

`crest` provides specific implementations of `BaseNode` (though technically they inherit from `HierarchalTensorGraph`, they serve the conceptual role of simple nodes within a larger graph):

*   **`LambdaNode`:** Wraps a standard Python function or any Python object that can be "called" (has a `__call__` method). Useful for simple operations, data preprocessing steps, or custom logic written purely in Python.
*   **`KerasNode`:** Wraps a TensorFlow/Keras object, such as a `Layer`, `Model`, or `Sequential`. This is how you bring your neural network components into the `crest` graph.

By wrapping your operations in `BaseNode` implementations like `LambdaNode` or `KerasNode`, you make them compatible with the `crest` graph structure, allowing the [Model](05_model_.md) component to build and execute the full workflow using a backend like TensorFlow.

## Your Use Case: Defining the Steps for Normalization and Prediction

Let's revisit our simple workflow from [Chapter 6: Model Graph (HTG)](06_model_graph__htg__.md): we want to normalize temperature data and then feed it, along with precipitation and soil moisture, into a prediction step.

In the HTG chapter, we conceptually talked about 'Normalize' and 'Predict' nodes. Now, we'll define these concrete nodes using `BaseNode` implementations:

1.  The 'Normalize' step will be a `LambdaNode` wrapping a Python function.
2.  The 'Predict' step will be a `KerasNode` wrapping a simple Keras layer or model.

These `BaseNode` instances will then be the individual building blocks that we add to our main `HierarchalTensorGraph`.

## Using LambdaNode and KerasNode

To create a `BaseNode`, you instantiate one of its concrete implementations, like `LambdaNode` or `KerasNode`, wrapping the Python callable or Keras object you want to use. You also need to specify the node's `inputs` and `outputs`.

### Creating a LambdaNode

A `LambdaNode` wraps a Python function or any object with a `__call__` method. You pass the callable to the `LambdaNode` constructor, along with a name and specifications for what inputs it expects and what outputs it will produce.

The input/output specifications for *any* `BaseNode` (and indeed, any node in a `HierarchalTensorGraph`) are dictionaries using `TensorSpec` to describe the expected shape and data type, similar to how we defined them for the overall HTG in [Chapter 6](06_model_graph__htg__.md).

```python
# Example: Creating a LambdaNode for Normalization
from crest.model import LambdaNode, TensorSpec

# Assume we have a Python function to normalize temperature
def normalize_temperature_func(data_dict):
    # Input data_dict is expected to be {'temperature': tensor}
    temp_tensor = data_dict['temperature']
    # Hypothetical normalization logic (e.g., simple scaling)
    normalized_temp = temp_tensor / 100.0
    # Output must also be a dictionary
    return {'normalized_temp': normalized_temp}

# Define the expected input and output specs for this node
# Input: just 'temperature', same shape/dtype as in Dataset
input_spec_norm = {
    'temperature': TensorSpec((None, 7, 3, 3), dtype='float32'),
}
# Output: 'normalized_temp', same shape/dtype
output_spec_norm = {
    'normalized_temp': TensorSpec((None, 7, 3, 3), dtype='float32'),
}

# Create the LambdaNode instance
normalize_node = LambdaNode(
    node=normalize_temperature_func,
    name='NormalizeTempNode',
    inputs=input_spec_norm,
    outputs=output_spec_norm
)

print(f"Created LambdaNode: {normalize_node.name}")
print(f"Expected inputs: {list(normalize_node.inputs.keys())}")
print(f"Expected outputs: {list(normalize_node.outputs.keys())}")
```

**Explanation:**

*   We define a standard Python function `normalize_temperature_func`. It takes a dictionary as input and returns a dictionary as output – this is the standard way data is passed between nodes in the HTG when using dictionaries of features.
*   We define `input_spec_norm` and `output_spec_norm` using `TensorSpec` to tell `crest` the shape and data type of the tensors this node works with.
*   We instantiate `LambdaNode`, passing the function to the `node` parameter, giving it a `name`, and providing the input and output specifications.
*   This `normalize_node` is now a `BaseNode` compatible object that can be added to a `HierarchalTensorGraph`.

### Creating a KerasNode

A `KerasNode` wraps a Keras `Layer`, `Model`, or `Sequential` object. This allows you to use the power of Keras within your `crest` workflow. When wrapping a `Layer` or `Sequential`, you need to explicitly define the expected `inputs` and `outputs` using `TensorSpec`, as Keras layers themselves don't always have built-in knowledge of their full tensor shapes until connected. When wrapping a full Keras `Model`, the `KerasNode` can often infer the input and output specs from the Keras model itself if it was defined with named inputs/outputs.

Let's create a `KerasNode` wrapping a simple Keras layer for our prediction step.

```python
# Example: Creating a KerasNode for Prediction
from crest.model import KerasNode, TensorSpec
import tensorflow as tf
from tensorflow import keras

# Imagine a simple Keras Dense layer that takes concatenated features
# (Note: A real model would be more complex, e.g., Conv2D layers)
# Let's define the inputs it expects from the *previous* node (NormalizeTempNode)
# plus the other raw features. The KerasNode will handle accepting these
# and feeding them to the Keras layer/model internally.

# Define the expected input and output specs for this node
# Input: 'normalized_temp', 'precipitation', 'soil_moisture'
# Let's assume after some reshaping, these become flat vectors for a Dense layer
# (Shape: None for batch, 7*3*3 = 63 features per sample)
input_spec_predict = {
    'normalized_temp': TensorSpec((None, 63), dtype='float32'),
    'precipitation': TensorSpec((None, 63), dtype='float32'),
    'soil_moisture': TensorSpec((None, 63), dtype='float32'),
}
# Output: 'prediction', a single value per sample
output_spec_predict = {
    'prediction': TensorSpec((None, 1), dtype='float32'),
}

# Create the Keras Layer we want to wrap
# This layer expects a single input tensor, let's say the concatenated features
keras_dense_layer = keras.layers.Dense(units=1, name='OutputDenseLayer')

# Create the KerasNode instance
# For a Layer, we must provide inputs and outputs specs
predict_node = KerasNode(
    keras_obj=keras_dense_layer,
    name='PredictNode',
    inputs=input_spec_predict, # Specs expected *by the KerasNode*
    outputs=output_spec_predict # Specs produced *by the KerasNode*
)

print(f"\nCreated KerasNode: {predict_node.name}")
print(f"Expected inputs: {list(predict_node.inputs.keys())}")
print(f"Expected outputs: {list(predict_node.outputs.keys())}")
print(f"Wrapped Keras object type: {predict_node.type}") # KerasNodeType.LAYER
```

**Explanation:**

*   We define the expected `input_spec_predict` and `output_spec_predict` for the data flowing *into* and *out of* this specific `PredictNode` within the HTG.
*   We create a standard Keras `Dense` layer.
*   We instantiate `KerasNode`, passing the Keras layer to `keras_obj`.
*   Since we wrapped a `Layer`, we must explicitly provide the `inputs` and `outputs` specs. The `KerasNode` internally creates a minimal Keras `Model` that takes the specified inputs, passes them through the wrapped layer, and produces the specified outputs, ensuring it conforms to the `crest` node interface.

If you were wrapping a full Keras `Model` that was already built with named inputs/outputs (e.g., using the Functional API), you might not need to provide `inputs` and `outputs` explicitly to the `KerasNode`, as it can often infer them from the wrapped model.

```python
# Example: Creating a KerasNode wrapping a Keras Model (conceptual)
# This requires the Keras Model to be built and compiled *before* wrapping

# Assume you have a Keras Functional API model named 'my_keras_model'
# my_keras_model has inputs={'feature1': Input(...), 'feature2': Input(...)}
# my_keras_model has outputs={'prediction': OutputTensor, 'uncertainty': OutputTensor}

# Create the KerasNode wrapping the pre-built Keras Model
# KerasNode can often infer inputs/outputs here
# keras_model_node = KerasNode(
#     keras_obj=my_keras_model,
#     name='FullKerasModelNode',
#     # inputs and outputs can potentially be inferred here
# )

# print(f"Created KerasNode: {keras_model_node.name}")
# print(f"Expected inputs: {list(keras_model_node.inputs.keys())}")
# print(f"Expected outputs: {list(keras_model_node.outputs.keys())}")
# print(f"Wrapped Keras object type: {keras_model_node.type}") # KerasNodeType.MODEL
```

Now that we have our individual `BaseNode` instances (`normalize_node`, `predict_node`), we can add them to a `HierarchalTensorGraph` and define the edges, as shown in [Chapter 6](06_model_graph__htg__.md).

```python
# Example: Adding BaseNodes to an HTG (Revisiting Chapter 6)
from crest.model import HierarchalTensorGraph

# Assume normalize_node and predict_node are created

# Create the master HTG
master_workflow_nodes = HierarchalTensorGraph(name='MyPredictionWorkflowWithNodes')

# Add the BaseNode instances as nodes in the HTG
master_workflow_nodes.add_node(normalize_node)
master_workflow_nodes.add_node(predict_node)

# Define overall workflow inputs and outputs (using specs from the first/last nodes)
# The input to the workflow is the input to the first node (but needs all features)
# The output of the workflow is the output of the last node
# Let's define them manually for clarity, matching the data we get from the Batcher
# (Soil Moisture, Temperature, Precipitation)
master_workflow_nodes.inputs = {
    'soil_moisture': TensorSpec((None, 7, 3, 3), dtype='float32'),
    'temperature': TensorSpec((None, 7, 3, 3), dtype='float32'),
    'precipitation': TensorSpec((None, 7, 3, 3), dtype='float32'),
}
master_workflow_nodes.outputs = {
    'prediction': TensorSpec((None, 1), dtype='float32'), # Matches predict_node output
}


# Add edges between the nodes and to the overall input/output
# Edge from overall input to NormalizeTempNode (only sending 'temperature')
master_workflow_nodes.add_edge('input', normalize_node, inputs_map={'temperature': 'temperature'})

# Edge from overall input to PredictNode (sending 'precipitation', 'soil_moisture')
master_workflow_nodes.add_edge('input', predict_node, inputs_map={
    'precipitation': 'precipitation',
    'soil_moisture': 'soil_moisture'
})

# Edge from NormalizeTempNode to PredictNode (sending 'normalized_temp')
master_workflow_nodes.add_edge(normalize_node, predict_node, inputs_map={
    'normalized_temp': 'normalized_temp' # Map output feature 'normalized_temp' from norm_node
                                         # to input feature 'normalized_temp' for predict_node
})

# Edge from PredictNode to overall output
master_workflow_nodes.add_edge(predict_node, 'output')


print(f"\nHTG created with BaseNodes: {master_workflow_nodes.name}")
print(f"Nodes: {list(master_workflow_nodes.nodes.keys())}")
print(f"Edges: {list(master_workflow_nodes.edges)}")
# Edges also include the mapping information if inputs_map was used
# print(master_workflow_nodes.graph.get_edge_attributes('inputs_map'))
```

**Explanation:**

*   We create the main `HierarchalTensorGraph`.
*   We add the `LambdaNode` and `KerasNode` instances we created as nodes to this HTG.
*   We define the overall `inputs` and `outputs` for the *entire workflow*.
*   When adding edges, we use the optional `inputs_map` parameter. This is very important! It tells `crest` how to map output features from a source node (or the overall `'input'`) to the expected input features of a target node (or the overall `'output'`). For example, the edge from `'input'` to `normalize_node` maps the overall input feature `'temperature'` to the input feature `'temperature'` expected by `normalize_node`. The edge from `normalize_node` to `predict_node` maps the output feature `'normalized_temp'` from `normalize_node` to the input feature `'normalized_temp'` expected by `predict_node`. Edges from `'input'` to `predict_node` pass the other necessary features.

This refined example shows how `BaseNode` implementations are the concrete components plugged into the `HierarchalTensorGraph` using defined inputs/outputs and edge mappings.

## Behind the Scenes: What Happens When a Base Node is Called?

When the `crest` [Model](05_model_.md) component executes a `HierarchalTensorGraph` (as described in [Chapter 6](06_model_graph__htg__.md)), and it reaches a `BaseNode` instance (like our `LambdaNode` or `KerasNode`), it essentially triggers that node's operation.

Let's look at a simplified view of what happens when the HTG traversal (`__call__` method from [Chapter 6](06_model_graph__htg__.md)) processes a `LambdaNode`:

```{mermaid}
sequenceDiagram
    participant HTG as HierarchalTensorGraph(__call__)
    participant LambdaNode as LambdaNode Object
    participant WrappedFunc as normalize_temperature_func
    participant InputData as Input Dictionary
    participant OutputData as Output Dictionary

    HTG->>LambdaNode: Call LambdaNode (passing input data dictionary)
    LambdaNode->>LambdaNode: Check if it's a basenode (True)
    LambdaNode->>LambdaNode: Prepare input data (feature mapping, flatten)
    LambdaNode->>WrappedFunc: Call wrapped function (passing prepared data)
    WrappedFunc->>WrappedFunc: Execute normalization logic
    WrappedFunc-->>LambdaNode: Return raw output (dictionary)
    LambdaNode->>LambdaNode: Prepare output data (feature mapping, reshape)
    LambdaNode-->>HTG: Return processed output dictionary
```

For a `KerasNode`, the process is similar, but the internal call goes to the wrapped Keras object:

```{mermaid}
sequenceDiagram
    participant HTG as HierarchalTensorGraph(__call__)
    participant KerasNode as KerasNode Object
    participant WrappedKerasObj as Keras Layer/Model
    participant _NodeWrap as _NodeWrap utility
    participant InputData as Input Dictionary
    participant OutputData as Output Dictionary

    HTG->>KerasNode: Call KerasNode (passing input data dictionary)
    KerasNode->>KerasNode: Check if it's a basenode (True)
    KerasNode->>KerasNode: Prepare input data (feature mapping, flatten)
    KerasNode->>_NodeWrap: Call internal _NodeWrap instance (passing prepared data)
    _NodeWrap->>WrappedKerasObj: Call wrapped Keras object (`call` method)
    WrappedKerasObj->>WrappedKerasObj: Execute Keras layer/model logic
    WrappedKerasObj-->>_NodeWrap: Return raw output tensor(s)
    _NodeWrap-->>KerasNode: Return output tensor(s) (possibly wrapped in dict by _NodeWrap)
    KerasNode->>KerasNode: Prepare output data (feature mapping, reshape)
    KerasNode-->>HTG: Return processed output dictionary
```

Notice the role of `_NodeWrap` in the KerasNode diagram. As seen in the provided code, `LambdaNode` and `KerasNode` actually inherit from `HierarchalTensorGraph`. The `HierarchalTensorGraph` class, when initialized with a `node` (making it a Basenode), internally wraps that callable `node` object within a `_NodeWrap` instance (a Keras `Layer`). This allows the `crest` graph execution to work seamlessly with TensorFlow's graph execution and enables features like automatic histogram logging for node inputs/outputs within TensorBoard (as shown in `_NodeWrap.call`).

Let's peek at the core structure of `BaseNode` and its implementations (`LambdaNode`, `KerasNode`).

**`BaseNode` Abstract Structure (from `base/BaseNode.py`):**

```python
# From base/BaseNode.py (simplified concept)
class BaseNode(BaseAbstract):
    inputs: IO_TYPE  # Class-level annotation for expected input structure
    outputs: IO_TYPE # Class-level annotation for expected output structure

    def __init__(self, normalize=(), loss='mse', ...):
        # Handles configuration like normalization, loss, debug
        self.normalize = ...
        self.loss = ...
        # ... stores other configs ...

    # @abstractmethod # BaseNode defines an abstract call method, but specific
    # def call(self, X, training): # implementations override or use a wrapped callable
    #     """ Must be implemented by inheriting classes """
    #     raise NotImplementedError

    @cached_property
    def graph(self) -> 'HierarchalTensorGraph':
        """ Creates the default Basenode graph object """
        from crest.model import HierarchalTensorGraph as HTG
        # Creates an HTG wrapping an internal _NodeWrap instance that
        # calls the object's core logic (often self._call or self.call)
        return HTG(**{
            'node': _NodeWrap(self, f'{self}-call'), # Wrap self for the HTG!
            'name': f'{self}',
            'inputs': self.input_spec, # Uses the class-level inputs
            'outputs': self.output_spec, # Uses the class-level outputs
        })

    @classproperty
    def input_spec(cls) -> dict[str, tf.TensorSpec]:
        """ Converts class.inputs into TensorSpec objects """
        return cls._generate_spec(cls.inputs)

    @classproperty
    def output_spec(cls) -> dict[str, tf.TensorSpec]:
        """ Converts class.outputs into TensorSpec objects """
        return cls._generate_spec(cls.outputs)

    # ... _add_data_transform, _call, convert_onehot, convert_coords ...
    # ... __call__ method (calls self.graph(X)) ...
    # ... __init_subclass__ (validates inputs/outputs class attributes) ...
    # ... __post_init__ (handles normalization setup) ...

# Simplified NodeWrap used internally by HTG for Basenodes
class _NodeWrap(tf.keras.layers.Layer):
    def __init__(self, obj: Callable, name: str = ''):
        super().__init__(name=name)
        # Copies Keras/TF objects from the wrapped 'obj' to itself
        # so Keras can track them
        self.obj = obj # Stores the actual object (e.g., BaseNode instance)

    def call(self, X, *args, **kwargs):
        """ Calls the wrapped object's method and logs histograms """
        # Calls obj._call or obj.call depending on how obj is structured
        out = getattr(self.obj, '_call', self.obj)(X, *args, **kwargs)
        # Adds histograms for tensorboard
        self._add_histograms(X, 'input')
        self._add_histograms(out, 'output')
        return out

```

**Explanation:** `BaseNode` provides the template, enforcing the definition of `inputs` and `outputs` at the class level (`__init_subclass__`). It defines how to generate `TensorSpec`s from these using `_generate_spec`. Crucially, its `graph` property automatically creates a `HierarchalTensorGraph` instance that wraps the `BaseNode` object itself (using `_NodeWrap`) and uses the defined `input_spec`/`output_spec`. The `__call__` method of a `BaseNode` simply calls this internal graph, which in turn calls the `_NodeWrap`, which finally calls the `BaseNode`'s core logic (`_call` or `call`). The `_NodeWrap` acts as a bridge into the TensorFlow graph.

**`LambdaNode` and `KerasNode` Implementation Details (from `LambdaNode.py` and `KerasNode.py`):**

```python
# From LambdaNode.py (simplified)
class LambdaNode(HierarchalTensorGraph): # Inherits from HTG
    def __init__(self, node, name=None, inputs={}, outputs={}):
        # Calls the parent HTG's init, passing the user-provided 'node' callable.
        # HTG's init will wrap this 'node' in a _NodeWrap instance internally.
        super().__init__(node=node, name=name,
                         inputs=inputs, outputs=outputs)
        # node (the callable) is stored in self.node by HTG's init
        # self.graph is an HTG wrapping self.node via _NodeWrap


# From KerasNode.py (simplified)
class KerasNode(HierarchalTensorGraph): # Inherits from HTG
    def __init__(self, keras_obj, name=None, inputs={}, outputs={}):
        self.type = self._check_type(keras_obj) # Identify Keras object type

        if self.type == KerasNodeType.MODEL:
            self.keras_obj = keras_obj
            # Infer inputs/outputs from Keras Model if possible
            self.inputs = {k: TensorSpec(v) for k, v in self.keras_obj.input.items()}
            self.outputs = {k: TensorSpec(v) for k, v in self.keras_obj.output.items()}
            # Pass the Keras Model itself as the 'node' to the parent HTG
            super().__init__(node=self.keras_obj, name=name,
                             inputs=self.inputs, outputs=self.outputs)

        elif self.type in [KerasNodeType.LAYER, KerasNodeType.SEQUENTIAL]:
            # Require inputs/outputs specs for Layers/Sequential
            if not inputs or not outputs: raise Exception(...)
            self.inputs = {k: TensorSpec(v) for k, v in inputs.items()}
            self.outputs = {k: TensorSpec(v) for k, v in outputs.items()}

            # Create a minimal Keras Model that wraps the Layer/Sequential
            # to conform to the crest node interface (dict inputs/outputs)
            key_x = list(self.inputs.keys())[0] # Assume single input for layers
            key_y = list(self.outputs.keys())[0] # Assume single output for layers
            x = tf.keras.Input(type_spec=self.inputs[key_x].tf)
            y = keras_obj(x) # Pass input through the layer/sequential
            self.keras_obj = keras.Model(inputs={key_x: x}, outputs={key_y: y})

            # Pass this internal Keras Model as the 'node' to the parent HTG
            super().__init__(node=self.keras_obj, name=name,
                             inputs=self.inputs, outputs=self.outputs)

        # In both cases, the parent HTG's init wraps self.keras_obj in _NodeWrap
        # self.graph is an HTG wrapping self.keras_obj via _NodeWrap

    # ... _check_type, to_json, from_json, save, load methods ...
```

**Explanation:** `LambdaNode` and `KerasNode` leverage the `HierarchalTensorGraph`'s ability to be a Basenode when a `node` (callable) is provided to its `__init__`. `LambdaNode` passes the user's function directly. `KerasNode` is more complex: if wrapping a Keras `Model`, it passes the model itself. If wrapping a `Layer` or `Sequential`, it first creates a minimal Keras `Model` that *contains* the layer and conforms to the expected dictionary input/output format, and then passes *that* internal `Model` to the parent HTG's init. In both cases, the parent HTG then wraps the provided `node` (either the user's function, their Keras Model, or the internally created Keras Model) within a `_NodeWrap` instance. This `_NodeWrap` instance becomes the core callable that the `HierarchalTensorGraph` actually executes when this Basenode is called during the graph traversal.

This explains why `LambdaNode` and `KerasNode` inherit from `HierarchalTensorGraph` and not `BaseNode` directly in the provided code structure. `HierarchalTensorGraph` seems to be the class that handles the Basenode vs. Multi-node distinction and the wrapping of the core callable, while `BaseNode` provides common properties and methods useful for *any* node, particularly those that represent a distinct processing *component* with configs like loss or normalization (even if the node itself is a complex HTG). This structure allows flexibility, where even a complex, multi-node HTG could inherit from `BaseNode` if it represents a single conceptual component with `BaseNode`-like configurations.

In essence, `LambdaNode` and `KerasNode` are specialized `HierarchalTensorGraph`s that serve as Basenodes by wrapping specific types of callables, making them usable within the larger HTG structure.

## Conclusion

In this chapter, we explored the **Base Node** concept, understanding that they are the standardized building blocks or individual operation steps used within a `crest` [Model Graph (HTG)](06_model_graph__htg__.md). We saw how `LambdaNode` wraps Python callables and `KerasNode` wraps TensorFlow/Keras objects, providing a common interface compatible with the `crest` graph structure. You learned how to create these nodes by defining their expected `inputs` and `outputs` using `TensorSpec` and wrapping your specific function or Keras object. We also saw how these `BaseNode` instances are added to a `HierarchalTensorGraph` using edges and `inputs_map` to define the workflow.

You now understand how the individual components of your workflow are defined and made compatible with the `crest` graph. We've covered the data (`Data Abstractions`, `Data Matching`), how it's delivered (`Batching`), the abstract blueprint (`Model Graph (HTG)`), the individual operational units (`Base Node`), and the conductor that runs the blueprint (`Model`).

Putting all these pieces together allows `crest` to build complex systems, especially simulations and digital replicas. Ready to see how `crest` orchestrates these components for building such systems? Let's move on to the next chapter: [Digital Replica Engine](08_digital_replica_engine_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)