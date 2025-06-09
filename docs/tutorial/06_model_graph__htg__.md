# Chapter 6: Model Graph (HTG)

Welcome back to the `crest` tutorial! In the [previous chapter](05_model_.md), we learned about the **Model** component and how it acts as the conductor, using a *blueprint* to translate your desired computation (like a neural network) into something a backend library (like Keras) can run, using batches of data from the [Batching](04_batching_.md) component.

But where does this blueprint come from? How do you design the actual steps and connections that make up your complex workflow or model?

Imagine you're building a sophisticated machine, say, one that processes satellite images to predict local weather conditions. It's not a single box; it's a system with multiple stages: first, maybe some data cleaning, then perhaps a step to combine satellite data with ground station readings, followed by a machine learning model prediction, and finally, some post-processing. How do you describe the sequence of these steps and how data flows between them?

This is where `crest`'s **Model Graph (HTG)** comes in.

## What is a Model Graph (HTG) in CREST? (The Master Flowchart Analogy)

The core blueprint in `crest` is the `HierarchalTensorGraph`, or **HTG**. Think of the HTG as a **master flowchart** for your computational workflow. It's a structure that defines:

1.  **Nodes:** These represent the individual steps or components in your workflow. A node could be something simple like a mathematical operation, a data normalization step, or a complex component like an entire machine learning model.
2.  **Edges:** These represent the flow of data between the nodes. An edge from Node A to Node B means that the output of Node A becomes an input to Node B.

What makes the HTG especially powerful is the "Hierarchal" part. Just like a master flowchart can have smaller flowcharts drawn within specific steps, an HTG node can itself be *another* HTG! This allows you to build incredibly complex systems by combining smaller, reusable, and understandable sub-workflows.

There are two main types of HTG nodes:

*   **Basenode:** This is the simplest kind of node. It represents an atomic operation or a single, fundamental process. It's typically created by wrapping a standard Python function or a simple class instance that can be "called" (like a Keras layer). It doesn't contain other nodes or edges internally. Think of it as a single instruction like "Normalize Data" or "Apply Neural Network Layer".
*   **Multi-node HTG:** This is an HTG that contains other nodes (which can be Basenodes or other Multi-node HTGs) and edges connecting them. It represents a sub-workflow or a system composed of multiple steps. Think of it as a "Data Preprocessing Module" or "Prediction Subsystem" that encapsulates several simpler steps.

The HTG provides the structure that the [Model](05_model_.md) component uses to build the actual computational graph in a backend like Keras or TensorFlow.

## Your Use Case: Building a Simple Processing Chain

Let's go back to our satellite and weather data example. We want to build a simple processing chain that:

1.  Takes the raw satellite soil moisture and weather temperature/precipitation data.
2.  Applies a hypothetical 'Normalize' step to the temperature data.
3.  Feeds the normalized temperature, raw precipitation, and raw soil moisture into a simple 'Prediction Model' step.
4.  Outputs the prediction from the model.

This requires defining three main steps and the data flow between them: Normalize -> Predict -> Output.

We'll use HTGs to build this blueprint.

## Using the HTG to Define Your Workflow

You interact with the `HierarchalTensorGraph` class to define your workflow.

First, you typically create **Basenodes** for your fundamental operations. For our example, let's imagine we have simple Python functions or classes that perform 'Normalize' and 'Predict' (details on creating actual Basenodes like `LambdaNode` or `KerasNode` are covered in [Chapter 7: Base Node](07_base_node_.md)).

```python
# Example: Conceptual Basenodes (Simplified - actual nodes covered in Chapter 7)
# Imagine these are simple functions or Keras layers

def normalize_temperature(temp_data):
    # Hypothetical normalization logic
    print("--> Running Normalize step")
    return temp_data / 100.0 # Dummy normalization

class PredictionModel:
    def __call__(self, inputs):
        # Inputs would be a dictionary like {'normalized_temp': ..., 'precipitation': ..., 'soil_moisture': ...}
        print("--> Running Prediction Model step")
        # Hypothetical prediction logic
        combined_input = inputs['normalized_temp'] + inputs['precipitation'] + inputs['soil_moisture']
        return {'prediction': combined_input * 0.5} # Dummy prediction

# In crest, you'd wrap these in actual Basenode types like LambdaNode or KerasNode
# We'll see how in Chapter 7. For now, just know these are the *operations*.
```

Next, you use the `HierarchalTensorGraph` class to build the flowchart structure. You can create empty HTGs and add nodes and edges to them.

```python
# Example: Creating a Multi-node HTG and adding Basenodes
from crest.model import HierarchalTensorGraph
from crest.model import LambdaNode # We'll use LambdaNode as a simple Basenode type

# Create Basenodes (wrapping our conceptual operations)
# In reality, LambdaNode wraps a function, KerasNode wraps a Keras layer
normalize_node = LambdaNode(normalize_temperature, name='NormalizeTemp')
predict_node = LambdaNode(PredictionModel(), name='Predict')

# Create the master HTG (a Multi-node HTG)
master_workflow = HierarchalTensorGraph(name='MyPredictionWorkflow')

# Add the Basenodes to the master HTG
master_workflow.add_node(normalize_node)
master_workflow.add_node(predict_node)

print(f"Created workflow HTG: {master_workflow.name}")
print(f"Nodes added: {list(master_workflow.nodes.keys())}")
```

**Explanation:**

*   We import `HierarchalTensorGraph` and a simple Basenode type, `LambdaNode`.
*   We create instances of `LambdaNode`, wrapping our conceptual `normalize_temperature` function and `PredictionModel` class instance. Each gets a unique name.
*   We create a new `HierarchalTensorGraph` called `'MyPredictionWorkflow'`. Since we don't pass a `node` callable to its `__init__`, it becomes a Multi-node HTG (an empty graph initially).
*   We use the `add_node` method to add our `normalize_node` and `predict_node` Basenodes to the `master_workflow`. The HTG keeps track of these nodes internally.

Now, we need to define the flow of data using edges. We want data to go from an overall `input` to `normalize_node`, then from `normalize_node` to `predict_node`, and finally from `predict_node` to an overall `output`.

HTGs automatically handle special `input` and `output` nodes. You just refer to them by name (`'input'`, `'output'`).

```python
# Example: Adding edges to define the data flow
# Assume master_workflow, normalize_node, and predict_node are created

print("\nAdding edges to define flow...")

# Add edge from the overall input to the NormalizeTemp node
# This means data entering the HTG should go to NormalizeTemp
master_workflow.add_edge('input', normalize_node)

# Add edge from NormalizeTemp to the Predict node
# The output of NormalizeTemp becomes input for Predict
master_workflow.add_edge(normalize_node, predict_node)

# Add edge from the Predict node to the overall output
# The output of Predict becomes the final output of the HTG
master_workflow.add_edge(predict_node, 'output')

print("Edges added.")
print(f"Workflow edges: {list(master_workflow.edges)}")
```

**Explanation:**

*   We use the `add_edge` method to define connections.
*   Edges are specified as `(source_node, target_node)`.
*   We pass the `HierarchalTensorGraph` objects (`normalize_node`, `predict_node`) or the special strings `'input'` and `'output'`. `crest` looks them up within the graph.
*   The HTG's internal graph representation is updated with these connections.

Now, let's specify the expected data structure for the overall input and output of this `master_workflow` HTG using the `inputs` and `outputs` parameters we saw in the `__init__` method description. These usually use `TensorSpec` ([Chapter 5: Model](05_model_.md) briefly mentioned it, and it's defined in `TensorSpec.py`), which describes the shape and data type of the expected data.

```python
# Example: Specifying overall HTG inputs and outputs using TensorSpec
from crest.model import TensorSpec

# Define the expected input data structure
# We expect 'soil_moisture', 'temperature', 'precipitation' as inputs
# Let's say they are 4D arrays (batch, time, lat, lon) of floats
input_specs = {
    'soil_moisture': TensorSpec((None, 7, 3, 3), dtype='float32'), # (batch, time, lat, lon)
    'temperature': TensorSpec((None, 7, 3, 3), dtype='float32'),
    'precipitation': TensorSpec((None, 7, 3, 3), dtype='float32'),
}

# Define the expected output data structure
# We expect a single output feature named 'prediction', same shape
output_specs = {
    'prediction': TensorSpec((None, 7, 3, 3), dtype='float32'),
}

# You would typically set these during HTG initialization or later
# For simplicity, let's assume we set them during init
# master_workflow = HierarchalTensorGraph(name='MyPredictionWorkflow', inputs=input_specs, outputs=output_specs)
# If adding later:
master_workflow.inputs = input_specs
master_workflow.outputs = output_specs

print(f"\nMaster workflow inputs: {master_workflow.inputs.keys()}")
print(f"Master workflow outputs: {master_workflow.outputs.keys()}")
```

**Explanation:**

*   `TensorSpec` is used to describe the shape and data type of the data that flows through the graph. `(None, 7, 3, 3)` indicates the first dimension is the batch size (flexible), followed by 7 time steps, 3 latitude points, and 3 longitude points.
*   Setting the `inputs` and `outputs` dictionaries on the master HTG tells `crest` what data structure this blueprint expects to receive and what structure it will produce. This is crucial information for the [Model](05_model_.md) component when it builds the backend model.

**Hierarchical Example:**

What if our 'Predict' step itself involved multiple sub-steps, like applying a convolutional layer, then a pooling layer, etc.? We could define a separate HTG for the 'Predict' subsystem:

```python
# Example: Creating a Hierarchical Structure
# Assume Keras layers or similar operations exist conceptually

# Create a separate HTG for the prediction subsystem
prediction_subsystem = HierarchalTensorGraph(name='PredictionSubsystem')

# Assume we have basenodes for Keras layers
# conv_node = KerasNode(Conv2D(...), name='ConvLayer') # Details in Chapter 7
# pool_node = KerasNode(MaxPooling2D(...), name='PoolLayer') # Details in Chapter 7
# We'll use LambdaNodes for simplicity here

conv_node = LambdaNode(lambda x: x, name='ConvLayer') # Dummy op
pool_node = LambdaNode(lambda x: x, name='PoolLayer') # Dummy op

# Add nodes and edges within the subsystem HTG
prediction_subsystem.add_node(conv_node)
prediction_subsystem.add_node(pool_node)
prediction_subsystem.add_edge('input', conv_node)
prediction_subsystem.add_edge(conv_node, pool_node)
prediction_subsystem.add_edge(pool_node, 'output')

# Now, when building the master workflow, use this subsystem HTG as a node
# Remove the old predict_node first if it exists
# master_workflow.remove_node('Predict') # If you ran the previous example

master_workflow_hierarchical = HierarchalTensorGraph(name='MyHierarchicalWorkflow')
master_workflow_hierarchical.add_node(normalize_node) # Our normalize basenode
master_workflow_hierarchical.add_node(prediction_subsystem) # Add the subsystem HTG as a node!

master_workflow_hierarchical.add_edge('input', normalize_node)
master_workflow_hierarchical.add_edge(normalize_node, prediction_subsystem)
master_workflow_hierarchical.add_edge(prediction_subsystem, 'output')

print(f"\nCreated hierarchical workflow HTG: {master_workflow_hierarchical.name}")
print(f"Top-level nodes: {list(master_workflow_hierarchical.nodes.keys())}")
# You can access nested nodes using paths (tuple or string)
print(f"Nodes inside PredictionSubsystem: {list(master_workflow_hierarchical['PredictionSubsystem'].nodes.keys())}")
print(f"Edges inside PredictionSubsystem: {list(master_workflow_hierarchical['PredictionSubsystem'].edges)}")

# Access a deeply nested node using a path tuple
# print(master_workflow_hierarchical['PredictionSubsystem', 'ConvLayer'])
# Or using string path (if enabled/configured)
# print(master_workflow_hierarchical['PredictionSubsystem.ConvLayer']) # Note: String path depends on specific implementation details/helpers
```

**Explanation:**

*   We created a totally separate `HierarchalTensorGraph` (`prediction_subsystem`) with its own internal structure (input -> Conv -> Pool -> output).
*   Crucially, we added this *entire HTG object* as a node (`prediction_subsystem`) to our `master_workflow_hierarchical`.
*   The edges then connect to this subsystem node just like any other node.
*   When the `master_workflow_hierarchical` is processed (e.g., by the [Model](05_model_.md)), when it gets to the `prediction_subsystem` node, it will execute the internal graph defined *within* that subsystem HTG.
*   You can access nodes at any level of the hierarchy using paths, like `my_htg[parent_node_name][child_node_name]` or using a tuple `my_htg[(parent_name, child_name)]`. The provided `HierarchalTensorGraph` code shows tuple access and string access helpers (`_get_node`).

This hierarchical capability is fundamental to building large, complex workflows in `crest` while maintaining modularity.

Finally, once your HTG is defined, you would pass it to the [Model](05_model_.md) component to build the backend model and run it, as shown in [Chapter 5: Model](05_model_.md):

```python
# Example: Using the HTG with the Model component (revisiting Chapter 5)
# Assume my_dataset is configured and my_batch_size is set
# Assume master_workflow (or master_workflow_hierarchical) is defined

from crest.model import Model

# Create the CREST Model using the HTG blueprint
my_crest_model = Model(graph=master_workflow, name='MyModel')

# Compile the model (backend-specific setup)
my_crest_model.compile(optimizer='adam', loss='mse') # Example for Keras backend

# Train the model using a batcher derived from your dataset
# (The Model's fit method handles Batcher creation internally)
# my_crest_model.fit(dataset=my_dataset, epochs=10, batch_size=my_batch_size)

# Make predictions
# predictions = my_crest_model.predict(dataset=my_prediction_dataset, batch_size=my_batch_size)

print("\nHTG successfully used to create a CREST Model.")
```

**Explanation:** The `Model` class takes the `HierarchalTensorGraph` as its `graph` argument. It uses this HTG to understand the structure of the computation it needs to perform and then builds the corresponding structure using the selected backend.

## Visualizing the HTG

Seeing the flowchart is often easier than reading the code. The `HierarchalTensorGraph` includes methods to draw the graph. `draw_int` creates an interactive HTML visualization using `pyvis`.

```python
# Example: Visualizing the HTG
# Assume master_workflow_hierarchical is defined

print("\nGenerating interactive graph visualization...")

# This will create an HTML file and open it in your browser
# Use inline=True in a notebook environment
master_workflow_hierarchical.draw_int(html_file='my_prediction_workflow.html')

print("Interactive graph saved to my_prediction_workflow.html")
```

**Explanation:** The `draw_int` method visualizes the HTG. If you used the hierarchical example, the graph would show `input -> NormalizeTemp -> PredictionSubsystem -> output`, and you could click on `PredictionSubsystem` to expand it and see its internal nodes (`input -> ConvLayer -> PoolLayer -> output`).

## Behind the Scenes: How the HTG Works

The `HierarchalTensorGraph` class (`model/HierarchalTensorGraph.py`) uses the `networkx` library internally to manage the graph structure.

*   **Graph Representation:** The core is a `networkx.DiGraph` (Directed Graph), stored in `self.graph`.
*   **Nodes in `networkx`:** Each node in the `networkx` graph is represented by a string (the node's name). The actual `HierarchalTensorGraph` object that the node represents is stored as an attribute of the `networkx` node, typically under the key `'htg'` (see `NetworkXGraph.py` `get_node_attributes`).
*   **Adding Nodes/Edges:** Methods like `add_node` and `add_edge` interact with the internal `networkx` graph (`self.graph`). They ensure that the nodes being added are also `HierarchalTensorGraph` instances (wrapping callables if necessary using `get_node`) and correctly store them as node attributes.
*   **Hierarchy:** The hierarchical structure is managed by the fact that the `htg` attribute of a `networkx` node can itself be a `HierarchalTensorGraph` with its own internal `networkx` graph. The `__getitem__` method handles accessing nodes by path, recursively navigating down the hierarchy using the node names. The `_get_node` and `_get_parent` helpers manage this path traversal logic.
*   **`__call__`:** This is the most complex method. When you "call" an HTG (`my_htg(data)`), this method is executed. For a Basenode, it simply calls the wrapped callable (`self.node(_X)` where `_X` is the processed input). For a Multi-node HTG, it performs a topological sort or similar traversal of its internal graph. It starts with the overall `input` node (which receives the initial data `_X`), then processes nodes whose inputs are ready, flowing data along the edges until it reaches the overall `output` node. It uses caching (`functools.cache` on the `traverse` helper function) to avoid recomputing results for nodes that are inputs to multiple downstream nodes. This recursive traversal is how the data is propagated through the nested graph structure. The `feature_map` method handles the renaming and selection of specific data items (`TensorSpec`s) as they flow between nodes, based on the `inputs`, `outputs`, `_inputs_map`, and `_outputs_map` attributes of each node.

Let's look at simplified snippets illustrating these points from `model/HierarchalTensorGraph.py`:

**Initialization and Basenode Check:**

```python
# From model/HierarchalTensorGraph.py (simplified __init__ and is_basenode)
class HierarchalTensorGraph(TensorGraph):
    def __init__(self, node: None | Callable = None, name: None | str = None, ...):
        self.name = name or HierarchalTensorGraph.get_name(node)
        self.node = node or self # If node is None, it's a multi-node HTG
        self.graph = NetworkXGraph() # Internal graph structure
        # ... other attributes ...

        # If it's a basenode (wrapping a callable), add itself as the only node
        if (self.node is not self):
             self.graph.add_node(self.name, htg=self) # Basenode contains itself as the only node

    @property
    def is_basenode(self) -> bool:
        """ Check if graph is empty """
        # A basenode's internal graph (self.graph) represents its child nodes.
        # A basenode itself doesn't have children graphs, so its internal graph is empty.
        return self.graph.is_empty
```

**Explanation:** The `__init__` sets up the name, decides if it's a Basenode (`node is not self`), and initializes the internal `NetworkXGraph`. A Basenode's `self.graph` remains empty of *other* nodes; the Basenode *is* the node itself in a parent graph. The `is_basenode` property checks if the internal graph is empty, which is the defining characteristic of a Basenode's internal state.

**Adding Nodes and Edges:**

```python
# From model/HierarchalTensorGraph.py (simplified add_node and add_edge)
class HierarchalTensorGraph: # ... other methods ...
    def add_node(self, node: Union[Callable, TensorGraph]):
        # Ensures the node object is wrapped in a HTG and adds it to self.graph
        htg_node = self.get_node(node)
        # get_node handles checking if it already exists and adds it if needed
        # get_node also sets the 'htg' attribute on the networkx node

    def add_edge(self, source: Union[str, Callable], target: Union[str, Callable], ...):
        # Retrieves the HTG objects for source and target names/callables
        source_htg = self.get_node(source)
        target_htg = self.get_node(target)

        if (source_htg is not None) and (target_htg is not None):
            # Adds the edge to the internal networkx graph using node names
            self.graph.add_edge(source_htg.name, target_htg.name, ...)
            # Add edge attributes like rollout_axis if needed
            # self.graph.set_edge_attributes(...)
```

**Explanation:** These methods abstract away the `networkx` calls. `add_node` uses `get_node` to ensure the node is a proper HTG object and manages adding it to the internal graph. `add_edge` similarly resolves the source and target into HTG objects and adds the edge using their names in the internal graph.

**Accessing Nodes by Path:**

```python
# From model/HierarchalTensorGraph.py (simplified __getitem__)
class HierarchalTensorGraph: # ... other methods ...
    def __getitem__(self, path: str | tuple | TensorGraph) -> 'HierarchalTensorGraph':
        """ Retrieve the node which has the given name or path """
        if isinstance(path, str):
            # Look for a direct child node by name in the internal graph
            # Handles special 'input'/'output' nodes
            if path in ['input', 'output'] and path not in self.graph:
                 # Create dummy input/output nodes if they don't exist
                 io = HierarchalTensorGraph.identity(path)
                 io.parent = self # Link back to parent HTG
                 self.graph.add_node(io.name, htg=io)
                 return io
            try:
                # Return the HTG object stored as an attribute of the networkx node
                return self.graph.nodes[path]['htg']
            except KeyError:
                 raise ImproperTensorGraphError(f'node with path = {path} not in graph.')

        if isinstance(path, tuple):
            # Recursively traverse for tuple paths
            return self[path[0]][path[1:]] # Get the first step, then access the rest from there

        # Handle accessing by HTG object itself or unsupported types
        # ... simplified ...
```

**Explanation:** `__getitem__` is how you access nodes. If you give a string, it looks for a direct child by name in `self.graph.nodes` and returns the stored HTG object. If you give a tuple path like `('Parent', 'Child')`, it recursively calls itself, first getting the 'Parent' node, and then calling `__getitem__` on that Parent HTG object with the rest of the path `('Child',)` (or just `'Child'` if it's the last element). This builds the path-based access.

**The Call Method (`__call__`):**

```python
# From model/HierarchalTensorGraph.py (highly simplified __call__)
class HierarchalTensorGraph: # ... other methods ...
    def __call__(self, X: dict) -> dict:
        """Propagate the given input X through the graph."""
        # Apply input feature mapping/selection
        _X = self.feature_map(X, 'input')

        # If it's a basenode, just call the wrapped function/object
        if self.is_basenode:
            # Flatten input dict if it came from 'input' node
            flattened_X = self.flatten(_X)
            # Call the underlying callable node
            raw_output = self.node(flattened_X)
            # Apply output feature mapping/selection
            return self.feature_map(raw_output, 'output')

        # If it's a multi-node HTG, traverse the internal graph
        # This involves complex logic to find nodes with ready inputs,
        # handle recurrence (not shown), and recursively call child nodes
        # The 'traverse' function (cached) does the core work of finding
        # results for a given node by recursively getting results from its inputs.
        traverse = cache(lambda name, depth=0: (name, self.search(name, depth))) # search needs input from predecessors

        # Start traversal from the overall output node(s) to trigger computation backwards
        # The results dictionary holds the outputs of all nodes, keyed by their path/name
        results = self.feature_map(self.flatten(dict(map(traverse, self.sinks))), 'output') # Sinks should usually only be 'output'
        return results

    def search(self, name, depth):
         # This is where the recursive logic happens within __call__
         # It gets results for nodes that are inputs to 'name'
         # Calls self[input_node_name] to get the input node's HTG object
         # Calls self[input_node_name](input_data_for_this_node) to compute its output
         # Collects outputs from all predecessors to form input for node 'name'
         # Calls self[name](collected_input_data) to compute the output of node 'name'
         # Handles rollout_axis for recurrent edges (complex, not fully shown)
         pass # Placeholder for actual complex logic
```

**Explanation:** The `__call__` method is the execution engine. For Basenodes, it's straightforward: map inputs, call the underlying function, map outputs. For Multi-node HTGs, it's much more involved. It uses a recursive traversal function (`traverse`) to calculate the output of a node by ensuring the outputs of all its predecessors are computed first. This forms the computational graph execution. The results are then potentially mapped and returned. The `flatten` helper removes the dictionary wrapping that happens when data flows through special `'input'` or `'output'` nodes.

The `HierarchalTensorGraph` provides a flexible and powerful way to define complex, modular workflows, serving as the essential blueprint for the `crest.model.Model` component.

## Conclusion

In this chapter, you learned that the `HierarchalTensorGraph` (HTG) is the blueprint for your computational workflow in `crest`. It defines steps as **nodes** and data flow as **edges**, acting like a master flowchart. Crucially, you saw how the HTG is **hierarchical**, meaning nodes can themselves be other HTGs, enabling modularity. You learned how to build a simple workflow by adding Basenodes and defining edges, and how to specify the expected input and output data structure.

You now understand how the abstract design of your workflow is captured by the HTG, which the [Model](05_model_.md) uses. But what exactly goes *inside* a Basenode?

Ready to look at the simplest building blocks of the HTG? Let's move on to the next chapter: [Base Node](07_base_node_.md).

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)