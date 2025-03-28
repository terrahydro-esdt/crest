Models
======


- *HierarchalTensorGraph: a graph-based specification of model that exchanges information (I/O) in the form of tensors (actually, dictionaries of batches of tensors).* The HierarchalTensorGraph (HTG) is where users inject the code for their models and specify how they are connected or coupled. In a HTG, nodes simply represent some callable Python function that maps input dictionaries of tensor to output dictionaries of tensors. Empty HTGs, or graphs with a single node and no children, are referred to as Basenodes and sit at the lowest level of the hierarchy. Basenodes are where users implement their models. Once all Basenodes are defined, the hierarchal structure of model can be defined simply by creating sub-graphs and drawing edges that connect nodes to exchange I/O. The framework will handle the execution of the graph and provide functionality (such as saving and loading) of the complete graph. In this way, the complexity of the hierarchal model is removed from the user.

- *Model: a realization of an HTG compiled with a specific tensor-based software backend (e.g., TensorFlow) which provides all the required functionality for evaluating and training models.* To build models CREST separates model specification/code  (HTG) with compilation. The reason for this is to allow the framework to parse a HierarchalTensorGraph and figure out how to compile it using a specific tensor-based software backend. For example, users may express their models using PyTorch, TensorFlow, or even Numpy, and the framework will parse this information, and create modification to express them all in, say TensorFlow. This provides a much more flexible framework. Once compiled, the tensor-based backend will generally provide the critical functionality (prediction, training, etc.) and CREST provides this through the Model’s standard interfaces.

- *GriddedModel: a configurable object that packages together a model with a gridded region (e.g. a latitude-longitude grid), and associated data.* The GriddedModel is designed to make it easy for users elevate CREST models to running systems that are configurable through a YAML.


