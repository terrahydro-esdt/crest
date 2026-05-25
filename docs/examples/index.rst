CREST Fundamentals
==================


**CREST** is a modular machine learning pipeline framework engineered to process, train, and make predictions on high-dimensional **geospatial datasets**. It translates complex data processing blueprints into *Hierarchal Tensor Graphs* which act as workstation assembly lines, wrapping around frameworks like *TensorFlow* and *Keras*.  By partitioning gigantic datasets into aligned *blocks* and matching independent coordinate grids using high-performance *neighbor finder* algorithms, it streams optimized, balanced batches straight to model training loops and archives predictions cleanly to disk.

These examples introduce the Hierarchal Tensor Graph (HTG), which is the core object in CREST where the Earth System Model (ESM) is encoded and specified, and the Batcher, which is the engine that executes the HTG and feeds data to the model training loop. We also use CREST to load/train/predict with the well-known :doc:`MNIST dataset <mnist_demo>` and finally  introduce a more realistic example used to predict soil moisture.

Applications
------------

.. toctree::
   :maxdepth: 1

   HTG_overview
   HTG_algebra
   HTG_recurrent
   metrics_demo
   mnist_demo
