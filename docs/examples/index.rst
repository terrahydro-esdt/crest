CREST Fundamentals
==================

The Hierarchal Tensor Graph (HTG) is the core object in CREST where the Earth System Model (ESM) is encoded and specified.

In this section we include examples that show how to use the :doc:`CREST HTG <HTG_overview>` and how to perform basic :doc:`HTG computations <HTG_algebra>`.

The HTG :doc:`Keras Node <HTG_keras>` class implements a wrapper around a `Keras` model or layer and the HTG :doc:`Lambda Node <HTG_lambda>` class implements a wrapper around a `lambda` function. In CREST we can also :doc:`serialize <HTG_tofromJSON>` an HTG network graph as a JSON string.

Batch processing is a technique of processing large volumes of data in groups or batches, rather than individually or continuously. The :doc:`batcher demo <batcher_demo>` shows how this is done in CREST. Additionally, the :doc:`data loader package <data_loader_demo>` is responsible for loading data from single or multiple sources.

Metrics are quantitative measures that help evaluate the effectiveness and reliability of models. This is implemented in the :doc:`CREST metrics <metrics_demo>` class.

The :doc:`Archiver <archiver_demo>` class is responsible for inserting the model predictions into an `xArray` dataset at the correct coordinates and save the results to disk.

Applications
------------

One can use CREST to load/train/predict with the well-known :doc:`MNIST dataset <mnist_demo>` as well as with the :doc:`CIFAR-10 dataset <cifar_demo>`.

.. toctree::
   :hidden:

   HTG_overview
   HTG_algebra
   HTG_keras
   HTG_lambda
   HTG_tofromJSON
   batcher_demo
   data_loader_demo
   metrics_demo
   archiver_demo
   mnist_demo
   cifar_demo
