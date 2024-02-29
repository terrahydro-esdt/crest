Examples
========

CREST Fundamentals
------------------

The Hierarchal Tensor Graph (HTG) is the core object in CREST where the Earth System Model (ESM) is encoded and specified.

In this section we include examples that show how to use the `CREST HTG`_ and how to perform basic `HTG computations`_.

.. _CREST HTG: HTG_overview.nblink

.. _HTG computations: HTG_algebra.nblink

The HTG `Keras Node`_ class implements a wrapper around a Keras model or layer and the HTG `Lambda Node`_ class implements a wrapper around a lambda function.

.. _Keras Node: HTG_keras.nblink

.. _Lambda Node: HTG_lambda.nblink


Batch processing is a technique of processing large volumes of data in groups or batches, rather than individually or continuously. The `batcher demo`_ shows how this is done in CREST. Additionally, the `data loader package`_ is responsible for loading data from single or multiple sources.

.. _batcher demo: batcher_demo.nblink

.. _data loader package: data_loader_demo.nblink

Metrics are quantitative measures that help evaluate the effectiveness and reliability of models. This is implemented in the `CREST metrics`_ class.

.. _CREST metrics: metrics_demo.nblink

The `Archiver`_ class is responsible for inserting the model predictions into an xarray dataset at the correct coordinates and save the results to disk.

.. _Archiver: archiver_demo.nblink

Applications
------------

One can use CREST to load/train/predict with the well-known `MNIST dataset`_ and well as the with the `CIFAR-10 dataset`_.

.. _MNIST dataset: mnist_demo.nblink

.. _CIFAR-10 dataset: cifar_demo.nblink


The `Soil Moisture Demo`_ is an end-to-end worflow that shows how to use CREST to build/train a model using soil moisture data.

.. _Soil Moisture Demo: soil_moisture_demo.nblink

