CREST Examples
==============

The following examples introduce the main CREST components. We start by showing how
to construct and use a `CREST HTG`_, then demonstrate how to perform basic
`HTG computations`_.

.. _CREST HTG: HTG_overview.nblink

.. _HTG computations: HTG_algebra.nblink

CREST also supports `recurrent HTGs`_, in which a Node is unrolled across a fixed
number of time steps. This enables sequence modeling and temporal architectures
while staying entirely within the HTG composition model.

.. _recurrent HTGs: HTG_recurrent.nblink

Large geospatial datasets rarely fit in memory all at once. The `batcher demo`_
shows how CREST partitions data into aligned Blocks and streams balanced
mini-batches to the training loop. The `data loader demo`_ covers loading data
from single or multiple sources using the Dataset and StructuredDataset classes.

.. _batcher demo: batcher_demo.nblink

.. _data loader demo: data_loader_demo.nblink

Metrics are quantitative measures that help evaluate model effectiveness and
reliability. The `CREST metrics`_ demo shows how to compute and log them during
training and evaluation.

.. _CREST metrics: metrics_demo.nblink

The `Archiver`_ class inserts model predictions into an xarray Dataset at the
correct coordinates and writes the results to disk.

.. _Archiver: archiver_demo.nblink

Finally, we use CREST to load, train, and predict with the well-known
`MNIST dataset`_, bringing the HTG model layer and data pipeline together in a
single end-to-end example.

.. _MNIST dataset: mnist_demo.nblink


Applications
------------

.. toctree::
   :maxdepth: 1

   HTG_overview
   HTG_algebra
   HTG_recurrent
   batcher_demo
   data_loader_demo
   metrics_demo
   archiver_demo
   mnist_demo
