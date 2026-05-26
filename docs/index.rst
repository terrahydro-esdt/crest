======================================================
Coupled Reusable Earth System Tensor (CREST) framework
======================================================



**CREST**  is a Python framework for building AI-enabled Earth system models. Models
are constructed by wiring *Nodes* into a *Hierarchical Tensor Graph* (HTG) — a
directed acyclic graph in which each Node wraps an arbitrary Python callable or Keras
layer. Because a Node is itself an HTG, graphs compose hierarchically, enabling
hybrid architectures that mix neural networks, physical equations, and empirical
relationships at any depth of nesting. The :class:`~crest.model.Model` class compiles
an HTG into a standard Keras model, exposing familiar ``compile``, ``fit``, and ``predict``
workflows.

The data pipeline centers on *Datasets* that load large geospatial files and partition them into
spatially and temporally aligned *Blocks* and *Blocksets*. A built-in coordinate matcher
collocates data from independent grid geometries without manual resampling. Multiprocess
*Batchers* stream balanced mini-batches to the training loop, while the *Archiver* layer writes
model predictions back to disk in Zarr, TileDB, or NetCDF format.

.. mermaid::

   flowchart TD
       subgraph ModelLayer["Model Layer"]
           M["Model"]
           HTG["HierarchalTensorGraph (HTG)"]
           N["Node"]
           RN["RecurrentNode"]
           M -- "wraps & compiles" --> HTG
           N -. "is a leaf HTG" .-> HTG
           RN -- "extends" --> N
       end
       subgraph DataLayer["Data Layer"]
           DS["Dataset / StructuredDataset"]
           BKS["Block & Blockset"]
           BAT["Batcher / MultiBatcher"]
           DS -- "partitions into" --> BKS
           BKS -- "aligned via coordinate matching" --> BAT
       end
       subgraph ArchiveLayer["Archive Layer"]
           ARC["Archiver"]
           STO["Zarr | TileDB | NetCDF"]
           ARC -- "writes to" --> STO
       end
       BAT -- "streams batches to" --> M
       M -- "writes predictions via" --> ARC


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quickstart
   examples/index
   api/index
