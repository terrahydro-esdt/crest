import dask.array as da
import numpy as np 

from crest.src.base import BaseAbstract



class Batcher(BaseAbstract):
    """Handles creating batches of data samples. 

    Parameters
    ----------
    samples    : np.ndarray
        Numpy array of Sample objects, which should be generated from
        the Crest data loader by calling Dataset.generate_samples(). 
    batch_size : int
        Number of samples that each batch should contain. 
    shuffle    : bool 
        Whether samples should be shuffled to generate batches, or 
        returned in the original order of the sample array. 
    seed       : int | None
        Seed for reproducible randomness. 
    
    """
    def __init__(self, 
        samples    : da.Array,
        batch_size : int,
        shuffle    : bool = False,
        seed       : int | None = None,
    ):
        self.samples  = samples 
        self.n_batch  = batch_size 
        self.shuffle  = shuffle
        self.random   = np.random.default_rng(seed)
        self.drandom  = da.random.RandomState(seed)


    def __iter__(self):
        """ Iterate over batches of samples """
        # Calculate the number of blocks to operate over at once
        blocks  = np.arange(self.samples.blocks.size, dtype='int32')
        n_block = max(3, 1e5 // self.samples.chunksize[0]) 
        n_split = max(1, len(blocks) // n_block)
        if self.shuffle: self.random.shuffle(blocks)

        # Iterate over the subsets of blocks
        batch = []
        for subset in np.array_split(blocks, n_split):
            samples = self.samples.blocks[subset]
            if self.shuffle: samples = self.drandom.permutation(samples)

            # Iterate over the blocks in the given blocks subset
            for block in samples.blocks:
                block = block.compute()
                count = self.n_batch - len(batch)
            
                # Yield the remainder batch from the prior subset
                if len(batch) < self.n_batch:
                    batch = np.append(batch, block[:count], 0)
                    if len(batch) >= self.n_batch:
                        yield batch

                # Chunk the rest of this block into the requested batch size
                for i in range(count, len(block), self.n_batch):
                    batch = block[i:i+self.n_batch]
                    if len(batch) >= self.n_batch:
                        yield batch

        # Yield any remainder batch at the end if it exists
        if len(batch) < self.n_batch: yield batch
