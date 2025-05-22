from functools import cached_property
from itertools import zip_longest
from queue import Empty

from .Batcher import Batcher
from .BlockConfig import BlockConfig
from .BatchCombiner import BatchCombiner
from .NonzeroSampler import NonzeroSampler


class MultiBatcher(Batcher):
    """
    Notes
    -----
    Assume we have multiple Datafiles in a given Dataset, where some Datafiles
    do not overlap at all with others; e.g. FLUXNET tower locations have no 
    overlap with SNOTEL tower locations. 

    We want to train a coupled or multihead model which has both FLUXNET and 
    SNOTEL outputs, and so would like batches to contain samples representing 
    both FLUXNET and SNOTEL (e.g. 64 samples of FLUXNET and 64 of SNOTEL in a 
    128 sample batch). 

    To do so we could create multiple Batchers which individually generate 
    FLUXNET and SNOTEL batches, then concatenate the output batches together 
    before feeding into the model. However, this leads to worker process 
    inefficiencies, large memory overhead, I/O issues, block duplication, and 
    generally complicated code.

    Instead, MultiBatcher (which can be used as a drop-in replacement for 
    Batcher) allows us to use only a single object to handle this scenario.
    MultiBatcher handles the orchestration of combining batches together from
    different worker processes, while also allowing its workers to be allocated
    in a dynamic way towards the different Dataset configurations:

        1. Worker needs a new configuration to use for batch creation
        2. MultiBatcher calculates the expected number of samples available
            from each configuration (based on current availability and the
            status of other worker processes)
        3. MultiBatcher assigns Worker the configuration which has the smallest
            number of expected samples, thus maximizing the expected number of
            combined batches that can be produced. 

    In this way, worker processes are able to be dynamically assigned work, in
    order to maximize the speed at which combined batches are created - even if
    the individual Dataset configurations being used require different amounts 
    of time and effort to create their respective batches.

    Parameters
    ----------
    *args, **kwargs
        Same as Batcher - see its docstring for available parameters. Note that
        total number of samples contained in each batch will be the requested
        batch_size multiplied by the number of configurations, i.e.:
            `batch_size * max(len(valid_percents), len(drop_datafiles))`

    """
    def __init__(self, *args, directed_sampling=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.directed_sampling = directed_sampling
        self._init_keys.add('directed_sampling')

        # Order of blocks is not guaranteed if using directed sampling
        assert(not directed_sampling or self.shuffle), (
          'When using directed sampling, Batcher.shuffle must be set to True')

        # Order of blocks is not guaranteed with multiple workers
        assert(self.shuffle or self.workers <= 1), (
          'When using multiple workers, Batcher.shuffle must be set to True')

        # Doesn't make sense to synchronize MultiBatcher workers 
        self.block_sync = False


    def close(self, *args, **kwargs):
        """ Clean up the additional resources allocated by MultiBatcher """
        for key in ['_batch_combiner', '_block_configs']:
            try:    self.__dict__.pop(key, None)
            except: pass
        super().close(*args, **kwargs)


    @property
    def _first(self) -> bool:
        """ Don't wait for first block since combined batches need multiple """
        return True


    @property
    def _shared_attrs(self) -> dict:
        """ Include the BlockConfigs in attributes shared across processes """
        return super()._shared_attrs | {'_block_configs': self._block_configs}


    @cached_property
    def _block_configs(self) -> list[BlockConfig]:
        """ Creates a list of BlockConfigs for the specified configurations """
        configs = { 'valid_percents' : self.valid_percents or [None],
                    'drop_datafiles' : self.drop_datafiles or [None], }
        return [BlockConfig(numblocks=self.numblocks, **dict(zip(configs.keys(), vals))) 
                                    for vals in zip_longest(*configs.values())]


    def _generate_batches(self, blocks: list):
        """ Initialize Sampler if requested """
        for c in self._block_configs:
            c.set_n_blocks(len(blocks))
        if self.directed_sampling:
            self._sampler = NonzeroSampler(blocks, self._block_configs, self.random)
        return super()._generate_batches(blocks)


    @cached_property
    def _batch_combiner(self) -> 'BatchCombiner':
        """ Object that combines batches from all BlockConfigs together """
        return BatchCombiner(self._block_configs,self.features,self.batch_size)


    def _get_block_config(self, *block_idxs) -> BlockConfig | None:
        """ Return the best config to use for the next block computation """
        nonzero = [c for c in self._block_configs if any(c.valid_blocks[i]==1 for i in block_idxs)]
        if nonzero:
            config = min(nonzero)
            string = '\n\t'.join(map(str, ['']+self._block_configs))
            self.info(f'Selected {config}')
            self.debug(f'All configs: {string}')
            return config.add_worker() # Increment the number of workers
        self.info(f'No configurations generate samples for block {block_idxs}')

    
    def _blocker(self, blocks):
        """ Use the Sampler to select new blocks """
        if self.directed_sampling:
            blocks = next(self._sampler)
        return super()._blocker(blocks)


    def _compute_block(self, blocks, config: BlockConfig, block_idxs):
        """ Update BlockConfig counters when no samples are found """
        samples = super()._compute_block(blocks, config, block_idxs)
        if not samples.size: 
            self.debug(f'No samples found in block using {config}')
            config.add_samples(block_idxs, n_samples=0)
            # self.error(f'{config}: {np.array(config.zero_blocks)}')
            if len(config.nonzero) == 0:
                self.error(f'No blocks generate samples for {config}')
        return samples


    def _batcher(self, samples, config: BlockConfig, block_idxs) -> list:
        """ Update BlockConfig counters and return the hash with each batch """
        batches = super()._batcher(samples, config, block_idxs)

        # Update configuration attributes with newly created batches
        if config is not None:
            config.add_samples(block_idxs, n_samples=len(batches) * self.batch_size)
            hashval = hash(config)
            batches = [(batch, hashval) for batch in batches]
        return batches


    def _parse_batch(self, batch):
        """ Ingest BlockConfig batches and return combined batched """
                # Remainder samples aren't paired with any configuration
        if (len(batch) == 2) and isinstance(batch[1], int):
            batch = self._batch_combiner(*batch).next()
        return batch