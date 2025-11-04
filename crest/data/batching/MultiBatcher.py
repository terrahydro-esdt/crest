from functools import cached_property
from itertools import zip_longest
from queue import Empty
import numpy as np
import time 

from .Batcher import Batcher
from .BlockConfig import BlockConfig
from .BatchCombiner import BatchCombiner
from .NonzeroSampler import NonzeroSampler
from .FutureSampler import FutureSampler
from .MultiBatcherQueue import MultiBatcherQueue


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
    directed_sampling : bool
        Whether blocks should be chosen in a directed manner. If False, blocks
        and configurations are sampled uniformly at random (with configurations
        only filtered by whether they produce > 0 samples for the current block
        a worker is computing). If True (default), the speed of batch creation
        can be significantly increased by choosing blocks and configurations to
        maximize the number of expected future batches. This is accomplished by
        first choosing a configuration based on the current batch expectations,
        then uniformly sampling all blocks that produce > 0 samples for that
        configuration. While further optimization is possible by performing a
        weighted sampling over the blocks based on the number of samples they
        produce for the chosen configuration, this strategy would introduce a
        risk of bias in which samples are generated (e.g. high-density regions
        would be over-sampled relative to low-density regions). By uniformly
        sampling over all blocks that produce any samples, we avoid this bias.
    duplicate : bool
        Whether to duplicate blocks across workers (same functionality as the
        parameter in Batcher). In contrast to Batcher, duplication is True by
        default since there are multiple configurations to run for each block.
        With duplicate=False, the expected number of blocks computed in a data
        epoch for a given configuration will be `n_blocks // n_configurations`.
        Duplicating blocks across workers mitigates this, since the number of
        blocks computed in total will be `n_blocks * n_workers` and the percent
        of blocks per configuration is then `n_workers / n_configurations`. 
    *args, **kwargs
        Same as Batcher - see its docstring for available parameters. Note that
        total number of samples contained in each batch will be the requested
        batch_size multiplied by the number of configurations, i.e.::

            batch_size * max(len(valid_percents), len(drop_datafiles))

    """
    
    def __init__(self, *args, directed_sampling=True, duplicate=True, **kwargs):
        super().__init__(*args, duplicate=duplicate, **kwargs)
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

    
    def _get_status(self) -> list[str]:
        """ Include configuration queue information in the status """
        status = [str(self._batch_combiner)] + super()._get_status()
        
        # Forcing configs to reflect the true queue counts improves resiliency
        self._batch_combiner.update_queued()
        return status

    
    # def get_queue(self, context):
    #     """ One queue per configuration, encapsulated in a Queue-like API """
    #     return MultiBatcherQueue(self.max_queue, self._batch_combiner, context)


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
            # self._sampler = NonzeroSampler(blocks, self._block_configs, self.random)
            self._sampler = FutureSampler(blocks, self._block_configs, **{
                'random'     : self.random, 
                'batch_size' : self.batch_size, 
                'max_queue'  : self.max_queue, 
                'exit_flag'  : (lambda: self._exit),
            })
        yield from super()._generate_batches(blocks)
        
        # After completing an epoch, reduce block counts so that adapting to
        # actual block averages is faster (now that fewer zero blocks remain)
        for c in self._block_configs:
            with c:
                c.n_blocks.value = c.n_blocks.value // 2

    
    def _finish_epoch(self):
        """ Yield remainder samples as the final batch in an epoch """
        # If we repeat over multiple epochs, save remainders for the next epoch
        if not self.repeat and any(map(len, self._remainder.values())):

            # In contrast to Batcher, ingest remainders separately and combine
            size = lambda v: getattr(v, '__len__', lambda: 0)()
            keys = [k for k, v in self._remainder.items() if size(v)]
            data = map(self._finalize_batch, map(self._remainder.pop, keys))
            list(map(self._batch_combiner, keys, data))
            yield from self._batch_combiner
            
    
    @cached_property
    def _batch_combiner(self) -> 'BatchCombiner':
        """ Object that combines batches from all BlockConfigs together """
        return BatchCombiner(self._block_configs, self.features, self.shuffle, self.seed)


    def _get_block_config(self, blocks, *block_idxs):# -> BlockConfig | None:
        """ Return the best config to use for the next block computation """
        string = '\n\t'.join(map(str, ['']+self._block_configs))
        self.debug(f'All configs: {string}')
        if not self.directed_sampling:
            nonzero = [c for c in self._block_configs if any(c.valid_blocks[i]==1 for i in block_idxs)]
            assert(len(nonzero)), f'No configurations generate samples for block {block_idxs}'
            config = min(nonzero)
            self.info(f'Selected {config}')
            return blocks, block_idxs, config.add_worker() # Increment the number of workers            
            
        # Enforce a soft-cap on the number of batches queued for each config
        config = min(self._block_configs)
        count = 0
        while (len(config) > self.max_queue) and not self._exit:
            config = self._block_configs[np.argmin(list(map(len, self._block_configs)))]
            if len(config) <= self.max_queue:
                break
            if (count % 100) == 0:
                self.info(f'All queues full! Waiting for batches to be' +
                          f' pulled: {len(config)=} > {self.max_queue=}')
            count += 1
            time.sleep(0.1)

        if not self._exit:
            # Maximizing n_expected avoids all blocked workers choosing one config
            config = min(self._block_configs)
            block_idxs, blocks = zip(*self._sampler.get_block(config))
            self.info(f'Selected {config} for {block_idxs=}')
            return blocks, block_idxs, config.add_worker()
        return blocks, block_idxs, None

    
    # def _blocker(self, blocks):
    #     """ Use the Sampler to select new blocks """
    #     if self.directed_sampling:
    #         blocks = next(self._sampler)
    #     return super()._blocker(blocks)


    def _compute_block(self, blocks, config: BlockConfig, block_idxs):
        """ Update BlockConfig counters when no samples are found """
        samples = super()._compute_block(blocks, config, block_idxs)
        if not samples.size: 
            self.debug(f'No samples found in block(s) {block_idxs} using {config}')
            config.add_batches(block_idxs, n_batches=0)
            # self.error(f'{config}: {np.array(config.zero_blocks)}')
            if len(config.nonzero) == 0:
                self.error(f'No blocks generate samples for {config}')
            else: self.debug(f'{len(config.nonzero)} block options remaining')
        return samples


    def _batcher(self, samples, config: BlockConfig, block_idxs) -> list:
        """ Update BlockConfig counters and return the hash with each batch """
        batches = super()._batcher(samples, config, block_idxs)
        
        # Update configuration attributes with newly created batches
        if config is not None and batches is not None:
            self.debug(f'Sending {len(batches)} batches for {config}')
            config.add_batches([], n_batches=len(batches))
            hashval = hash(config)
            batches = [(hashval, batches)]
        return batches


    def _parse_batch(self, batch):
        """ Ingest BlockConfig batches and return combined batched """
        # `batch` should have the format [config_hash, [samples]]
        if isinstance(batch, (list, tuple)) and len(batch) == 2: 
            if isinstance(batch[0], int) and not isinstance(batch[1], int):
                self.debug(f'Received {len(batch[1])} batches for {batch[0]}')
                yield from self._batch_combiner(*batch, method='extend')
                return

        # An unexpected format might be a bug, but we can still just yield it
        message = f'Unexpected batch format: {type(batch)=}'
        if hasattr(batch, '__len__'):
            message += f' {len(batch)=} {list(map(type, batch))[:5]=}'
        self.info(message)
        yield batch