from __future__ import annotations
from collections import defaultdict as dd
from functools import cached_property, wraps
from itertools import zip_longest, product
import numpy as np
import time
import dask
import re

from .Batcher import Batcher
from .BlockConfig import BlockConfig
from .BatchCombiner import BatchCombiner
from .FutureSampler import FutureSampler
from crest.utils import induce_bins


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
    remove_empty_bins : bool
        If True (default): any bins given to, or created by, the sampling_edges
        parameter will be checked to verify they contain samples (and dropped
        if they do not). This means that it is possible for the actual number
        of bins used during sampling to be less than the number that was given
        or requested. Note that currently, this validation is only applied to
        marginal sampling distributions, as joint distributions may require
        coordinate matching to determine whether a joint bin contains data;
        though, this could be approximated in the future by resampling to a
        common coordinate grid and then checking with sufficiently large radii.
    sampling_edges : list[list | dict]
        Defines a list of configurations to use when generating batches, in the
        same manner as valid_percents and drop_datafiles. Note that all three
        of these parameters must contain the same number of items if they are
        given. A list of configurations given to sampling_edges enables control
        over the distribution of data that is generated when batching. There
        are three formats that can be used to specify a configuration, with a
        'key' referring to either a feature name or a tuple of feature names::

            1. list of keys; e.g. feature names A,B,C: [A, B, (A, C)]
            2. keys dict of bin edges; e.g. {A: [(e_0, e_1), (e_1, e_2)]}
            3. keys dict of induce_bins kwargs; e.g. {A: {'n_bins': 5}}

        For example, if the following list of configurations is given::

            [
              {'feature_a': [(-np.inf, 0), (0, np.inf)],
               'feature_b': [(-np.inf, -1), (-1, 1), (1, np.inf)]},
              {'feature_a': {'n_bins': 5}},
            ]

        this defines two configurations which will each contribute samples to
        every generated batch. The first configuration defines binning schema
        for two features: feature_a samples should be drawn uniformly from two
        bins (negative values, and positive values); and likewise, feature_b
        samples drawn from three bins (<-1, between -1 and 1, >1). In the first
        case, this means 50% of samples generated by this binning schema will
        contain only negative feature_a values, and 50% of samples will contain
        only positive feature_a values. When multiple features are used to set
        schema within a single configuration as is the case in configuration 1,
        the marginal distribution of each feature is used to create samples. In
        other words, one of the five options (2 feature_a + 3 feature_b) will
        be selected every time samples are generated - resulting in the Batcher
        producing an induced distribution which is a mixture of each feature's
        induced marginal and their conditional with respect to the other.

        Rather than inducing marginal distributions when multiple features are
        used in a single configuration, it is also possible to induce the joint
        distribution over several features. For example, instead of treating
        the feature_a and feature_b schemas as independent choices in the first
        configuration, the joint distribution can be drawn from by defining the
        configuration as::

            {('feature_a', 'feature_b'): ([(-np.inf,0), (0,np.inf)],
                                          [(-np.inf,-1), (-1,1), (1,np.inf)])}

        By using a tuple of multiple features as the key, and a tuple of their
        respective binning schema as the value, the cartesian product of schema
        will be selected from when generating samples; i.e. one of six options
        (2 feature_a * 3 feature_b) will be used whenever the Batcher generates
        samples. Concretely: 1/6 of configuration 1 samples will contain (only
        negative feature_a, only <-1 feature_b); 1/6 of samples will contain
        (only positive feature_a, only <-1 feature_b); 1/6 will contain (only
        negative feature_a, only between -1 and 1 feature_b); etc. for all six
        of the schema combinations. Note that this format enables specification
        of multiple (possibly overlapping) joint distributions to pull samples
        from, by simply defining multiple feature tuples in the configuration::

            {('feature_a', 'feature_b'): ...,
             ('feature_a', 'feature_c'): ...,
             ('feature_b', 'feature_d', 'feature_g': ...}

        However, as more features in a joint schema definition increases the
        dimensionality of the induced distribution, observed data points become
        more sparse across bins. This will result in slower sample generation,
        and so should be used with caution.

        In the case of the second configuration, there is only a single feature
        binning scheme that will be used. However, feature_a uses {'n_bins': 5}
        to define what that binning scheme should look like - this dictionary
        is passed as kwargs to crest/utils/induce_bins.py in order to create
        the concrete bin edges which will be used for this feature. The details
        of exactly how those bins are created can be found within the docstring
        of `induce_bins` along with the valid kwargs that can be used - but the
        takeaway for this example is that a list of 5 (left edge, right edge)
        bin tuples will be automatically created for feature_a and used when
        generating samples for this second configuration. The exception to this
        is if the MultiBatcher keyword `remove_empty_bins` is set to True, or
        any generated bins have approximately equal left/right values (i.e. a
        zero-width bin); in which case, the final number of bins to be used in
        sampling will be less than the requested number due to bin removals.

        In addition to defining a list of bin edges directly for a feature, and
        passing a dict of kwargs for automatic generation of the edges, it is
        also valid to simply pass a list of feature names as the configuration
        in order to use the default parameters of induce_bins to create the bin
        edges (e.g. configuration 2 can be equivalently defined as [feature_a],
        and induce_bins will use its default n_bins value to create the list of
        edge tuples).

    *args, **kwargs
        Same as Batcher - see its docstring for available parameters. Note that
        total number of samples contained in each batch will be the requested
        batch_size multiplied by the number of configurations, i.e.::

            batch_size * max(len(valid_percents), len(drop_datafiles))

    """

    def __init__(self,
        *args,
        directed_sampling : bool | None = None,
        remove_empty_bins : bool = True,
        sampling_edges    : list[list | dict] = [],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if directed_sampling is None:
            directed_sampling = self.duplicate
        self.directed_sampling = directed_sampling
        self._init_keys.add('directed_sampling')

        # Directed sampling necessitates sampling blocks multiple times
        assert(not directed_sampling or self.duplicate), (
          'When using directed sampling, Batcher.duplicate must be set to True')

        # Order of blocks is not guaranteed if using directed sampling
        assert(not directed_sampling or self.shuffle), (
          'When using directed sampling, Batcher.shuffle must be set to True')

        # Specifying sampling_edges requires directed sampling
        assert(not sampling_edges or directed_sampling), (
            'When using sampling_edges, directed_sampling must be set to True')

        # Order of blocks is not guaranteed with multiple workers
        assert(self.shuffle or self.workers <= 1), (
          'When using multiple workers, Batcher.shuffle must be set to True')

        # Doesn't make sense to synchronize MultiBatcher workers
        self.block_sync = False

        # There must be a single, universal number of configurations
        configs = kwargs.get('valid_percents', kwargs.get('drop_datafiles', []))
        assert(
            (len(sampling_edges) in [0, len(configs)])
            or (len(configs) == 0)
        ), (f'The number of sampling bin configurations ({len(sampling_edges)}'
            f') must equal the valid_percents/drop_datafiles {len(configs)=}')
        self._set_sampling_edges(sampling_edges, remove_empty_bins)
        self._init_keys.add('sampling_edges')


    def close(self, *args, **kwargs):
        """ Clean up the additional resources allocated by MultiBatcher """
        for key in ['_batch_combiner', '_block_configs', '_sampler']:
            try:    self.__dict__.pop(key, None)
            except: pass  # noqa: S110
        super().close(*args, **kwargs)


    def _get_status(self) -> list[str]:
        """ Include configuration queue information in the status """
        status = [str(self._batch_combiner)] + super()._get_status()

        # Forcing configs to reflect the true queue counts improves resiliency
        self._batch_combiner.update_queued()
        return status


    # def get_queue(self, context):
    #     """ One queue per configuration, encapsulated in a Queue-like API """
    #     return MultiBatcherQueue(self.max_queue,self._batch_combiner,context)


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
        numblocks = self.numblocks
        if isinstance(numblocks, dict):
            options = self.dataset.dims[0]
            unknown = [d for d in numblocks if d not in options]
            assert(len(unknown) == 0), f'Dims {unknown=}; {options=}'
            numblocks = [numblocks.get(d, 1) for d in options]

        kwargs = { 'numblocks' : numblocks,
                   'random'    : self.random,
                   'logger'    : self._logger}
        configs = { 'valid_percents' : self.valid_percents or [None],
                    'drop_datafiles' : self.drop_datafiles or [None],
                    'sampling_edges' : self.sampling_edges or [None]}
        return [BlockConfig(config_index=i,
            **(kwargs | dict(zip(configs.keys(), vals))))
            for i,vals in enumerate(zip_longest(*configs.values()))]


    def _samples_cache_key(self, config, block_idxs) -> str:
        """ Return a unique key for the given configuration """
        return config.sampling_uid(block_idxs)


    @property
    def n_total_config(self) -> int:
        """ Number of block configurations, including binning configs """
        n_edge_configs = sum(max(len(e), 1) for e in self.sampling_edges)
        return max(super().n_total_config, n_edge_configs)


    def _config_index(self, config) -> int:
        """ Index of the configuration within the n_total_config """
        n_totals = sum(map(len, self.sampling_edges[:config.config_index]))
        self.debug(f'{n_totals=} {config.bin_index=} {config=} ' +
                   f'{self.sampling_edges=}{config.config_index=}')
        return n_totals + config.bin_index


    def _config_name(self, index) -> str:
        """ Name of the configuration at the given index """
        total = 0
        config = 0
        for i, bins in enumerate(self.sampling_edges):
            if index >= len(bins):
                total += len(bins)
                index -= len(bins)
                config = i+1
            else: break
        return f'Config {config} Bin {index}'


    def _create_sampler(self, blocks: list) -> FutureSampler:
        """ Create the block sampler """
        # return NonzeroSampler(blocks, self._block_configs, self.random)
        return FutureSampler(blocks, self._block_configs, **{
            'random'     : self.random,
            'batch_size' : self.batch_size,
            'max_queue'  : self.max_queue,
            'exit_flag'  : (lambda: self._exit),
        })


    def _generate_batches(self, blocks: list):
        """ Initialize Sampler if requested """
        for c in self._block_configs:
            c.set_n_blocks(len(blocks))
        if self.directed_sampling and not hasattr(self, '_sampler'):
            self._sampler = self._create_sampler(blocks)
        yield from super()._generate_batches(blocks)

        # After completing an epoch, reduce block counts so that adapting to
        # actual block averages is faster (now that fewer zero blocks remain)
        for c in self._block_configs:
            with c:
                c.n_blocks = c.n_blocks // 2


    def _finish_epoch(self):
        """ Yield remainder samples as the final batch in an epoch """
        # If we repeat over multiple epochs, save remainders for the next epoch
        if not self.repeat and any(map(len, self._remainder.values())):

            # In contrast to Batcher, ingest remainders separately and combine
            size = lambda v: getattr(v, '__len__', lambda: 0)()
            keys = [k for k, v in self._remainder.items() if size(v)]
            ckey = [int(k.split(':')[0]) for k in keys]
            data = map(self._finalize_batch, map(self._remainder.pop, keys))
            yield from zip(ckey, ([d] for d in data))


    @cached_property
    def _batch_combiner(self) -> BatchCombiner:
        """ Object that combines batches from all BlockConfigs together """
        return BatchCombiner(self._block_configs, self.features,
                             self.shuffle, self.seed)


    def _get_block_config(self, blocks, *block_idxs):# -> BlockConfig | None:
        """ Return the best config to use for the next block computation """
        string = '\n\t'.join(map(str, ['']+self._block_configs))
        self.debug(f'All configs: {string}')
        if not self.directed_sampling:
            nonzero = [c for c in self._block_configs
                       if any(c.valid_blocks[i]==1 for i in block_idxs)]
            if len(nonzero):
                # assert(len(nonzero)), (f'No configurations generate samples '
                #                        f'for block {block_idxs}')
                config = min(nonzero)
                self.info(f'Selected {config}')
                # Increment the number of workers when returning
                return blocks, block_idxs, config.add_worker()
            else:
                self.info(f'No configs generate samples for {block_idxs=}')
                return blocks, block_idxs, None

        # Enforce a soft-cap on the number of batches queued for each config
        config = min(self._block_configs)
        count = 0
        while (len(config) > self.max_queue) and not self._exit:
            mincfg = np.argmin(list(map(len, self._block_configs)))
            config = self._block_configs[mincfg]
            if len(config) <= self.max_queue:
                break
            if (count % 100) == 0:
                self.info('All queues full! Waiting for batches to be' +
                          f' pulled: {len(config)=} > {self.max_queue=}')
            count += 1
            time.sleep(0.1)

        if not self._exit:
            config = min(self._block_configs).select_bin()
            # self.debug(f'Selecting block for {config}: '
            #            f'{list(config.valid_index)=}')
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
            self.debug(f'No samples found in block(s) {block_idxs} '
                       f'using {config}')
            config.add_batches(block_idxs, n_batches=-1)
            # self.error(f'{config}: {np.array(config.zero_blocks)}')
            # if len(config.nonzero) == 0:
            #     self.error(f'No blocks generate samples for {config}')
            # else:self.debug(f'{len(config.nonzero)} block options remaining')

        # Add global bin statistics if this is the first compute for a block
        for block_index in zip(blocks, block_idxs):
            config.compute_block_stats(*block_index)
        return samples


    def _batcher(self, samples, config: BlockConfig, block_idxs) -> list:
        """ Update BlockConfig counters and return the hash with each batch """
        batches = super()._batcher(samples, config, block_idxs)

        # Update configuration attributes with newly created batches
        if config is not None and batches is not None:
            self.debug(f'Sending {len(batches)} batches for '
                       f'{config} ({hash(config)=})')
            config.add_batches([], n_batches=len(batches))
            hashval = hash(config)
            batches = [(hashval, batches)]
        return batches


    def _parse_batch(self, batch):
        """ Ingest BlockConfig batches and return combined batched """
        # `batch` should have the format [config_hash, [samples]]
        if (
            isinstance(batch, (list, tuple))
            and len(batch) == 2
            and isinstance(batch[0], int)
            and not isinstance(batch[1], int)
        ):
            # self.debug(f'Received {len(batch[1])} batches for {batch[0]}')
            yield from self._batch_combiner(*batch, method='extend')
            return

        # An unexpected format might be a bug, but we can still just yield it
        message = f'Unexpected batch format: {type(batch)=}'
        if hasattr(batch, '__len__'):
            message += f' {len(batch)=} {list(map(type, batch))[:5]=}'
        self.warning(message)
        yield batch


    def _validate_cache(self, config: BlockConfig):
        """ Warn the user if all blocks for this config produce few samples """
        # Verify that any bin+block pairs marked as empty are None in the cache
        for i, empty in enumerate(list(config.empty_blocks[config.bin_index])):
            if empty:
                key = self._samples_cache_key(config, [i])
                if key not in self._samples_cache:
                    self._samples_cache[key] = []
                assert(self._samples_cache[key] is not None), (
                    f'Previously found > {self.block_size*0.9} '
                    f'samples for {config=} {key=}')
                assert(len(self._samples_cache[key]) == 0), (
                    f'Cache {key=} should be empty: '
                    f'{len(self._samples_cache[key])}')

        if not config.bin_is_empty[config.bin_index]:
            super()._validate_cache(config)


    def _map_to_batch(self, function):
        """ Allows wrapping a function that is applied to pre-queue batches """

        @wraps(function)
        def prequeue_wrapper(batch: tuple):
            """ Handle the (config_hash, batches) format """
            if len(batch) != 2 or not isinstance(batch[0], int):
                message = f'Unexpected batch format: {type(batch)=}'
                if hasattr(batch, '__len__'):
                    message += f' {len(batch)=} {list(map(type, batch))[:5]=}'
                self.warning(message)
            return batch[0], [function(b) for b in batch[1]]
        return prequeue_wrapper


    def _set_sampling_edges(self, sampling_edges: list, remove_empty: bool):
        """ Ensure sampling edge schema are all in a universal format """
        sampling_edges = list(sampling_edges)
        safelen = lambda v: getattr(v, '__len__', lambda: None)()
        summary = None
        self.debug(f'Parsing schema {sampling_edges=}')

        # Warn the user if the given configuration might not be intentional
        #  i.e. very few samples may be generated due to this configuration
        for i, edges in enumerate(sampling_edges):
            features = list(edges)
            for df in self.dataset:
                if not (
                    any(k in df.data_features for k in features)
                    and (max(df.window_total.values()) > 1)
                    and (min(df.valid_percent.values()) >= 1)
                ):
                    continue
                self.warning(f'sampling_edges[{i}] {features=} matches {df},'
                    f' which specifies a window ({df.window_total=}) that'
                    f' requires elements to be valid ({df._valid_percent=}).'
                    ' This means all window values must be contained in a'
                    ' given sampling_edges bin for a sample to be valid. If'
                    ' the intention is actually to generate samples in which'
                    ' any of the sample values fall within a given bin, use a'
                    ' valid_percent specification thatallows < 100% of a'
                    ' sample window to be valid.')

        def parse_schema(feature, schema) -> list[tuple[float, float]]:
            """ Parse a binning schema into the (left, right) edge format """
            # Assume user is just passing in n_bins value
            if isinstance(schema, int):
                schema = {'n_bins': schema}

            # None / empty config is assumed to be empty kwargs dict
            if schema is None or safelen(schema) == 0:
                schema = {}

            # Generate induced distribution bins from the kwargs
            if isinstance(schema, dict):
                nonlocal summary
                if summary is None:
                    summary = self.dataset.summaries()

                # Find all 'pN' summary percentile keys
                pval = lambda k: re.findall(r'^p(\d{1,3})$', k)
                keys = list(filter(pval, summary.statistics.values))
                vals = [int(pval(k)[0]) for k in keys]

                # Sorted percentiles and their respective data keys
                keys_vals = sorted(zip(keys, vals), key=lambda kv: kv[1])
                keys,vals = map(list, zip(*keys_vals))

                # Extract feature quantiles from dataset summary
                if feature not in summary.features and '@' in feature:
                    feature = feature.split('@')[0]
                f_summary = summary.sel(features=feature, drop=True)
                f_dataset = f_summary.to_dataset('statistics')
                quantiles = f_dataset[['min']+keys+['max']].to_array().values

                # Create bin edges that induce the requested distribution
                values = np.array([0] + vals + [100]) / 100
                schema = induce_bins(quantiles, values, **schema)

            # Ensure all items define bins' (left edge, right edge)
            if isinstance(schema, (tuple, list)):
                for edge_lr in schema:
                    if safelen(edge_lr) != 2:
                        raise ValueError(f'{edge_lr=} should be a tuple '+
                                         'of (left edge, right edge)')
            else:
                raise TypeError(f'Unknown {type(schema)=}: {schema}')
            return schema

        # Collect all configs into representations that can apply independently
        for i, config in enumerate(sampling_edges):
            if isinstance(config, (str, tuple)):
                config = [config]

            # Just a list of strings: default args for induce_bins
            if isinstance(config, list):
                if not all(isinstance(c, (str, tuple)) for c in config):
                    raise TypeError('Sampling configs must only contain str '+
                                     f'or tuple if a list is used: {config=}')

                # Convert into dictionary of empty kwargs dicts
                config = {f: {} for f in config}

            # Build {feature: (lo, hi)} schema list for config independence
            independent: list[dict[str, tuple[float, float]]] = []
            for feature, schema in config.items():

                # Joint distribution schema
                if isinstance(feature, tuple):
                    if isinstance(schema, dict):
                        schema = (schema,) * len(feature)
                    if not len(feature) == len(schema):
                        raise ValueError(f'len({feature=}) != {len(schema)=}')
                    schema = [parse_schema(*fs) for fs in zip(feature, schema)]

                    # Add the cartesian product over all schema
                    for s in product(*schema):
                        assert(len(feature) == len(s))
                        independent.append(dict(zip(feature, s)))

                # Marginal distribution schema
                elif isinstance(feature, str):
                    for s in parse_schema(feature, schema):
                        independent.append({feature: s})
                else:
                    raise TypeError(f'Unknown {type(schema)=}: {schema=}')

            # If requested, remove bins which do not actually contain any data
            if remove_empty:
                independent = self._remove_empty(independent)
            sampling_edges[i] = independent
        self.debug(f'Final {sampling_edges=}')
        self.sampling_edges = sampling_edges


    def _remove_empty(self, independent):
        """ Remove configurations that specify bins with no data """
        self.debug(f'Removing empty bins from {independent}')
        # # This version is significantly simpler, but does not short-circuit
        # names = set()
        # tasks = []
        # index = []
        # data = {}

        # # Build list of delayed tasks to check if bins are empty
        # for i, features_edges in enumerate(independent):

        #     # Only check marginal sampling configurations
        #     if len(features_edges) == 1:
        #         [(feature, (lo, hi))] = features_edges.items()
        #         if feature not in data:
        #             data[feature] = self.dataset.get_feature(feature)
        #         in_bin = (data[feature] >= lo) & (data[feature] < hi)
        #         tasks.append(in_bin.sum().data)
        #         index.append(i)
        #         names.add(feature)

        # if len(tasks):
        #     self.debug(f'Computing valid bins for {tuple(names)}...')
        #     valid = dask.compute(*tasks)
        #     independent = [f for i,f in enumerate(independent)
        #                   if i not in index or valid[index.index(i)]]
        #     self.info(f'Kept {sum(valid)}/{len(valid)} {tuple(names)} bins')
        #     print(f'\n{names=} {valid=}\n')
        # return independent

        # This version is significantly faster by short-circuiting `any`, as
        # well as being able to handle joint feature configurations
        names = set()
        tasks = []
        data = {}
        flag = {}

        @dask.delayed
        def sc_any(chunks, intervals, features):
            nonlocal flag

            unresolved = {k:v for k,v in intervals.items()
                          if not flag.get(k, False)}
            if not unresolved:
                return {k: True for k in intervals}

            if all(c.size >= 0 for c in chunks):
                parsed = []
                for c in chunks:
                    if hasattr(c, 'nnz'):
                        c = c.data
                    c = c.ravel()
                    c.sort()
                    parsed.append(c)

                for k, lo_hi in unresolved.items():
                    if flag.get(k, False):
                        continue

                    found = True
                    for i,f in enumerate(features):
                        if found:
                            chunk = parsed[i]
                            lo,hi = lo_hi[f]
                            found &= (np.searchsorted(chunk, hi, 'left') >
                                      np.searchsorted(chunk, lo, 'left') )
                    if found:
                        flag[k] = True
            return {k: flag.get(k, False) for k in intervals}

        # Build list of delayed tasks to check if bins are empty
        f_tasks = dd(dict)
        n_tasks = len(independent)
        for features_edges in independent:
            for feature in features_edges:
                if feature not in data:
                    data[feature] = self.dataset.get_feature(feature)

            edges = tuple(features_edges)
            f_tasks[edges][str(features_edges)] = features_edges
            names.add(edges)

        for features, ind in f_tasks.items():
            blocks = []
            for f in features:
                blocks.append(data[f].data.to_delayed().ravel())
                assert(len(blocks[0]) == len(blocks[-1])), (
                    f'Differing block count for {features}: '
                    f'{list(map(len, blocks))}')
            tasks.append([sc_any(c, ind, features) for c in zip(*blocks)])

        if len(tasks):
            self.debug(f'Computing valid bins for {tuple(names)}...')
            valid = dict(zip(f_tasks, dask.compute(*tasks)))

            def check(ind):
                return any(v[str(ind)] for v in valid[tuple(ind)])

            independent = [i for i in independent if check(i)]
            self.info(f'Kept {len(independent)}/{n_tasks} {tuple(names)} bins')
            # print(f'Kept {len(independent)} / {n_tasks} {tuple(names)} bins')
        return independent
