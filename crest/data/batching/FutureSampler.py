from .NonzeroSampler import NonzeroSampler
import numpy as np
import time


class FutureSampler(NonzeroSampler):
    """ Chooses a BlockConfig that has the minimum expected samples.

    Notes
    -----
    Rather than choosing a BlockConfig that has the least number of samples
    currently queued (like NonzeroSampler), this sampler takes into account
    the number of samples expected to be queued in the future. As well, it
    handles the logic of waiting until a queue has room for more samples if
    all BlockConfig queues are currently full.

    Parameters
    ----------
    *args
        See NonzeroSampler for the standard Sampler arguments.
    batch_size : int
        Number of samples contained in a batch.
    max_queue : int
        Max number of samples that should be contained in a BlockConfig queue.
    **kwargs
        See NonzeroSampler for the standard Sampler arguments.

    """
    
    def __init__(self, *args, batch_size: int=32, max_queue: int=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_queue = max_queue

    def __next__(self):
        """ Yield the block most needed currently from blocks with samples """
        config = min(self.configs)
        while (len(config) > self.max_queue) and not self.exit_flag():
            config = self.configs[np.argmin(list(map(len, self.configs)))]
            if len(config) <= self.max_queue:
                break
            config = min(self.configs)
            time.sleep(0.1)
        if not self.exit_flag():
            return self.get_block(config)