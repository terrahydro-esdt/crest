from .NonzeroSampler import NonzeroSampler
import numpy as np
import time


class FutureSampler(NonzeroSampler):
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
            index = self.random.choice( config.nonzero )
            return [[index, self.blocks[index]]]

    def get_block(self, config):
        index = self.random.choice( config.nonzero )
        return [[index, self.blocks[index]]]