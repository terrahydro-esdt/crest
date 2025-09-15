class NonzeroSampler:
    def __init__(self, blocks: list, configs: list, random, exit_flag):
        self.random = random
        self.blocks = blocks
        self.configs = configs
        self.exit_flag = exit_flag

    def __iter__(self):
        return self

    def __next__(self):
        """ Yield the block most needed currently from blocks with samples """
        index = self.random.choice( min(self.configs).nonzero )
        return [[index, self.blocks[index]]]

    def get_block(self, config):
        return next(self)