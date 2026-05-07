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
        return self.get_block(min(self.configs))

    def get_block(self, config):
        index = self.random.choice( config.valid_index )
        return [[index, self.blocks[index]]]