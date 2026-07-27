class NonzeroSampler:
    """ Randomly samples blocks that can create > 0 samples.

    Notes
    -----
    This Sampler class performs two steps when it yields a block:
      1. Get the BlockConfig that has the fewest number of queued samples
      2. Randomly pick a block that generates > 0 samples for the chosen config

    Parameters
    ----------
    blocks : list
        The list of block objects that should be selected from when sampling.
    configs : list[BlockConfig]
        The list of BlockConfig objects which track block meta information such
        as the number of samples generated from each block.
    random : numpy.random.Generator
        Random generator used as the source of randomness when sampling blocks.
    exit_flag : multiprocessing.synchronize.Event
        Event object that signals the current workflow should exit immediately.
        
    """
    
    def __init__(self, blocks: list, configs: list, random, exit_flag):
        self.random = random
        self.blocks = blocks
        self.configs = configs
        self.exit_flag = exit_flag

    def __iter__(self):
        """ Yield blocks """
        return self

    def __next__(self):
        """ Yield the block most needed currently from blocks with samples """
        return self.get_block(min(self.configs))

    def get_block(self, config):
        """ Return a block that generates samples for the given BlockConfig.

        Parameters
        ----------
        config : BlockConfig
            The configuration which should be used for sampling a block.

        Returns
        -------
        list[list]
            Returns a nested list object that contains two elements: the index
            of the block which was chosen, and the block itself.

        """
        index = self.random.choice( config.valid_index )
        return [[index, self.blocks[index]]]