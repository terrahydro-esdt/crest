from queue import Empty
import numpy as np
import time 


class MultiBatcherQueue:
    """ Replicates multiprocessing Queue API while wrapping multiple Queues.

    Notes
    -----
    This class handles abstracting an interface over multiple Queues in order
    to allow each BlockConfig to pass samples via its own independent pipe.

    Parameters
    ----------
    size : int
        Size of the Queue (same for every Queue object).
    combiner : BatchCombiner
        The BatchCombiner object in the current MultiBatcher pipeline that is
        handling combining batches sourced from different BlockConfigs.
    context : 

    """
    
    def __init__(self, size: int, combiner, context):
        self.combiner = combiner
        self.queues = {c: context.Queue(size) for c in combiner.configs}
        self._buffer = []

    
    def get(self, timeout: float=0.):
        sizes = {c:len(b) for c,b in self.combiner.batches.items()}
        # hs, qs = zip(*sorted(self.queues.items(), key=lambda cq: self.combiner.configs[cq[0]]))# sizes[cq[0]]))
        hs, qs = zip(*sorted(self.queues.items(), key=lambda cq: sizes[cq[0]]))
        for q in qs[:-1]:
            try:
                return q.get_nowait()
            except Empty: pass
        return qs[-1].get(timeout=timeout)
            # return np.random.choice(qs[1:]).get(timeout=timeout)

            
    def put(self, batch):
        return self.put_nowait(batch)

    
    def put_nowait(self, batch):
        if (len(batch) == 2) and isinstance(batch[1], int):
            self.queues[batch[1]].put_nowait(batch)

    
    def full(self):
        time.sleep(0.01)
        return False

    
    def cancel_join_thread(self):
        for q in self.queues.values():
            q.cancel_join_thread()

    
    def empty(self):
        return all(q.empty() for q in self.queues.values())

    
    def get_nowait(self):
        for q in self.queues.values():
            while not q.empty():
                try:    q.get_nowait()
                except: break

    def qsize(self):
        return [q.qsize() for q in self.queues.values()]