from collections.abc import Collection
from functools import partial
import numpy as np


def optimize_blocks(
    current : Collection[Collection[int]], 
    target  : Collection[int],
    maximum : Collection[int],
) -> Collection[int]:
    """Parsimonious Mixed-Integer Nonlinear optimization.

    Notes
    ----- 
    Uses a greedy approach to optimize the number of blocks per axis, 
    attempting to get as close as possible to the targeted total number 
    of blocks without dividing them along non-integer boundaries (as this
    would duplicate / discard far more data when generating blocks). 
    Integer boundaries are maintained by using only integer multiples of 
    the original block structure. 
    The overall strategy is to simply use the problem boundaries (0, maximum]
    as initial conditions, and perform a greedy optimization at each step
    until a state is repeated. Once this has been performed for all initial
    conditions, return the minimum cost state that was found.
    
    Parameters
    ----------
    current : Collection[Collection[int]]
        The current block structure, e.g. [[1, 5, 5], [2, 3, 5]] (which would
        mean the first Datafile has the first dimension contained in 1 block,
        the second dimension contained in five blocks, and the third dimension
        contained in five blocks; and similarly for the second Datafile, with
        the respective 2, 3, 5 block scheme).
    target  : Collection[int]
        The requested block structure, e.g. [5, 3, 3].
    maximum : Collection[int]
        The maximum number of blocks that should be created along each
        dimension, e.g. [10, 20, 15]. 

    Returns
    -------
    Collection[int]
        The block scheme that was found, which optimizes the given constraints
        while being as close as possible to the requested target scheme. 

    """
    # Get the initial state, which is the minimum between the current 
    # dataset(s) states and the maximum allowable size
    init = np.min([np.lcm.reduce(current, axis=0), maximum], axis=0)
    best = dict()
    seen = set()

    # Optimization uses the running seen/best objects
    optimize = partial(_optimize, **{
        'target'  : target,
        'maximum' : maximum,
        'step'    : init, 
        'best'    : best, 
        'seen'    : seen,
    })

    # First optimize using the current blocks as the initial state
    best, seen = optimize(init)

    # Then if we're increasing the number of blocks, optimize with a new
    # state for each axis in the data (increasing the blocks to the max
    # targeted/allowed size for each respective axis)
    if np.prod(init) < target:
        for i, c in enumerate(init):
            targeted = target // np.prod(init)
            allowed  = maximum[i] // c
            incstate = init.copy()
            incstate[i] *= max(1, min(targeted, allowed))
            best, seen = optimize(incstate)

    # Otherwise the current number of blocks needs to be decreased along
    # one or more dimensions (which is done by evenly combining existing
    # blocks using their prime factors)
    # The way this is currently set up will artificially limit the lower
    # end of block sizes, as only 2,3,5 are used as (single) divisors.
    # Eventually we need to remove this limit to allow more control over
    # blocks via blocksize.
    else: 
        for dim in range(len(init)):
            primefac = [[f for f in [2, 3, 5] if c%f == 0] for c in init]
            skipdims = set()
            decstate = init.copy() 

            # Reduce the number of blocks as much as necessary, iterating
            # repeatedly over axes until either the number of blocks fall
            # below the target or we run out of prime factors
            while np.prod(decstate) > target:
                for factor in primefac[dim]:
                    if (decstate[dim] % factor) == 0:
                        decstate[dim] = init[dim] // primefac[dim].pop(0)
                        skipdims.add(dim)
                        break    
                    else: primefac[dim].pop(0)

                dim = (dim + 1) % len(decstate)
                if max(map(len, primefac)) == 0: break

            # Any axes which are reduced need to be skipped over when 
            # optimizing, as it would otherwise allow non-integer multiples
            # with respect to the original block sizes
            best, seen = optimize(decstate, skip=skipdims)

    # Return the best (lowest cost) state that was found
    return best[min(best)]



def _optimize(state, target, maximum, step, best, seen, skip=[]):
    """ Greedy optimization starting with `state` as the initial """
    state = state.copy()

    # Store the initial block state we're searching from
    state_tuple = best[abs(target - np.prod(state))] = tuple(state)

    # Iteratively add multiples of the original block sizes until we
    # reach the total block target, or we encounter a state seen before
    while (state_tuple not in seen) and (np.prod(state) != target):

        # If the current block combination has been seen already,
        # we don't need to try and optimize with it again
        seen.add(state_tuple)

        # The order of preference for steps that increase or decrease 
        # block size depends on the direction to the total target size
        order = 1 if np.prod(state) < target else -1

        # Each axis can step up or down, but must fulfill the condition
        # 0 < new size < max. If the preferred step direction fails, we
        # can instead use the alternative direction choice as backup
        st1,st2 = [step * inc_dec for inc_dec in [1,-1][::order]]
        isvalid = lambda new_state: (0<new_state)&(new_state<=maximum)
        backups = np.where(isvalid(state + st2), st2, np.nan)
        stepdir = np.where(isvalid(state + st1), st1, backups)

        # Helpers:
        # - keep the step choice if finite and not part of skip
        # - swap a given index in the state with a new value
        # - calculate the difference between target and the new state 
        keep = lambda i,s: (i not in skip) and np.isfinite(s)
        swap = lambda i,s: [v + s*(j==i) for j,v in enumerate(state)]
        diff = lambda i,s: (i, abs(target - np.prod(swap(i, s))))

        # Calculate distance to the target for each valid state option
        options = [diff(*s) for s in enumerate(stepdir) if keep(*s)]
        if len(options) == 0: break

        # Select the state which minimizes distance to the target
        idx, option = min(options, key=lambda i_o: i_o[1])
        state[idx] += stepdir[idx]
        state_tuple = best[option] = tuple(state)

    seen.add(state_tuple)
    return best, seen
