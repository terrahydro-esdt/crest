from collections.abc import Collection
import numpy as np
import logging 

from crest.utils.print_table import print_table
from crest.utils.Stopwatch import Stopwatch
from .Grid import Grid
from .Pair import Pair


def brute(
    coordinates : Collection[np.ndarray], 
    resolutions : Collection[np.ndarray],
    grid_labels : Collection[str],
    axis_labels : Collection[str],
    radius      : float = 0.5,
    eps         : float = 1e-5,
    logger      : logging.Logger | None = None,
) -> np.ndarray:

    # Create the grid objects
    format_res = lambda r: (np.moveaxis(r, -1, 0) if r.ndim == 3 else r) * radius + eps
    keys, vals = zip(*{
        'coordinates' : coordinates, 
        'resolutions' : [format_res(r) for r in resolutions], 
        'index'       : range(len(coordinates)),
        'name'        : grid_labels, 
        'dims'        : axis_labels,
    }.items())
    grids = [Grid(**dict(zip(keys, v))) for v in zip(*vals)]

    # Optimize column and grid orderings
    optimize = True
    debug = False  

    if optimize: 
        sizes = lambda G: float(f'{G.resolutions.size}.{G.coordinates.size}')
        order = np.argsort(list(map(sizes, grids)))[::-1]
    else: order = list(range(len(grids)))
    grids = [grids[i] for i in order]

    if debug:
        print('\nShapes:')
        with print_table(['Grid', 'Coordinates', 'Resolutions']) as printer:
            for G in grids: 
                printer(G, G.coordinates.shape, G.resolutions.shape)

        for i, (label, values) in enumerate([
            ('Coordinate', [G.coordinates    for G in grids]), 
            ('Resolution', [G.resolutions[0] for G in grids]),
        ]):
            print(f'\n{label} Extents:')
            with print_table(['Grid', 'Axis', 'Minim', 'Maxim']) as printer:
                for G, v in zip(grids, values):
                    for d, minim, maxim in zip(G.dims, v.min(0), v.max(0)):
                        if (d == 'datetime') and np.isfinite(v).all():
                            otype = np.timedelta64 if i else np.datetime64
                            minim = otype(int(minim), 'm')
                            maxim = otype(int(maxim), 'm')
                        printer(G, d, minim, maxim)
                        G = ''

        print(f'\nStarting neighbor search with {grids[0]}: {len(grids[0]):,} possible matches')

    if logger is not None:
        logger.debug(f'\tStarting neighbor search with {grids[0]}: {len(grids[0]):,} possible matches')

    for i in range(1, len(order)):
        if debug:
            iter_timer = Stopwatch(f'Iteration {i}/{len(order)-1}')
            iter_timer.__enter__()
            print(f'\n{iter_timer}')
            print(''.join(['-']*len(str(iter_timer))))

            print('\nSelecting pair from:')
            with print_table(['Grid', 'Coordinates', 'Resolutions']) as printer:
                for G in grids: 
                    printer(G, G.coordinates.shape, G.resolutions.shape)

        pair = Pair.select_pair(grids)
        ix1, ix2 = pair.match_indices(logger, optimize, debug)

        if ix1.size == 0:
            return np.empty((0, 0))

        grids.append( pair.combined_grid(ix1, ix2, deduplicate=len(grids) > 0) )
        if debug:
            iter_timer.__exit__()
            print()

    # Reorder the columns correctly 
    if debug: print('\nReordering table...')
    table = grids[0].table
    order = grids[0].index
    return table[:, np.argsort(order)]
