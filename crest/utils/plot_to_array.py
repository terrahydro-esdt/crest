from contextlib import contextmanager
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import warnings
import io


@contextmanager
def plot_to_array(
    dpi        : int   = 150,
    width      : float = 4.8,
    height     : float = 3.6,
    pad_inches : float = 0.05,
):
    """ Save a matplotlib plot to a numpy array for later use.

    Parameters
    ----------
    dpi : int
        Dots per inch (image resolution).
    width : float
        Image width.
    height : float
        Image height.
    pad_inches : float
        Padding to retain around the tight plot bounding box.

    Examples
    --------
    >>> with plot_to_array() as array:
    ...     _ = plt.scatter([1, 2], [2, 1])
    >>> _ = plt.imshow(array)

    """

    # Define the array which will hold the plot image
    tgt_w = int(width * dpi)
    tgt_h = int(height * dpi)
    shape = (tgt_h, tgt_w, 4)
    array = np.empty(shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Define the initial figure and yield the array container for the user
        plt.figure('plot_to_array', (width, height), dpi, 
                   clear=True, layout='constrained')
        yield array

        # Save the current figure using the complete bounds of all plot artists
        fig = plt.gcf()
        with io.BytesIO() as buffer:
            fig.savefig(buffer, **{
                'dpi'         : dpi,
                'pad_inches'  : pad_inches,
                'format'      : 'png',
                'bbox_inches' : 'tight',
            })

            # Read the rendered image and resize while preserving aspect ratio
            buffer.seek(0)
            image = Image.open(buffer).convert('RGBA')
            wp,hp = image.size # pixels
            scale = min(tgt_w / wp, tgt_h / hp)
            new_w = max(1, int(round(wp * scale)))
            new_h = max(1, int(round(hp * scale)))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Center the resized plot in the requested output dimensions
            color = np.round(np.array(fig.get_facecolor())*255).astype('uint8')
            output = Image.new('RGBA', (tgt_w, tgt_h), tuple(color))
            x0 = (tgt_w - new_w) // 2
            y0 = (tgt_h - new_h) // 2
            output.paste(image, (x0, y0))

            # Set the yielded array equal to the final image and clean up
            array[:] = np.asarray(output)
            plt.close(fig)