from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import warnings
import io


@contextmanager
def plot_to_array(dpi: int = 120, width: float = 4.8, height: float = 3.6):
    """ Save a matplotlib plot to a numpy array for later use. 
    
    Parameters
    ----------
    dpi : int
        Dots per inch (image resolution).
    width : float
        Image width.
    height : float
        Image height.

    Examples
    --------
    >>> with plot_to_array() as array:
    ...     plt.scatter([1, 2], [2, 1])
    >>> plt.imshow(array)

    """

    # Define the array which will hold the plot image
    shape = (1, int(height*dpi), int(width*dpi), 4)
    array = np.empty(shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plt.clf()
        yield array
    
        # Save the current figure into the previously yielded array object
        with io.BytesIO() as buffer:
            figure = plt.gcf()
            figure.set_size_inches([width, height])
            plt.tight_layout()
            figure.canvas.draw() 
            figure.savefig(buffer, format='raw', dpi=dpi)

            # Clear the figure / close the plot
            figure.clf()
            plt.close(figure)
            plt.close('all')

            # Set the array equal to the image buffer contents
            buffer.seek(0)
            array.ravel()[:] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)