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
    fsize = (width, height)
    shape = (1, int(height*dpi), int(width*dpi), 4)
    array = np.empty(shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Define the initial figure and yield the array container for the user
        plt.figure('plot_to_array', fsize, dpi, clear=True, layout='constrained')
        yield array
    
        # Save the current figure into the previously yielded array object
        with io.BytesIO() as buffer:

            # Perform initial canvas draw to place artists
            figure = plt.gcf()
            figure.canvas.draw() 

            # Find the tight bounding box and apply it to the figure
            bbox = figure.get_tightbbox(figure.canvas.get_renderer())
            figure.set_size_inches([bbox.width, bbox.height], forward=True)
            figure.set_dpi(dpi)

            # Re-draw artists to update placement and get h/w in pixels
            figure.canvas.draw()
            w_px, h_px = figure.canvas.get_width_height()
            figure.savefig(buffer, format='raw', dpi=dpi, bbox_inches=bbox)

            # Read the buffer to create an image array from the saved figure
            buffer.seek(0)
            img = np.frombuffer(buffer.getvalue(), dtype=np.uint8)            
            img = img.reshape((h_px, w_px, 4))
            
            # Clear and close the figure, then reinitialize with proper sizes
            # figure.clf()
            plt.close(figure)
            figure = plt.figure('plot_to_array', figsize=fsize, clear=True)
            ax = figure.add_axes([0, 0, 1, 1])
            ax.set_axis_off()

            # Compute conversion from current image size to the targeted size
            pad_px = 0
            tgt_w = max(1, width*dpi - 2 * pad_px)
            tgt_h = max(1, height*dpi - 2 * pad_px)
            scale = min(tgt_w / w_px, tgt_h / h_px)
            new_w = max(1, int(round(w_px * scale)))
            new_h = max(1, int(round(h_px * scale)))
        
            # Use the conversion to plot the final image in the proper size
            ax.set_xlim(0, width*dpi)
            ax.set_ylim(height*dpi, 0)
            x0 = (width*dpi - new_w) / 2
            y0 = (height*dpi - new_h) / 2
            ax.imshow(img, extent=(x0, x0 + new_w, y0 + new_h, y0))

            # Reset the buffer and use it to plot the final image
            buffer.seek(0)
            buffer.truncate(0)
            figure.savefig(buffer, format='raw', dpi=dpi)

            # Set the array equal to the new buffer contents, and clean up
            buffer.seek(0)
            array.ravel()[:] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            plt.close(figure)