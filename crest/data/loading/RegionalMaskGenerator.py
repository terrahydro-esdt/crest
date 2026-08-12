from __future__ import annotations
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import numpy as np
import pandas as pd
import logging
import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
import cv2
import shapely
from scipy.interpolate import LinearNDInterpolator
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionalMaskGenerator():
    """ RegionalMaskGenerator take one or more regions as input and provided
    multple functions to generate a mask corresponding to those regions. The
    mask generator works primarily with GeoDataFrames. It uses the osmnx library
    to create dataframes on the fly based on the natural language regional
    names that are provided.
    """

    def __init__(self, regions: list[str]):
        """ Users must provide a list of one or more natural language names of
        different regions in order for this library to generate a mask. By default
        all regions are combined to generate a single mask.

        Parameters
        ----------
        regions: list(str)
            This is a list of natural language names corresponding to region over which
            the user would like the generate a mask.
        """

        logger.info('Initializing RegionalMaskGenerator')

        self.regions = regions
        self.mask = None

        self.gdf = self.load_by_name()

    @staticmethod
    def load_shapefile(shapefile_path: str):
        """ Loads the shapefile into a GeoDataFrame. """

        try:
            return gpd.read_file(shapefile_path)
        except Exception as e:
            logger.exception(f'Could not load shapefile: {e}')
            raise e

    @staticmethod
    def select_subset(regions: list[str], superset: GeoDataFrame):
        """ Selects a subregion the region specified by the superset GeoDataFrame.

        Parameters
        ----------
        regions: list[str]
            List of string natural language names of the regions the user wishes to select
            from the given superset of the region sepecified within the GeoDataFrame.
        superset: GeoDataFrame
            GeoDataFrame containing boundary definitions of multiple regions defined within
            a larger boundary. For example, state boundaries within the United States.

        Returns
        -------
        GeoDataFrame
            A dataframe of the boundary definitions specific to the list of regions provided
            as input.
        """

        try:
            superset['name_clean'] = superset['name'].str.strip().str.lower()
            regions = [region.strip().lower() for region in regions]

            selected = superset[superset['name_clean'].isin(regions)]
            return selected
        except Exception as e:
            logger.exception(
                f'Could not select a subset {regions=} within the dataframe specified.')
            raise e

    def load_by_name(self, dissolve_regions=False):
        """ Allows the user to automatically create the appropriate GeoDataFrame based
        on a natural language list of regions provided as input to this class. The primary
        driver of how GDFs are created. Utilitizes OSMNX library.

        Returns
        -------
        GeoDataFrame
            A dataframe with the list of boundary definitions associated with the list of
            regions specified as part of the class definition.

        """
        try:
            import osmnx as ox

            logger.info(
                'Calling geocode_to_gdf of on each of the regions specified')

            gdfs = []
            for region in self.regions:
                gdf = ox.geocode_to_gdf(region)
                gdfs.append(gdf)

            logger.info('Combining all the dataframes into one')

            self.gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

            if (dissolve_regions):
                logger.info("Dissolving geometries to unify overlapping regions.")
                self.gdf = gpd.GeoDataFrame(geometry=[self.gdf.unary_union])

            logger.info(f"Regions loaded: {list(self.gdf['name'])}")

            return self.gdf

        except ImportError as e:
            logger.exception("Need to install OSMNX to use this functionality {e}")
            raise e
        except Exception as e:
            logger.exception(
                f"Region names {self.regions} could not be resolved: {e}")
            raise e

    def as_array(self, polygon: shapely.geometry.Polygon, scale: float | int = 1e3) -> np.ndarray:
        """ Return the sparse coordinate representation for a polygon's exterior.

        Parameters
        ----------
        polygon : shapely.Polygon
            The Polygon object to return coordinates for.
        scale : float
            Controls the dimension size of the array required for the dense
            representation of the returned indices, i.e. [scale, scale]. Note
            that this parameter also controls the order of magnitude for the error
            due to type conversion (proportional to 1/scale), which is a result
            of converting the continuous domain of polygon coordinates into the
            discrete domain of image indices.

        Returns
        -------
        numpy.ndarray
            Array shaped [N, 2] that contains the indices of the polygon exterior
            in a 2D image; e.g. `image[tuple(as_array(polygon).T)] = 1` would have
            1 set for all locations that represent the polygon exterior. The max
            index in the returned array will be equal to the requested scale-1,
            and so would require an image shaped [scale, scale].

        """
        logger.info('Getting array of the polygons exterior coordinates')
        coords = np.array(polygon.exterior.coords)

        logger.info('Shifting the coordinates to be anchored at 0')
        offset = coords.min(0)
        zeroed = coords - offset

        logger.info(
            'Adjusting coordinates to the requested scale and converting to int32')
        scaler = zeroed.dtype.type(scale - 1) / zeroed.max(0)
        scaled = (zeroed * scaler).astype('int32')
        errors = abs(coords - (scaled.astype(coords.dtype)/scaler + offset))
        logger.info(
            f'Type conversion errors: Avg={errors.mean(0)} Max={errors.max(0)}')

        return scaled

    def gen_mask(self, data: xr.DataArray, resolution=1e3):
        """ Creates a mask based on all the regions specified within this class.

        Parameters
        ----------
        data: xr.DataArray
            The data over which the mask is generated. A mesh grid is generated
            with the latitude and longitude attributes of the data and the mask
            is applied over this meshgrid.
        resolution:
            This is the resolution of the mask. 1e3 is the default.

        Returns
        -------
        mask: numpy.ndarray
            The mask is a multi-dimensional list of booleans.
        """

        logger.info('Generating mask for the data provided.')

        if (self.gdf.empty):
            raise ValueError(f"Region '{self.regions} not found in shapefile")

        try:
            length = int(resolution)
            region_masks = []
            all_bounds = []

            logger.info(
                'Iterating through the rows within the gdf defined for this class.')

            for _, row in self.gdf.iterrows():
                geom = row.geometry
                if isinstance(geom, shapely.geometry.MultiPolygon):
                    geom = max(geom.geoms, key=lambda g: g.area)

                logger.debug(
                    'Getting geometric bounds for the region and creating a raster.')
                min_x, min_y, max_x, max_y = geom.bounds
                all_bounds.append((min_x, min_y, max_x, max_y))

                coords = self.as_array(geom, scale=length)
                filled = np.zeros((length, length), dtype='uint8')
                cv2.fillPoly(filled, [coords], 255)

                x = np.linspace(min_x, max_x, length)
                y = np.linspace(min_y, max_y, length)
                x, y = np.meshgrid(x, y)

                mask_coords = list(zip(x.ravel(), y.ravel()))

                logger.debug('Interpolating on the actual data.')
                interp = LinearNDInterpolator(
                    mask_coords, filled.ravel().astype('float'))
                X, Y = np.meshgrid(data.longitude, data.latitude)

                region_mask = interp(X, Y) > 127
                region_masks.append(region_mask)

            logger.info('Generating a final mask for all regions combined.')
            final_mask = np.any(region_masks, axis=0)

            min_xs, min_ys, max_xs, max_ys = zip(*all_bounds)
            full_bounds = (min(min_xs), min(min_ys), max(max_xs), max(max_ys))

            logger.info(
                'Creating a transform object with the boundaries of the combined mask.')
            self.transform = from_bounds(
                *full_bounds, data.longitude.size, data.latitude.size)

            logger.info(
                'Making the final mask explicitly a boolean representation.')
            self.mask = final_mask.astype(bool)
            return self.mask
        except Exception as e:
            logger.exception(f'Failed to generate mask: {e}')
            raise e

    def get_extent(self):
        """Generate the extent of the mask in terms of the latitude and longitude
        coordinates.

        Returns
        -------
        dict
            A dictionary with the keys 'latitude' and 'longitude', each containing a
            list of the minimum and maximum values for the respective coordinate.
            For example::

                {
                    'latitude': [min_lat, max_lat],
                    'longitude': [min_lon, max_lon]
                }
        """
        if self.mask is None:
            raise ValueError('Mask needs to be initialized to get extent of the mask.')

        try:
            masked_indices = np.argwhere(self.mask)
            min_row, min_col = masked_indices.min(axis=0)
            max_row, max_col = masked_indices.max(axis=0)

            # self.transform maps (col, row) → (lon, lat)
            min_lon, max_lat = self.transform * (min_col, min_row)
            max_lon, min_lat = self.transform * (max_col, max_row)

            extent = {
                'latitude':  [min_lat, max_lat],
                'longitude': [min_lon, max_lon]
            }

            logger.info(f'Extent of the mask: {extent}')
            return extent
        except Exception as e:
            logger.exception(f'Failed to get extent of the mask: {e}')
            raise e

    def gen_mask_boundary(self, width: int, height: int):
        """ Generates a fixed resolution binary mask based on the
        width and height specified.

        Parameters
        ----------
        width, height: int
            These parameters refer to the diemsions (in pixels) of the raster
            being asked to generate by the user.

        Returns
        -------
        self.mask: numpy.ndarray
            The mask is a multi-dimensional list of booleans.
        """

        logger.info(f"Generating Mask: {width}, {height}")

        try:

            logger.info('Get bounds of the region geometry.')
            bounds = self.gdf.total_bounds

            logger.info('Creating an affine transform for the raster grid.')
            self.transform = rasterio.transform.from_bounds(
                *bounds, width, height)

            logger.info('Rasterize the vector data.')
            self.mask = rasterize(
                [(geom, 1) for geom in self.gdf.geometry],
                out_shape=(height, width),
                transform=self.transform,
                fill=0,
                dtype=np.uint8
            )

            return self.mask
        except Exception as e:
            logger.exception(f'Failed to generate mask based on boundary: {e}')
            raise e

    def __repr__(self):
        return f"<Masked region='{self.regions}'>"

    def visualize_mask(self):
        """ Visualizes the ndarray self.mask to help validate regional mask. """
        logger.info("Visualizing mask")

        if (self.mask is None):
            raise ValueError(
                'Mask needs to be initialized to visualize mask.')

        fig, ax = plt.subplots(figsize=(10, 10))

        plt.imshow(self.mask, cmap='gray', alpha=0.5)

        self.gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=1)

        plt.title("Geospatial Mask")
        plt.axis('off')
        plt.show()

    def save_mask(self, output_path):
        """ Saves the mask for later reuse. """
        logger.info(f"Saving mask with GTiff driver: {output_path}")

        if (self.mask is None):
            raise ValueError(
                'Mask needs to be initialized to visualize mask.')

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=self.mask.shape[0],
            width=self.mask.shape[1],
            count=1,
            dtype=np.uint8,
            crs=self.gdf.crs,
            transform=self.transform
        ) as dst:
            dst.write(self.mask, 1)

    def load_mask(self, input_path):
        """ Load a previously saved mask from a GeoTIFF file. """
        logger.info(f"Loading mask from: {input_path}")

        try:
            with rasterio.open(input_path) as src:
                self.mask = src.read(1).astype(bool)
                self.transform = src.transform

                logger.info("Mask and transform loaded successfully.")
            return True
        except Exception as e:
            logger.exception(f"Failed to load mask: {e}")
            raise e
