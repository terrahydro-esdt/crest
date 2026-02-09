import xarray as xr
import numpy as np
import json, os
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class DataMetrics():
    def open_dataset(self, path: str, decode_times: bool = True) -> xr.Dataset:
        return xr.open_zarr(path, decode_times=decode_times)

    def get(self, path: str, decode_times: bool = True):
        
        ds = None
        if path:
            try:
                log.info(f"DataMetrics.get opening dataset {path}")
                ds = self.open_dataset(path, decode_times=decode_times)
            except Exception as e:
                raise e

        if not ds:
            raise Exception("Data could not be loaded.")

        data_vars = list(ds.data_vars)
        if not data_vars:
            log.error("No data variables found in dataset.")
            raise Exception("No data variables found in dataset.")

        if 'datetime' in ds.coords and np.issubdtype(ds['datetime'].dtype, np.datetime64):
            time_coords = ds['datetime']
        else:
            log.error("Cannot find datetime coordinates")
            raise Exception("Cannot find datetime coordinates")

        return (ds, data_vars, time_coords)


    def autodetect_rescale(self, dataset: xr.Dataset, variable: str, time_coord=None):
        try:
            v = dataset[variable]

            # If variable has time dimension, sample first N time steps
            if time_coord is not None and time_coord.name in v.dims:
                v_sample = v.isel({time_coord.name: slice(
                    0, min(10, v.sizes[time_coord.name]))})
            else:
                v_sample = v

            # Compute robust quantiles to avoid outliers
            q = v_sample.quantile([0.02, 0.98], dim=[
                                d for d in v_sample.dims if d != time_coord.name] if time_coord is not None else None)
            vmin = float(q.isel(quantile=0).values)
            vmax = float(q.isel(quantile=1).values)
        except Exception:
            raise Exception("Cannot load dataset enough to calculate scale.")

        rescale_default = f"{vmin},{vmax}"
        return rescale_default
    
    @staticmethod
    def write_metadata(path: str, date: str, decode_times=True, metadata_path=None):

        try:
            if (not path) or (not os.path.exists(path)):
                raise Exception(f"Data path {path} does not exist.")
            
            path = Path(path)
            if (path.suffix == '.zarr'):
                parent_path = path.parent
                data_path = str(path)
            else:
                parent_path = path
                data_path = None
                zarr_dirs = list(path.rglob("*.zarr"))

                if (not zarr_dirs or len(zarr_dirs) == 0):
                    raise Exception(f"No .zarr directories found in {path}.")
                else:
                    zarr_dirs = sorted(zarr_dirs)
                    for zarr_dir in zarr_dirs:
                        # get first zarr file
                        if (zarr_dir.suffix == '.zarr'):
                            data_path = str(zarr_dir)
                            break

            if (metadata_path is None or len(metadata_path) == 0):
                metadata_path = os.path.join(parent_path, 'metadata.json')
            else:
                metadata_path = os.path.join(metadata_path, 'metadata.json')

            if (os.path.exists(metadata_path)):
                log.info(f"{metadata_path} already exists.")
                met = {}
                with open(metadata_path, 'r') as fp:
                    met = json.load(fp)

                if (len(met) > 0 and met['date'] == date):
                    log.info(f"Metadata for {data_path} on {date} already exists and is valid.")
                    return {data_path: metadata_path}

            datamet = DataMetrics()
            dataset, variables, times = datamet.get(data_path, decode_times)

            datetimes = np.array(times, dtype="datetime64[ns]").astype("datetime64[ns]").astype(str)
            datetimes = datetimes.tolist()

            metadata = {
                'date': date,
                'path': data_path,
                'variables': variables,
                'datetime': datetimes
            }

            for var in variables:
                rescale_default = datamet.autodetect_rescale(dataset, var)
                metadata[var] = rescale_default

            with open(metadata_path, 'w') as fp:
                json.dump(metadata, fp)

            return {data_path: metadata_path}
        except Exception as e:
            raise e

if __name__=="__main__":
    x = DataMetrics.write_metadata('/s3/thdro/data/HindCast/live/ERA5.zarr', "2025-12-11", True, "/efs/thdro/logs/ingest/HindCast/dataDay/")
    y = DataMetrics.write_metadata('/s3/thdro/data/ForeCast/live/IFS.zarr', "2025-12-18", True, "/efs/thdro/logs/ingest/ForeCast/")
    print(x, y)
