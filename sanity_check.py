from crest.data.loading.Datafile import Datafile
from crest.data.loading.RegionalMaskGenerator import RegionalMaskGenerator
import matplotlib.pyplot as plt
from pathlib import Path
from crest.data_server.DataServer import DataServer

df = Datafile('./soil.zarr', region=['New Hampshire', 'New York State'])

df.data.sel(features="Clay_frac").plot()

plt.show()

# regional_mask = RegionalMaskGenerator(['WA'])
# if (regional_mask.gen_mask(df.data)):
#     regional_mask.visualize_mask()