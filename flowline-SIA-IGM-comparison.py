from oggm import cfg
from oggm import tasks, utils, workflow, graphics
from oggm.core import massbalance
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import igm

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

## Initialize OGGM and set up the default run parameters
cfg.initialize(logging_level='WARNING')

## Local working directory (where OGGM will write its output)
# WORKING_DIR = utils.gettempdir('OGGM_distr4')
cfg.PATHS['working_dir'] = utils.get_temp_dir('OGGM_distributed', reset=True)

## Pick a glacier
# rgi_ids = ['RGI60-11.01450']  # This is Aletsch
# rgi_ids = ['RGI60-11.00897']  # This is Hintereisferner
rgi_ids = ['RGI60-11.03638']  # This is Argentiere
base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v2'

gdir = workflow.init_glacier_directories(rgi_ids, prepro_base_url=base_url, from_prepro_level=3, prepro_border=40)[0]
gdir

# ### Recalibrate OGGM to match fixed geometry MB from hugonnet
# The default in OGGM nowadays is to match Hugonnet within a dynamical spinup. Lets do that here:
massbalance.mb_calibration_from_geodetic_mb(gdir, overwrite_gdir=True)

# ### Recompute the volume to match Farinotti et al. 2019 (for consistency with IGM later) 
from oggm.global_tasks import calibrate_inversion_from_consensus
calibrate_inversion_from_consensus([gdir])
# get ready for modelling
tasks.init_present_time_glacier(gdir)

# ### Experiment: a random warming simulation
# I recommend to stick to simple experiments for now. Here is a random run based on the climate of the past 21 years

# Do a random run with a bit of warming
tasks.run_random_climate(gdir, nyears=100, 
                         y0=2009, halfsize=10, # Climate of 1999-2019
                         seed=1,  # Change for another randomness 
                         temperature_bias=0,  # casual warming - change for other scenarios
                         store_fl_diagnostics=True,  # important! This will be needed for the redistribution
                         output_filesuffix='_rdn_1',  # name the run
                        );


# ## Redistribute: preprocessing
# The required tasks can be found in the `distribute_2d` module of the sandbox:
from oggm.sandbox import distribute_2d

# This is to add a new topography to the file (smoothed differently)
distribute_2d.add_smoothed_glacier_topo(gdir)
# This is to get the bed map at the start of the simulation
tasks.distribute_thickness_per_altitude(gdir)
# This is to prepare the glacier directory for the interpolation (needs to be done only once)
distribute_2d.assign_points_to_band(gdir)

# Let's have a look at what we just did:
with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
    ds = ds.load()

# Inititial glacier thickness
f, ax = plt.subplots()
ds.distributed_thickness.plot(ax=ax);
ax.axis('equal');

# Which points belongs to which band, and then within one band which are the first to melt
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ds.band_index.plot(ax=ax1);
ds.rank_per_band.plot(ax=ax2);
ax1.axis('equal'); ax2.axis('equal'); plt.tight_layout();

# export band indices
import rioxarray
ds.band_index.rio.to_raster("/mnt/c/Users/kneibm/Documents/CAIRN/Accu_Argentiere/argentiere_pleiades_smb/output/flowline/flowline_oggm.tif")

# ## Redistribute simulation
# The tasks above need to be run only once. The next one however should be done for each simulation:
distribute_2d.distribute_thickness_from_simulation(gdir, input_filesuffix='_rdn_1')

# ## Plot 
with xr.open_dataset(gdir.get_filepath('gridded_simulation', filesuffix='_rdn_1')) as ds:
    ds = ds.load()
ds

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
ds.distributed_thickness.sel(time=0).plot(ax=ax1, vmax=400);
ds.distributed_thickness.sel(time=40).plot(ax=ax2, vmax=400);
ds.distributed_thickness.sel(time=80).plot(ax=ax3, vmax=400);
ax1.axis('equal'); ax2.axis('equal'); plt.tight_layout();

# ## Animation!
from matplotlib import animation
from IPython.display import HTML, display

# Get a handle on the figure and the axes
fig, ax = plt.subplots()
thk = ds['distributed_thickness']

# Plot the initial frame. 
cax = thk.isel(time=0).plot(ax=ax,
    add_colorbar=True,
    cmap='viridis',
    vmin=0, vmax=350,
    cbar_kwargs={
        'extend':'neither'
    }
)
ax.axis('equal')

def animate(frame):
    ax.set_title(f'Year {int(frame)}')
    cax.set_array(thk.values[frame, :].flatten())

ani_glacier = animation.FuncAnimation(fig, animate, frames=len(thk.time), interval=200);

HTML(ani_glacier.to_jshtml())

# Write to mp4?
# FFwriter = animation.FFMpegWriter(fps=10)
# ani2.save('animation.mp4', writer=FFwriter)

# ## Understand the calibration of the MB model 
# The MB model is trained to reproduce the MB seen by Hugonnet et al on a fixed glacier geometry. Lets check it:
mb_model = massbalance.MonthlyTIModel(gdir)

# This is just a utility
heights, widths = gdir.get_inversion_flowline_hw()

years = np.arange(2000, 2020)
mbts = pd.DataFrame(index=years)
for y in years:
    mb_on_h = mb_model.get_annual_mb(heights, year=y)
    mbts.loc[y, 'mb_1d'] = np.average(mb_on_h, weights=widths) * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']

mbts.mean() 

# Get the reference data
ref_mb_df = massbalance.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
ref_mb_df = ref_mb_df.loc[ref_mb_df['period'] == '2000-01-01_2020-01-01']
# dmdtda: in meters water-equivalent per year -> we convert to kg m-2 yr-1
ref_mb = ref_mb_df['dmdtda'].iloc[0] * 1000
ref_mb_err = ref_mb_df['err_dmdtda'].iloc[0] * 1000

ref_mb, ref_mb_err


############################### OGGM-SIA
from oggm.core.sia2d import Upstream2D

# Load the 'gridded_data' dataframe which holds data like the 'topo_smoothed'
with xr.open_dataset(gdir.get_filepath('gridded_data')) as gd:
    gd = gd.load()

# Check the calibration of mass balance - likely needs to be recalibrated?
elev_on_2d = gd.topo_smoothed.data[gd.glacier_mask.data == 1]

for y in years:
    mb_on_h = mb_model.get_annual_mb(elev_on_2d, year=y)
    mbts.loc[y, 'mb_2d'] = np.average(mb_on_h) * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']

mbts.mean()

# It's not far (that's good), so we try without recalibration:
bed = gd.topo_smoothed - gd.distributed_thickness.fillna(0)
bed.plot();

mb = massbalance.RandomMassBalance(gdir,
                                   y0=2009, halfsize=10, # Climate of 1999-2019
                                   seed=1,  # Change for another randomness 
                                   )
mb.temp_bias += 0  # casual warming

sdmodel = Upstream2D(bed.values, init_ice_thick=gd.distributed_thickness.fillna(0).values, 
                     dx=gdir.grid.dx, mb_model=mb, y0=0., mb_filter=gd.glacier_mask.values==1)

dsr = sdmodel.run_until_and_store(100, grid=gdir.grid)
dsr

(ds.distributed_thickness.sum(dim=['x', 'y'])*gdir.grid.dx**2*1e-9).plot(label='OGGM-Flowline');
(dsr.ice_thickness.sum(dim=['x', 'y'])*gdir.grid.dx**2*1e-9).plot(label='OGGM-SIA');
plt.legend();

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
dsr.where(dsr['ice_thickness'] > 1).ice_thickness.sel(time=0).plot(ax=ax1, vmax=400);
dsr.where(dsr['ice_thickness'] > 1).ice_thickness.sel(time=40).plot(ax=ax2, vmax=400);
dsr.where(dsr['ice_thickness'] > 1).ice_thickness.sel(time=80).plot(ax=ax3, vmax=400);
ax1.axis('equal'); ax2.axis('equal'); plt.tight_layout();

### Animation! 
fig, ax = plt.subplots()
thk = dsr['ice_thickness'].where(dsr['ice_thickness'] > 0) # Maybe larger than 1 do avoid random years regrowth?

# Plot the initial frame. 
cax = thk.isel(time=0).plot(ax=ax,
    add_colorbar=True,
    cmap='viridis',
    vmin=0, vmax=350,
    cbar_kwargs={
        'extend':'neither'
    }
)
ax.axis('equal')

def animate(frame):
    ax.set_title(f'Year {int(frame)}')
    cax.set_array(thk.values[frame, :].flatten())

ani_glacier = animation.FuncAnimation(fig, animate, frames=len(thk.time), interval=200);
HTML(ani_glacier.to_jshtml())

############################### OGGM-IGM
from igm.instructed_oggm import IGM_Model2D

sdmodel_igm = IGM_Model2D(bed.values, init_ice_thick=gd.distributed_thickness.fillna(0).values, 
                     dx=gdir.grid.dx, dy=gdir.grid.dy, x=bed.x, y=bed.y,
                     mb_model=mb, y0=0., mb_filter=gd.glacier_mask.values==1)

dsr_igm = sdmodel_igm.run_until_and_store(100, grid=gdir.grid)
dsr_igm

(ds.distributed_thickness.sum(dim=['x', 'y'])*gdir.grid.dx**2*1e-9).plot(label='OGGM-Flowline');
(dsr.ice_thickness.sum(dim=['x', 'y'])*gdir.grid.dx**2*1e-9).plot(label='OGGM-SIA');
(dsr_igm.ice_thickness.sum(dim=['x', 'y'])*gdir.grid.dx**2*1e-9).plot(label='OGGM-IGM');
plt.legend();

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
dsr_igm.where(dsr_igm['ice_thickness'] > 1).ice_thickness.sel(time=0).plot(ax=ax1, vmax=400);
dsr_igm.where(dsr_igm['ice_thickness'] > 1).ice_thickness.sel(time=40).plot(ax=ax2, vmax=400);
dsr_igm.where(dsr_igm['ice_thickness'] > 1).ice_thickness.sel(time=80).plot(ax=ax3, vmax=400);
ax1.axis('equal'); ax2.axis('equal'); plt.tight_layout();


### Animation! 
fig, ax = plt.subplots()
thk_igm = dsr_igm['ice_thickness'].where(dsr_igm['ice_thickness'] > 1) # Maybe larger than 1 do avoid random years regrowth?

# Plot the initial frame. 
cax = thk_igm.isel(time=0).plot(ax=ax,
    add_colorbar=True,
    cmap='viridis',
    vmin=0, vmax=350,
    cbar_kwargs={
        'extend':'neither'
    }
)
ax.axis('equal')

def animate(frame):
    ax.set_title(f'Year {int(frame)}')
    cax.set_array(thk_igm.values[frame, :].flatten())

ani_glacier = animation.FuncAnimation(fig, animate, frames=len(thk.time), interval=200);
HTML(ani_glacier.to_jshtml())



