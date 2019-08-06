from grass.script.utils import get_lib_path
from grass.pygrass.raster import RasterRow
from copy import deepcopy
from grass.pygrass.gis.region import Region
from grass.pygrass.modules.grid import split
import multiprocessing as mltp
from itertools import chain
import time
import numpy as np
import sys
path = get_lib_path('r.learn.ml')
sys.path.append(path)
from raster import RasterStack
import pandas as pd
reg = Region()

stack = RasterStack(group='terrain')
df = stack.extract_points(vect_name='picks_mnt_selected', 
                          fields='pick_thk_m',
                          as_df=True)
y = df.pick_thk_m.values
X = df.drop(labels=['x', 'y', 'pick_thk_m', 'cat'], axis=1).values
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, n_estimators=100)
rf.fit(X, y)
stack.predict(rf, 'test', height=100, overwrite=True)


arr.shape
pd.DataFrame(stack.names)
X[:, 48]
arr[48, :, :]

arr = stack.read()
arr.shape
img = arr

n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

# reshape each image block matrix into a 2D matrix
# first reorder into rows,cols,bands(transpose)
# then resample into 2D array (rows=sample_n, cols=band_values)
n_samples = rows * cols
flat_pixels = img.transpose(1, 2, 0).reshape(
    (n_samples, n_features))

# create mask for NaN values and replace with number
flat_pixels_mask = flat_pixels.mask.copy()
flat_pixels = np.ma.filled(flat_pixels, -99999)

# prediction
result = rf.predict(flat_pixels)

# replace mask and fill masked values with nodata value
result = np.ma.masked_array(
    result, mask=flat_pixels_mask.any(axis=1))

# reshape the prediction from a 1D matrix/list
# back into the original format [band, row, col]
result = result.reshape((1, rows, cols))
result.shape

import matplotlib.pyplot as plt
plt.imshow(result[0, :, :])
plt.colorbar()
result.dtype
result = np.ma.filled(result, np.nan)

from grass.pygrass.raster import numpy2raster
numpy2raster(result[0, :, :], 'FCELL', 'test', overwrite=True)



vect_name = 'picks_mnt_selected@bedrock_topo'
fields = 'pick_thk_m'

if isinstance(fields, str):
    fields = [fields]

points = VectorTopo(vect_name.split('@')[0])
points.open('r')
points.num_primitives()['point']

df = pd.DataFrame(points.table_to_dict()).transpose()
df_cols = points.table.columns
df_cols = [name for (name, dtype) in df_cols.items()]
df = df.rename(columns={old:new for old, new in zip(df.columns, df_cols)})
df = df.loc[:, fields + [points.table.key]]

from grass.pygrass.modules.shortcuts import vector as v
from subprocess import PIPE

rast_data = v.what_rast(
    map=vect_name,
    raster='dem',
    flags='p', quiet=True, stdout_=PIPE).outputs.stdout

rast_data = rast_data.split(os.linesep)[:-1]

X = (np.asarray([k.split('|')[1]
    if k.split('|')[1] != '*' else np.nan for k in rast_data]))
cat = (np.asarray([k.split('|')[0]
    if k.split('|')[1] != '*' else np.nan for k in rast_data]))
cat = [int(i) for i in cat]
    
src = RasterRow('dem@bedrock_topo')
src.open()
src.mtype
src.close()

if src.mtype == 'CELL':
    X = [int(i) for i in X]
else:
    X = [float(i) for i in X]

X = pd.DataFrame(np.column_stack((X, cat)), columns=[name, points.table.key])
df = df.merge(X, on='cat')



stack = RasterStack(rasters=["lsat5_1987_10", "lsat5_1987_20", "lsat5_1987_30", "lsat5_1987_40",
                             "lsat5_1987_50", "lsat5_1987_70"])

X, y, crd = stack.extract_points(
        vect_name='landclass96_roi',
        fields='value')

df = stack.extract_points(
        vect_name='landclass96_roi', fields='value', 
        as_df=True)

df = stack.extract_pixels(response='landclass96_roi', as_df=True)

X, y, crd = stack.extract_pixels(response='landclass96_roi')

stack.head()
stack.tail()

df = stack.to_pandas()
df = df.melt(id_vars=['x', 'y'])

from plotnine import *

(ggplot(df, aes(x="x", y="y", fill="value")) +
 geom_tile() + 
 coord_fixed() + 
 facet_wrap("variable") +
 theme_light() +
 theme(axis_title = element_blank()))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf = GridSearchCV(clf, param_grid={'min_samples_leaf': [1, 2, 5]})
clf.fit(X, y)

stack.predict(clf, output='test_script', overwrite=True, height=25)
stack.predict_proba(clf, output='test_script', overwrite=True, height=25)

from sklearn.model_selection import cross_validate
cross_validate(clf, X, y, cv=3)

# profile reading region-based blocks
testreg = split.split_region_tiles(width=reg.cols, height=300)
windows = list(chain.from_iterable(testreg))
windows = [[i.items(), "lsat5_1987_10"] for i in windows]

def worker(cmd):

    window, src = cmd
    
    reg = Region()
    old_reg = deepcopy(reg)
    
    try:
        # update region
        reg.north = dict(window)['north']
        reg.south = dict(window)['south']
        reg.west = dict(window)['west']
        reg.east = dict(window)['east']
        reg.set_current()
        reg.write()
        reg.set_raster_region()
        
        # read raster data
        with RasterRow(src) as rs:
            arr = np.asarray(rs)
    except:
        pass

    finally:
        # reset region
        old_reg.write()
        reg.set_raster_region()
    
    return(arr)

start = time.time()
pool = mltp.Pool(processes=8)
arrs = pool.map_async(func=worker, iterable=windows)
arrs.wait()
end = time.time()
print(end - start)

np.row_stack(arrs.get()).shape

# profile reading single thread per row
start = time.time()
with RasterRow("lsat5_1987_10") as src:
    arr = []
    for i in range(reg.rows):
        arr.append(src[i])
end = time.time()
print(end - start)

# profile reading multiprocessing per row
def worker(src):
    row, src = src
    with RasterRow(src) as rs:
        arr = rs[row]
    return row

rows = [(i, "lsat5_1987_10") for i in range(reg.rows)]

start = time.time()
pool = mltp.Pool(processes=8)
arrs = pool.map(func=worker, iterable=rows)
end = time.time()
print(end - start)

src = RasterRow("lsat5_1987_10")
src.open()
src[0]
src.close()


from grass.pygrass.modules.grid.grid import GridModule

grd = GridModule('r.learn.predict', width=100, height=100,
                 group='land@landsat',
                 load_model='/home/steven/Downloads/model.gz',
                 output='test',
                 overwrite=True)
grd.run()