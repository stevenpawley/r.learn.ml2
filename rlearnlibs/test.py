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
reg = Region()

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