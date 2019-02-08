from grass.pygrass.utils import set_path
set_path('r.learn.ml')

from raster import RasterStack

stack = RasterStack(rasters=["lsat5_1987_10", "lsat5_1987_20", "lsat5_1987_30", "lsat5_1987_40",
                             "lsat5_1987_50", "lsat5_1987_70"])
stack = RasterStack(rasters=maplist)
stack.lsat5_1987_10

maplist2 = deepcopy(maplist)
maplist2 = [i.split('@')[0] for i in maplist2]

stack = RasterStack(rasters=maplist2)
stack.lsat5_1987_10


X, y, crd = stack.extract_points(vect_name='landclass96_roi', fields=['value', 'cat'])
df = stack.extract_points(vect_name='landclass96_roi', field='value', as_df=True)
df = stack.extract_pixels(response='landclass96_roi', as_df=True)

X, y, crd = stack.extract_pixels(response='landclass96_roi')

stack.head()
stack.tail()

data = stack.read()
data.shape

df = stack.to_pandas()
# df = stack.to_pandas(res=500)
df = df.melt(id_vars=['x', 'y'])

from plotnine import *

(ggplot(df, aes(x="x", y="y", fill="value")) +
 geom_tile() + 
 coord_fixed() + 
 facet_wrap("variable") +
 theme_light() +
 theme(axis_title = element_blank()))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

stack.predict(clf, output='test', overwrite=True, height=25)
stack.predict_proba(clf, output='test', overwrite=True, height=25)


test = RasterRow('test')
from grass.pygrass.modules.shortcuts import raster as r
r.colors('test', color='random')
test
test.close()

from sklearn.model_selection import cross_validate
cross_validate(clf, X, y, cv=3)



    
from grass.pygrass.gis.region import Region
from grass.pygrass.modules.grid.grid import GridModule
from grass.pygrass.modules.grid import split
from grass.pygrass.modules.shortcuts import general as g
from grass.pygrass.raster import RasterRow
import multiprocessing as mltp
from itertools import chain
import time
import numpy as np

reg = Region()

# profile reading region-based blocks
# testreg = GridModule('g.region', width=100, height=100, processes=4)
testreg = split.split_region_tiles(width=reg.cols, height=100)

def worker(src):
    window, src = src
    window = dict(window)
    window['n'] = window.pop('north')
    window['s'] = window.pop('south')
    window['e'] = window.pop('east')
    window['w'] = window.pop('west')
    del(window['top'])
    del(window['bottom'])
    
    g.region(**window)
    
    with RasterRow(src) as rs:
        arr = np.asarray(rs)
    return(arr)

windows = list(chain.from_iterable(testreg))
windows = [[i.items(), "lsat5_1987_10"] for i in windows]

start = time.time()
pool = mltp.Pool(processes=8)
arrs = pool.map(func=worker, iterable=windows)
end = time.time()
print(end - start)


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