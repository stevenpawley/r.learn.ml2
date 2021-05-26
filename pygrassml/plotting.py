import math
import tempfile
from subprocess import PIPE

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from grass.pygrass.modules.shortcuts import raster as gr
from grass.pygrass.raster import RasterRow
from grass.script.utils import parse_key_val
from mpl_toolkits.axes_grid1 import make_axes_locatable


def convert_grass_color(raster):
    """Converts a GRASS color rules file to a matplotlib.colors.ListedColorMap
    and a matplotlib.colors.Normalize object

    Parameters
    ----------
    raster : str
        Name of the GRASS GIS raster map

    Returns
    -------
    cmap : matplotlib.colors.ListedColorMap
        The GRASS rgb rules converted to a cmap

    norm : matplotlib.colors.Normalize
        The raster value breaks used in mapping the cmap to the colours.
    """
    # export color rules
    rules_file = tempfile.NamedTemporaryFile().name
    gr.colors_out(map=raster, rules=rules_file)

    # read color rules
    rules = pd.read_table(
        filepath_or_buffer=rules_file,
        sep=" ",
        header=None,
        names=["raster_value", "rgb"]
    )
    rules = rules.loc[~rules["raster_value"].str.contains("nv"), :]
    rules = rules.loc[~rules["raster_value"].str.contains("default"), :]
    rules["raster_value"] = rules["raster_value"].astype("float")
    rules = rules.sort_values("raster_value")
    rules = rules.drop_duplicates(subset="raster_value")
    colors = rules["rgb"].str.split(":", expand=True)
    colors = colors.astype("float32")
    colors = colors.apply(lambda x: x / x.max(), axis=1)
    colors = colors.fillna(0.0)
    colors.columns = ["r", "g", "b"]
    cvals = rules["raster_value"]

    # get mtype
    with RasterRow(raster) as src:
        mtype = src.mtype

    # define the colors
    if mtype == "CELL":
        cmap = mpl.colors.ListedColormap(colors.values)
        norm = mpl.colors.BoundaryNorm(boundaries=cvals.values, ncolors=cmap.N)

    else:
        # convert colors into (raster_val, [r, g, b]) pairs
        colors_list = list()

        for i, v in colors.iterrows():
            colors_list.append(([v.r, v.g, v.b]))

        norm = plt.Normalize(cvals.min(), cvals.max())
        tuples = list(zip(map(norm, cvals), colors_list))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)

    return cmap, norm


class PlottingMixin(object):
    def plot(self, reg=None, cmap=None, norm=None, figsize=None,
             title_fontsize=8, label_fontsize=6, legend_fontsize=6,
             share_legend=False, names=None, fig_kwds=None, legend_kwds=None,
             subplots_kwds=None):
        """Plot a Raster object as a raster matrix

        Parameters
        ----------
        reg : grass.pygrass.gis.region.Region
            The region used for the plotting extent.

        cmap : str (opt), default=None
            Specify a single cmap to apply to all of the RasterLayers.
            This overrides the color map that is assigned the GRASS GIS raster.

        norm :  matplotlib.colors.Normalize (opt), default=None
            A matplotlib.colors.Normalize to apply to all of the rasters.
            This overrides any color maps that are associated to each GRASS GIS
            raster.

        figsize : tuple (opt), default=None
            Size of the resulting matplotlib.figure.Figure.

        out_shape : tuple, default=(100, 100)
            Number of rows, cols to read from the raster datasets for plotting.

        title_fontsize : any number, default=8
            Size in pts of titles.

        label_fontsize : any number, default=6
            Size in pts of axis ticklabels.

        legend_fontsize : any number, default=6
            Size in pts of legend ticklabels.

        share_legend : bool, default=False
            Optionally share a single legend between the plots. This assumes
            that all of the GRASS GIS rasters are using the same color scale,
            and the color scale used for plotting is taken from the last raster
            in the RasterStack.

        names : list (opt), default=None
            Optionally supply a list of names for each RasterLayer to override
            the default layer names for the titles.

        fig_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.figure call
            when creating the figure object.

        legend_kwds : dict (opt), default=None
            Additional arguments to pass to the matplotlib.pyplot.colorbar call
            when creating the colorbar object.

        subplots_kwds : dict (opt), default=None
            Additional arguments to pass to the
            matplotlib.pyplot.subplots_adjust function. These are used to
            control the spacing and position of each subplot, and can include
            {left=None, bottom=None, right=None, top=None, wspace=None,
            hspace=None}.

        Returns
        -------
        axs : numpy.ndarray
            array of matplotlib.axes._subplots.AxesSubplot or a single
            matplotlib.axes._subplots.AxesSubplot if Raster object contains
            only a single layer.
        """
        # some checks
        if reg is None:
            raise AttributeError("argument `reg` requires a region object.")

        if norm:
            if not isinstance(norm, mpl.colors.Normalize):
                raise AttributeError(
                    "norm argument should be a \
                    matplotlib.colors.Normalize object")

        # override grass raster colors
        if cmap:
            cmaps = [cmap for i in range(self.count)]
            if norm:
                norms = [norm for i in range(self.count)]
            else:
                norms = [None for i in range(self.count)]
        else:
            scales = [convert_grass_color(name) for name in self.names]
            cmaps, norms = zip(*scales)

        # override map titles
        if names is None:
            names = []

            for src in self.iloc:
                nm = gr.info(
                    src.fullname(), flags="e", stdout_=PIPE).outputs.stdout
                title = parse_key_val(nm)["title"]
                title = title.replace('"', "")
                names.append(title)
        else:
            if len(names) != self.count:
                raise AttributeError("arguments 'names' needs to be the same "
                                     "length as the number of RasterLayer "
                                     "objects")

        if fig_kwds is None:
            fig_kwds = {}

        if legend_kwds is None:
            legend_kwds = {}

        if subplots_kwds is None:
            subplots_kwds = {}

        if figsize:
            fig_kwds["figsize"] = figsize

        # estimate required number of rows and columns in figure
        rows = int(np.sqrt(self.count))
        cols = int(math.ceil(np.sqrt(self.count)))

        if rows * cols < self.count:
            rows += 1

        fig, axs = plt.subplots(rows, cols, **fig_kwds)

        if isinstance(axs, np.ndarray):
            # axs.flat is an iterator over the row-order flattened axs array
            for ax, n, cmap, norm, name in zip(
                    axs.flat, range(self.count), cmaps, norms, names
            ):
                arr = self.read(index=n)
                arr = arr.squeeze()
                ax.set_title(name, fontsize=title_fontsize, y=1.00)
                extent = [reg.west, reg.east, reg.south, reg.north]
                im = ax.imshow(arr, extent=extent, cmap=cmap, norm=norm)

                if share_legend is False:
                    divider = make_axes_locatable(ax)

                    if "orientation" not in legend_kwds.keys():
                        legend_kwds["orientation"] = "vertical"

                    if legend_kwds["orientation"] == "vertical":
                        legend_pos = "right"

                    elif legend_kwds["orientation"] == "horizontal":
                        legend_pos = "bottom"

                    cax = divider.append_axes(legend_pos, size="10%", pad=0.1)
                    cbar = plt.colorbar(im, cax=cax, **legend_kwds)
                    cbar.ax.tick_params(labelsize=legend_fontsize)

                # hide tick labels by default when multiple rows or cols
                ax.axes.get_xaxis().set_ticklabels([])
                ax.axes.get_yaxis().set_ticklabels([])

                # show y-axis tick labels on first subplot
                if n == 0 and rows > 1:
                    ticks_loc = ax.get_yticks().tolist()
                    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_yticklabels(
                        ax.yaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize)

                if n == 0 and rows == 1:
                    ticks_loc = ax.get_xticks().tolist()
                    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_xticklabels(
                        ax.xaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize)

                    ticks_loc = ax.get_yticks().tolist()
                    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_yticklabels(
                        ax.yaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize)

                if rows > 1 and n == (rows * cols) - cols:
                    ticks_loc = ax.get_xticks().tolist()
                    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax.set_xticklabels(
                        ax.xaxis.get_majorticklocs().astype("int"),
                        fontsize=label_fontsize)

            for ax in axs.flat[axs.size - 1: self.count - 1: -1]:
                ax.set_visible(False)

            if share_legend is True:
                cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax, **legend_kwds)
                cbar.ax.tick_params(labelsize=legend_fontsize)

            plt.subplots_adjust(**subplots_kwds)

        else:
            arr = self.read(index=0)
            arr = arr.squeeze()
            cmap = cmaps[0]
            norm = norms[0]
            axs.set_title(names[0], fontsize=title_fontsize, y=1.00)
            extent = [reg.west, reg.east, reg.south, reg.north]
            im = axs.imshow(arr, extent=extent, cmap=cmap, norm=norm)

            divider = make_axes_locatable(axs)

            if "orientation" not in legend_kwds.keys():
                legend_kwds["orientation"] = "vertical"

            if legend_kwds["orientation"] == "vertical":
                legend_pos = "right"

            elif legend_kwds["orientation"] == "horizontal":
                legend_pos = "bottom"

            cax = divider.append_axes(legend_pos, size="10%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax, **legend_kwds)
            cbar.ax.tick_params(labelsize=legend_fontsize)

        return axs
