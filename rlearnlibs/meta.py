import os
import numpy as np
from subprocess import PIPE
from grass.pygrass.modules.shortcuts import raster as r

class Meta(object):

    def covar(self, correlation=False):
        """
        Outputs a covariance or correlation matrix for the layers within the RasterStack object

        Parameters
        ----------
        correlation : logical, default is False.
            Whether to produce a correlation matrix or a covariance matrix.

        Returns
        -------
        numpy.ndarray
            Covariance/correlation matrix of the layers within the RasterStack with diagonal and
            upper triangle positions set to nan.
        """

        if correlation is True:
            flags = 'r'
        else:
            flags = ''

        corr = r.covar(map=self.names, flags=flags, stdout_=PIPE)
        corr = corr.outputs.stdout.split(os.linesep)[1:-1]
        corr = [i.strip() for i in corr]
        corr = [i.split(' ') for i in corr]
        corr = np.asarray(corr, dtype=np.float32)

        np.fill_diagonal(corr, np.nan)
        corr[np.triu_indices(corr.shape[0], 0)] = np.nan

        return corr
