# RasterStack
 
 A RasterStack enables a collection of RasterRow objects to be bundled into a multi-layer RasterStack object
        
  Parameters
  ----------
  rasters : list, str
      
      List of names of GRASS GIS raster maps. Note that although the
      maps can reside in different mapsets, they cannot have the same
      names.

  group : str (opt)
      
      Create a RasterStack from rasters contained in a GRASS GIS imagery
      group. This parameter is mutually exclusive with the `rasters`
      parameter.

  Attributes
  ----------
  loc : dict
      
      Label-based indexing of RasterRow objects within the RasterStack.

  iloc : int
      
      Index-based indexing of RasterRow objects within the RasterStack.

  mtypes : dict
      
      Dict of key, value pairs of full_names and GRASS data types.

  count : int
      
      Number of RasterRow objects within the RasterStack.
