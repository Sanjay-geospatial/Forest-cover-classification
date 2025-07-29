class Preproccesor:
  def __init__(self, bbox, start_date, end_date, cloud_cover):
    self.bbox = bbox
    self.start_date = start_date
    self.end_date = end_date
    self.cloud_cover = cloud_cover

    import data_accessor
    self.data = data_accessor.Getdata(bbox = self.bbox,
                                start_date = self.start_date,
                                end_date = self.end_date,
                                cloud_cover = self.cloud_cover)

  def preprocess_satellite_data(self):

    from shapely.geometry import box, shape
    import geopandas as gpd
    import xarray as xr
    from skimage.exposure import rescale_intensity

    l8, l8_bounds = self.data.get_landsat()
    alos, alos_bounds = self.data.get_alos()

    l8_gdf = gpd.GeoDataFrame({'data':['Landsat']}, geometry=[shape(l8_bounds)], crs = 'EPSG:4326')
    alos_gdf = gpd.GeoDataFrame({'data':['ALOS']}, geometry=[shape(alos_bounds)], crs = 'EPSG:4326')

    l8_gdf.to_crs(epsg=32643, inplace=True)
    alos_gdf.to_crs(epsg=32643, inplace=True)

    common_area = l8_gdf.intersection(alos_gdf)

    print('Clipping the data...')
    l8_subset = l8.rio.clip(common_area, crs = common_area.crs, drop = True).squeeze()
    alos_subset = alos.rio.clip(common_area, crs = common_area.crs, drop = True).squeeze()

    core_coords = {'x', 'y', 'band'}

    l8_subset = l8_subset.drop_vars([c for c in l8_subset.coords if c not in core_coords])
    alos_subset = alos_subset.drop_vars([c for c in alos_subset.coords if c not in core_coords])

    print('Rescaling the data...')
    nir = rescale_intensity(l8_subset.sel(band = 'nir08'))
    red = rescale_intensity(l8_subset.sel(band = 'red'))
    blue = rescale_intensity(l8_subset.sel(band = 'blue'))
    green = rescale_intensity(l8_subset.sel(band = 'green'))
    swir1 = rescale_intensity(l8_subset.sel(band = 'swir16'))
    swir2 = rescale_intensity(l8_subset.sel(band = 'swir22'))
    lwir = rescale_intensity(l8_subset.sel(band = 'lwir11'))

    HH = rescale_intensity(alos_subset.sel(band='HH'))
    HV = rescale_intensity(alos_subset.sel(band='HV'))

    print('Calculating NDMI....')
    NDMI = (nir - swir1)/(nir + swir1)
    NDMI = NDMI.expand_dims(dim = {'band' : ['ndmi']})

    print('Calculating NDVI....')
    NDVI = (nir - red)/(nir + red)
    NDVI = NDVI.expand_dims(dim = {'band' : ['ndvi']})

    print('Calculating NBR....')
    NBR = (nir - swir2)/(nir + swir2)
    NBR = NBR.expand_dims(dim = {'band' : ['nbr']})

    print('Calculating NBR2....')
    NBR2 = (swir1 - swir2)/(swir1 + swir2)
    NBR2 = NBR2.expand_dims(dim = {'band' : ['nbr2']})

    print('Calculating NDWI....')
    NDWI = (green - nir)/(green + nir)
    NDWI = NDWI.expand_dims(dim = {'band' : ['ndwi']})

    print('Calculating MNDWI....')
    MNDWI = (green - swir1)/(green + swir1)
    MNDWI = MNDWI.expand_dims(dim = {'band' : ['mndwi']})

    print('Calculating MNDWI2....')
    MNDWI2 = (green - nir)/(green + nir)
    MNDWI2 = MNDWI2.expand_dims(dim = {'band' : ['mndwi2']})

    print('Calculating RVI....')
    RVI = (HH - HV) / (HH + HV)
    RVI = RVI.expand_dims(dim = {'band' : ['rvi']})

    print('Combining the data...')
    combined_data = xr.concat([red,green,blue, swir1, swir2, nir, lwir, HH, HV, NDMI,
                              NDVI, NBR, NBR2, NDWI, MNDWI, MNDWI2, RVI], dim = 'band')

    return combined_data, common_area


  def preprocess_gedi_data(self, bounds):
    self.bounds = bounds
    
    gedi_data = self.data.get_gedi(self.bounds)

    gedi_data_filtered = gedi_data[(gedi_data['quality_flag']==1) & (gedi_data['sensitivity']> 0.95)].copy()

    gedi_data_filtered['canopy_height'] = gedi_data_filtered['canopy_height']/100.0

    return gedi_data_filtered
  
  def get_gedi_excel(self, bounds):
    
    self.bounds = bounds
    
    self.data.get_gedi(self.bounds)







