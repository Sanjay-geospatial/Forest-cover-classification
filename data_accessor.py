class Getdata:
    def __init__(self, bbox,
                 start_date: str,
                 end_date: str,
                 cloud_cover: int = 10):
      import subprocess
      import sys

      try :
        import earthaccess
      except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "earthaccess"])
        import earthaccess

      try:
        import pystac_client
      except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pystac_client"])
        import pystac_client

      try:
        import rioxarray as rxr
      except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rioxarray"])
        import rioxarray as rxr

      try:
        import stackstac
      except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "stackstac"])
        import stackstac

      try:
        import planetary_computer
      except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "planetary_computer"])
        import planetary_computer

      import geopandas as gpd
      import numpy as np
      import os
      import random
      from typing import Tuple, List
      import pandas as pd
      import h5py
      from tqdm.auto import tqdm
      import time

      """
      Query and download Sentinel-1/2 and GEDI data using STAC API and NASA Earthdata.
      """

      STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
      # os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

      self.earthaccess = earthaccess
      self.rxr = rxr
      self.pystac_client = pystac_client
      self.stackstac = stackstac
      self.gpd = gpd
      self.np = np
      self.os = os
      self.random = random
      self.pd = pd
      self.h5py = h5py
      self.tqdm = tqdm
      self.catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
      self.bbox = bbox
      self.start_date = start_date
      self.end_date = end_date
      self.year = start_date.split("-")[0]
      self.cloud_cover = cloud_cover

    def get_landsat(self):
        search = self.catalog.search(
            collections=["landsat-c2-l2"],
            bbox=self.bbox,
            datetime=f"{self.start_date}/{self.end_date}",
            query={"eo:cloud_cover": {"lt": self.cloud_cover}},
            max_items=20
        )

        items = list(i for i in list(search.get_all_items()) if i.id[0:4] in ['LC08','LC09'])

        if not items:
            print("No Landsat scenes found.")
            return None

        items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
        best_item = items_sorted[0]
        print(f"Best item: {best_item.id} | Cloud: {best_item.properties['eo:cloud_cover']}%")

        bands = ["blue", "green", "nir08", "red", "swir16", "swir22", "lwir11"]

        ds_s2 = self.stackstac.stack(
            items = [best_item],
            assets = bands,
            epsg = 32643,
            resolution = 30
        ).squeeze()

        return ds_s2, best_item.geometry

    def get_alos(self):

      search = self.catalog.search(
      collections = ['alos-palsar-mosaic'],
      datetime = f'{self.year}/{self.year}',
      bbox = self.bbox
      )

      items = list(search.get_all_items())

      if not items:
        print("No ALOS PALSAR scenes found.")
        return None

      alos = self.stackstac.stack(
          items = items,
          assets = ['HH', 'HV'],
          epsg = 32643,
          resolution=30
      ).squeeze()

      return alos, items[0].geometry

    def get_gedi(self, bounds):
        self.bounds = bounds
        self.earthaccess.login()

        collections = self.earthaccess.collection_query().keyword('GEDI').version('002').provider('LPCLOUD').get()
        collectionIDs = {c.summary()['short-name']: c.summary()['concept-id'] for c in collections}

        temp = (f"{self.year}-01-01", f"{self.year}-12-31")
        concept_id = collectionIDs['GEDI02_B']

        results = self.earthaccess.search_data(
            concept_id=concept_id,
            temporal=temp,
            bounding_box=self.bounds
        )

        files = self.earthaccess.open(results)

        beams = ['BEAM0000', 'BEAM0001','BEAM0010', 'BEAM0011',
                 'BEAM0101', 'BEAM0110', 'BEAM1000','BEAM1011']

        shotNum, Lat, Lon, canopyHeight, quality, sensitivity, cover, beamI = ([] for _ in range(8))

        for i, file in self.tqdm(enumerate(files), total=len(files), desc="Processing GEDI files"):
            with self.h5py.File(file, 'r') as gedi_ds:
                for beam in beams:
                    try:
                        shotNum.append(gedi_ds[f'{beam}/shot_number'][:])
                        Lat.append(gedi_ds[f'{beam}/geolocation/lat_lowestmode'][:])
                        Lon.append(gedi_ds[f'{beam}/geolocation/lon_lowestmode'][:])
                        canopyHeight.append(gedi_ds[f'{beam}/rh100'][:])
                        quality.append(gedi_ds[f'{beam}/l2b_quality_flag'][:])
                        sensitivity.append(gedi_ds[f'{beam}/sensitivity'][:])
                        cover.append(gedi_ds[f'{beam}/cover'][:])
                        beamI.append([beam] * len(gedi_ds[f'{beam}/shot_number'][:]))
                    except KeyError:
                        continue
            print(f"Processed file {i+1}/{len(files)}")

        shotNum = self.np.concatenate(shotNum)
        Lat = self.np.concatenate(Lat)
        Lon = self.np.concatenate(Lon)
        canopyHeight = self.np.concatenate(canopyHeight)
        quality = self.np.concatenate(quality)
        sensitivity = self.np.concatenate(sensitivity)
        cover = self.np.concatenate(cover)
        beamI = self.np.concatenate(beamI)

        gedi_df = self.pd.DataFrame({
            'shot_number': shotNum,
            'Latitude': Lat,
            'Longitude': Lon,
            'canopy_height': canopyHeight,
            'quality': quality,
            'sensitivity': sensitivity,
            'tree_cover': cover,
            'beam': beamI
        })

        min_lon, min_lat, max_lon, max_lat = self.bounds
        gedi_df_aoi = gedi_df[
            (gedi_df['Latitude'] >= min_lat) & (gedi_df['Latitude'] <= max_lat) &
            (gedi_df['Longitude'] >= min_lon) & (gedi_df['Longitude'] <= max_lon)
        ]

        gedi_gdf = self.gpd.GeoDataFrame(
            gedi_df_aoi,
            geometry=gpd.points_from_xy(gedi_df_aoi['Longitude'], gedi_df_aoi['Latitude']),
            crs='EPSG:4326'
        )

        return gedi_gdf
    
    def get_gedi_excel(self, bounds):
      self.bounds = bounds
      self.earthaccess.login()

      collections = self.earthaccess.collection_query().keyword('GEDI').version('002').provider('LPCLOUD').get()
      collectionIDs = {c.summary()['short-name']: c.summary()['concept-id'] for c in collections}

      temp = (f"{self.year}-01-01", f"{self.year}-12-31")
      concept_id = collectionIDs['GEDI02_B']

      results = self.earthaccess.search_data(
          concept_id=concept_id,
          temporal=temp,
          bounding_box= bounds
      )

      files = self.earthaccess.open(results)

      beams = ['BEAM0000', 'BEAM0001','BEAM0010', 'BEAM0011',
                  'BEAM0101', 'BEAM0110', 'BEAM1000','BEAM1011']

      for i, file in tqdm(enumerate(files), total=len(files), desc="Processing GEDI files"):
        shotNum, Lat, Lon, canopyHeight, quality, sensitivity, cover, beamI = ([] for _ in range(8))
        with self.h5py.File(file, 'r') as gedi_ds:
            for beam in beams:
                try:
                    shotNum.append(gedi_ds[f'{beam}/shot_number'][:])
                    Lat.append(gedi_ds[f'{beam}/geolocation/lat_lowestmode'][:])
                    Lon.append(gedi_ds[f'{beam}/geolocation/lon_lowestmode'][:])
                    canopyHeight.append(gedi_ds[f'{beam}/rh100'][:])
                    quality.append(gedi_ds[f'{beam}/l2b_quality_flag'][:])
                    sensitivity.append(gedi_ds[f'{beam}/sensitivity'][:])
                    cover.append(gedi_ds[f'{beam}/cover'][:])
                    beamI.append([beam] * len(gedi_ds[f'{beam}/shot_number'][:]))
                except KeyError:
                    continue
        print(f"Processing file {i+1}/{len(files)}")

        shotNum = self.np.concatenate(shotNum)
        Lat = self.np.concatenate(Lat)
        Lon = self.np.concatenate(Lon)
        canopyHeight = self.np.concatenate(canopyHeight)
        quality = self.np.concatenate(quality)
        sensitivity = self.np.concatenate(sensitivity)
        cover = self.np.concatenate(cover)
        beamI = self.np.concatenate(beamI)

        gedi_df = self.pd.DataFrame({
            'shot_number': shotNum,
            'Latitude': Lat,
            'Longitude': Lon,
            'canopy_height': canopyHeight,
            'quality': quality,
            'sensitivity': sensitivity,
            'tree_cover': cover,
            'beam': beamI
        })

        min_lon, min_lat, max_lon, max_lat = bounds
        gedi_df_aoi = gedi_df[
            (gedi_df['Latitude'] >= min_lat) & (gedi_df['Latitude'] <= max_lat) &
            (gedi_df['Longitude'] >= min_lon) & (gedi_df['Longitude'] <= max_lon) &
            (gedi_df['quality'] == 1)
        ]

        print(f"\nWriting csv file {i+1}/{len(files)}")

        file_name = f'gedi_file_{i}.csv'

        gedi_df_aoi.to_csv(file_name)
        time.sleep(10)

