# IMPORTS

import os
import xarray as xr
import pandas as pd
import geopandas as gpd
import re
import json
from tqdm import tqdm
import cdsapi
import zipfile
import shutil
import requests
from funcs import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

# DICTIONARIES FOR DATA DOWNLOAD

# Copernicus Climate Data Store API calls

cds_calls_historical = {
    "temperature": {
        'variable': '2m_temperature',
        "filename": 'historical_temperature.nc'},
    "pressure": {
        'variable': 'surface_pressure',
        "filename": "historical_pressure.nc"},
    "wind_speed": {
        'variable': '10m_wind_speed',
        "filename": "historical_wind_speed.nc"},
    "precipitation": {
        'variable': 'total_precipitation',
        "filename": "historical_precipitation.nc"},
    "ch4": {
        "name": "satellite-methane", 
        "processing_level": "level_3",
        "variable": "xch4",
        "sensor_and_algorithm": "merged_obs4mips",
        "version": "4.4",
        "format": "zip",
        "filename": "historical_ch4.nc"
    }
}

cds_calls_ssp = {
    "ssp1_temperature": {
        'experiment': 'ssp1_2_6',
        'variable': 'near_surface_air_temperature',
        "filename": "ssp1_temperature.nc"},
    "ssp2_temperature": {
        'experiment': 'ssp2_4_5',
        'variable': 'near_surface_air_temperature',
        "filename": "ssp2_temperature.nc"},
    "ssp3_temperature": {
        'experiment': 'ssp3_7_0',
        'variable': 'near_surface_air_temperature',
        "filename": "ssp3_temperature.nc"},
    "ssp4_temperature": {
        'experiment': 'ssp4_6_0',
        'variable': 'near_surface_air_temperature',
        "filename": "ssp4_temperature.nc"},
    "ssp5_temperature": {
        'experiment': 'ssp5_8_5',
        'variable': 'near_surface_air_temperature',
        "filename": "ssp5_temperature.nc"},
    
    "ssp1_pressure": {
        'experiment': 'ssp1_2_6',
        'variable': 'surface_air_pressure',
        "filename": "ssp1_pressure.nc"},
    "ssp2_pressure": {
        'experiment': 'ssp2_4_5',
        'variable': 'surface_air_pressure',
        "filename": "ssp2_pressure.nc"},
    "ssp3_pressure": {
        'experiment': 'ssp3_7_0',
        'variable': 'surface_air_pressure',
        "filename": "ssp3_pressure.nc"},
    "ssp4_pressure": {
        'experiment': 'ssp4_6_0',
        'variable': 'surface_air_pressure',
        "filename": "ssp4_pressure.nc"},
    "ssp5_pressure": {
        'experiment': 'ssp5_8_5',
        'variable': 'surface_air_pressure',
        "filename": "ssp5_pressure.nc"},
    
    "ssp1_wind_speed": {
        'experiment': 'ssp1_2_6',
        'variable': 'near_surface_wind_speed',
        "filename": "ssp1_wind_speed.nc"},
    "ssp2_wind_speed": {
        'experiment': 'ssp2_4_5',
        'variable': 'near_surface_wind_speed',
        "filename": "ssp2_wind_speed.nc"},
    "ssp3_wind_speed": {
        'experiment': 'ssp3_7_0',
        'variable': 'near_surface_wind_speed',
        "filename": "ssp3_wind_speed.nc"},
    "ssp4_wind_speed": {
        'experiment': 'ssp4_6_0',
        'variable': 'near_surface_wind_speed',
        "filename": "ssp4_wind_speed.nc"},
    "ssp5_wind_speed": {
        'experiment': 'ssp5_8_5',
        'variable': 'near_surface_wind_speed',
        "filename": "ssp5_wind_speed.nc"},
    
    "ssp1_precipitation": {
        'experiment': 'ssp1_2_6',
        'variable': 'precipitation',
        "filename": "ssp1_precipitation.nc"},
    "ssp2_precipitation": {
        'experiment': 'ssp2_4_5',
        'variable': 'precipitation',
        "filename": "ssp2_precipitation.nc"},
    "ssp3_precipitation": {
        'experiment': 'ssp3_7_0',
        'variable': 'precipitation',
        "filename": "ssp3_precipitation.nc"},
    "ssp4_precipitation": {
        'experiment': 'ssp4_6_0',
        'variable': 'precipitation',
        "filename": "ssp4_precipitation.nc"},
    "ssp5_precipitation": {
        'experiment': 'ssp5_8_5',
        'variable': 'precipitation',
        "filename": "ssp5_precipitation.nc"}
    
}

# ESGF database 

atmospheric_urls_ssp = {
    "ssp1_ch4": {
        "url": "https://dpesgf03.nccs.nasa.gov/thredds/fileServer/CMIP6/ScenarioMIP/NASA-GISS/GISS-E2-1-H/ssp126/r1i1p3f1/AERmon/ch4/gn/v20201215/ch4_AERmon_GISS-E2-1-H_ssp126_r1i1p3f1_gn_201501-205012.nc"
    }, 
    "ssp2_ch4": {
        "url": "https://dpesgf03.nccs.nasa.gov/thredds/fileServer/CMIP6/ScenarioMIP/NASA-GISS/GISS-E2-1-H/ssp245/r1i1p3f1/AERmon/ch4/gn/v20201215/ch4_AERmon_GISS-E2-1-H_ssp245_r1i1p3f1_gn_201501-205012.nc"
    },
    "ssp3_ch4": {
        "url": "https://dpesgf03.nccs.nasa.gov/thredds/fileServer/CMIP6/ScenarioMIP/NASA-GISS/GISS-E2-1-H/ssp370/r1i1p3f1/AERmon/ch4/gn/v20201215/ch4_AERmon_GISS-E2-1-H_ssp370_r1i1p3f1_gn_201501-205012.nc"
    }, 
    "ssp4_ch4": {
        "url": "https://dpesgf03.nccs.nasa.gov/thredds/fileServer/CMIP6/ScenarioMIP/NASA-GISS/GISS-E2-1-H/ssp460/r1i1p3f1/AERmon/ch4/gn/v20200115/ch4_AERmon_GISS-E2-1-H_ssp460_r1i1p3f1_gn_201501-205012.nc"
    }, 
    "ssp5_ch4": {
        "url": "https://dpesgf03.nccs.nasa.gov/thredds/fileServer/CMIP6/ScenarioMIP/NASA-GISS/GISS-E2-1-H/ssp585/r1i1p3f1/AERmon/ch4/gn/v20200115/ch4_AERmon_GISS-E2-1-H_ssp585_r1i1p3f1_gn_201501-205012.nc"
    }
    
}

# FUNCTIONS

# DATA PROCESSING

# Download historical climate variable data

def download_historical_climate_variable(variable, start_year, end_year, data_subdirectory):
    years_list = [str(year) for year in range(start_year, end_year+1)]
    months_list = [format(month, '02') for month in range(1, 13)]
    
    cds_dict = cds_calls_historical[variable]
    
    c = cdsapi.Client()
    
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            'variable': cds_dict["variable"],
            'year': years_list,
            'month': months_list,
            'time': "00:00",
            'format': "netcdf",
        },
        os.path.join(data_subdirectory, cds_dict["filename"]))

# Download SSP climate variable data

def download_SSP_climate_variable(variable, ssp, start_year, end_year, data_subdirectory):
    years_list = [str(year) for year in range(start_year, end_year+1)]
    months_list = [format(month, '02') for month in range(1, 13)]
    
    cds_dict = cds_calls_ssp[f"ssp{ssp}_{variable}"]
    c = cdsapi.Client()
    c.retrieve(
        "projections-cmip6",
        {
            'temporal_resolution': "monthly",
            'experiment': cds_dict["experiment"],
            'variable': cds_dict["variable"],
            'model': 'ipsl_cm6a_lr',
            'year': years_list,
            'month': months_list,
            'format': 'zip',
        },
        os.path.join(data_subdirectory, "download.zip"))
    
    # Extract the contents of the zip file
    with zipfile.ZipFile(os.path.join(data_subdirectory, "download.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_subdirectory, "download"))
    
    # Find the .nc file in the extracted folder
    extracted_files = os.listdir(os.path.join(data_subdirectory, "download"))
    nc_files = [file for file in extracted_files if file.endswith(".nc")]
    
    os.rename(os.path.join(data_subdirectory, "download", nc_files[0]), os.path.join(data_subdirectory, cds_dict["filename"]))
    os.remove(os.path.join(data_subdirectory, "download.zip"))
    shutil.rmtree(os.path.join(data_subdirectory, "download"))

# Download climate variable data

def download_climate_variables_data(data_subdirectory, climate_variables, SSPs, start_year, end_year):
    # Create the output folder if it doesn't exist
    os.makedirs(data_subdirectory, exist_ok=True)
    
    # Download data
    for climate_variable in climate_variables:
        print(f"Downloading {climate_variable} data...")
        download_historical_climate_variable(climate_variable, start_year, end_year, data_subdirectory)
        for SSP in SSPs:
            download_SSP_climate_variable(climate_variable, SSP, start_year, end_year, data_subdirectory)
    print(f"Data downloaded to {data_subdirectory}")

# Download historical atmospheric gases data

def download_historical_atmospheric_variable(variable, data_subdirectory):
    cds_dict = cds_calls_historical[variable]
    
    c = cdsapi.Client()
    
    c.retrieve(
        cds_dict["name"],
        {
            'processing_level': cds_dict["processing_level"],
            'variable': cds_dict["variable"],
            'sensor_and_algorithm': cds_dict["sensor_and_algorithm"],
            'version': cds_dict["version"],
            'format': cds_dict["format"],
        },
        os.path.join(data_subdirectory, "download.zip"))

    # Extract the contents of the zip file
    with zipfile.ZipFile(os.path.join(data_subdirectory, "download.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_subdirectory, "download"))
    
    # Find the .nc file in the extracted folder
    extracted_files = os.listdir(os.path.join(data_subdirectory, "download"))
    nc_files = [file for file in extracted_files if file.endswith(".nc")]

    # Save .nc file
    os.rename(os.path.join(data_subdirectory, "download", nc_files[0]), os.path.join(data_subdirectory, cds_dict["filename"]))
    os.remove(os.path.join(data_subdirectory, "download.zip"))
    shutil.rmtree(os.path.join(data_subdirectory, "download"))

# Download SSP atmospheric gases data

def download_SSP_atmospheric_variable(variable, ssp, download_directory): # Download SSP climate variable data for years 2015-2050
   
    # URL of the file to download
    file_url = atmospheric_urls_ssp[f"ssp{ssp}_{variable}"]["url"]
    
    # Local file name for downloaded file
    local_filename = f"ssp{ssp}_{variable}.nc"
    download_filepath = os.path.join(download_directory, local_filename)

    # Send an HTTP GET request to the URL
    response = requests.get(file_url, stream=True)
    
    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        # Open the local file in binary write mode
        with open(download_filepath, 'wb') as f:
            # Iterate over the response content in chunks and write to the local file
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {local_filename} successfully.")
    else:
        print(f"Failed to download the file. HTTP status code: {response.status_code}")

# Download atmospheric gases data

def download_atmospheric_variables_data(data_subdirectory, atmospheric_variables, SSPs):
    # Create the output folder if it doesn't exist
    os.makedirs(data_subdirectory, exist_ok=True)
    
    # Download data
    for variable in atmospheric_variables:
        print(f"Downloading {variable} data...")
        download_historical_atmospheric_variable(variable, data_subdirectory)
        for ssp in SSPs:
            download_SSP_atmospheric_variable(variable, ssp, data_subdirectory)
    print(f"Data downloaded to {data_subdirectory}")

# Process NetCDF4 data (climate variables, atmospheric data)

def process_netcdf_data(data_subdirectory, output_subdirectory, 
                        country_shapes_filepath, country_codes_filepath, alignments_filepath, 
                        frequency):
    print("Processing files...")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_subdirectory, exist_ok=True)
    
    # Open the country shapes file from Natural Earth
    country_shapes = gpd.read_file(country_shapes_filepath)
    
    # Read in the OECD country codes .csv
    OECD_countries = pd.read_csv(country_codes_filepath, header=None)
    OECD_countries.columns = ["country_code", "country_name"]
    
    # Read in the alignments .json
    with open(alignments_filepath, 'r') as file:
        loaded_alignments = json.load(file)
    
    # List all netCDF files in the input folder
    netcdf_files = [f for f in os.listdir(data_subdirectory) if f.endswith(".nc")]
    
    for netcdf_file in netcdf_files:
        print(netcdf_file)
    
        # Open the netCDF file using xarray
        file_path = os.path.join(data_subdirectory, netcdf_file)
        dataset = xr.open_dataset(file_path)
        
        # Rename the coordinates
        if 'latitude' in dataset.coords and 'longitude' in dataset.coords:
            new_coords = {'latitude': 'lat', 'longitude': 'lon'}
            dataset = dataset.rename(new_coords)
     
        # For datasets with multiple atmospheric levels, select data for the highest level (corresponding to near-surface)
        if "lev" in dataset.dims:
            dataset = dataset.sel(lev=dataset["lev"].max().item())
            dataset = dataset.drop("lev")

        # For surface datasets with a single height level, drop this coordinate
        if "height" in dataset.coords:
            if dataset["height"].values.shape == ():
                dataset = dataset.drop("height")

        # Check if expver is a dimension in the xarray dataset and call the merge_exp_versions function
        if "expver" in list(dataset.dims):
            dataset = merge_exp_versions(dataset)        

        # Identify the variable code
        variable_code = [variable_name for variable_name in dataset.data_vars 
                         if set(dataset[variable_name].dims) == set(dataset.coords)][0]
        
        # Call the aggregate_country_data function
        aggregated_dataset = aggregate_country_data(dataset, country_shapes)
        
        # Convert xarray dataset to pandas dataframe
        dataframe = convert_nc_to_csv(aggregated_dataset, variable_code)
        
        # Resample the dataframe to "frequency" time steps
        try: 
            resampled_dataframe = dataframe.resample(frequency).mean()
        except TypeError as e:
            dataframe.index = pd.to_datetime(dataframe.index.astype(str))
            resampled_dataframe = dataframe.resample(frequency).mean()

        # Filter out the duplicated columns
        df = resampled_dataframe.loc[:,~resampled_dataframe.columns.duplicated()].copy()
        
        # Loop through the alignments and apply the country code alignment function
        for alignment in loaded_alignments:
            OECD_code = alignment["OECD_code"]
            dataset_code = alignment["dataset_code"]
            country_name = alignment["country_name"]
            use_OECD_code = alignment["use_OECD_code"]
            manually_align_countries(df, OECD_countries, 
                                     OECD_code, dataset_code, country_name, use_OECD_code)
       
        # Create the output file path
        output_file_name = netcdf_file.split(".")[0]
        
        # Create the output folder if it doesn't exist
        os.makedirs(output_subdirectory, exist_ok=True)
        
        # Save the aggregated data to a new CSV file
        output_file_path = os.path.join(output_subdirectory, f"{output_file_name}.csv")
        df.to_csv(output_file_path)
        
        # Save the updated OECD country codes file separately
        updated_file_name = f"updated_{os.path.split(country_codes_filepath)[-1]}"
        updated_file_path = os.path.join(os.path.split(country_codes_filepath)[0], updated_file_name)
        OECD_countries.to_csv(updated_file_path)

    print(f"Processed data files saved to {output_subdirectory}")

# Aggregate xarray dataset to country level using lat/lon mask

def aggregate_country_data(dataset, country_shapes):
    aggregated_data = []

    # Countries loop with tqdm
    for index, country_row in tqdm(country_shapes.iterrows(), total = len(country_shapes), desc="Aggregating"):
        country_code = country_row["SOV_A3"]
        country_geometry = country_row["geometry"]

        # Create a boolean mask using the geometry and the xarray dataset coordinates
        country_mask = dataset.interp(lat=country_geometry.centroid.y, lon=country_geometry.centroid.x)

        # Apply the mask to the dataset
        country_aggregated = dataset.where(country_mask)

        # Compute the mean for each variable
        country_mean = country_aggregated.mean(dim=["lat", "lon"])
        country_mean["country"] = country_code
        aggregated_data.append(country_mean)

    # Concatenate the list of aggregated datasets along the country dimension
    aggregated_dataset = xr.concat(aggregated_data, dim="country")
    return aggregated_dataset

# Flatten xarray dataset with multiple experiment versions

def merge_exp_versions(dataset):
    
    # Start with the data in the first experiment version
    merged_dataset = dataset.sel(expver=dataset["expver"].values[0])
    
    # Loop through subsequent versions and fill in with the non-NaN values
    for version in dataset["expver"].values[1:]:
        merged_dataset = merged_dataset.combine_first(dataset.sel(expver=version))

    return merged_dataset

# Convert xarray dataset to pandas dataframe

def convert_nc_to_csv(dataset, variable_code):
    # Extract the necessary data variables
    time_steps = dataset.time.values
    countries = dataset.country.values
    variable = dataset[variable_code].values

    # Reshape the data into a 2D array
    # Rows: time steps, Columns: countries
    variable_reshaped = variable.transpose(1, 0)

    # Create a pandas DataFrame
    df = pd.DataFrame(data=variable_reshaped, index=time_steps, columns=countries)
    return df

# Align country codes across different data sources

def manually_align_countries(dataframe, OECD_df, OECD_country_code, dataset_country_code, country_name, use_OECD_code=True):

    if dataset_country_code not in dataframe.columns:
        print(f"Chosen dataset country code '{dataset_country_code}' not found in the dataset columns.")
        return

    # Update the dataset column name
    dataset_column_name_to_update = dataset_country_code
    new_country_code = OECD_country_code if use_OECD_code else dataset_country_code
    new_dataset_column_name = new_country_code

    # Update the column name in the dataset
    dataframe.rename(columns={dataset_column_name_to_update: new_dataset_column_name}, inplace=True)

    # # Update the OECD_countries dataframe
    # # Find the index of the chosen_country_code_OECD in the OECD_countries dataframe
    # index_to_update = OECD_df[OECD_df['country_code'] == OECD_country_code].index[0]

    # # Update the country code and country name in the OECD_countries dataframe
    # OECD_df.at[index_to_update, 'country_code'] = new_country_code
    # OECD_df.at[index_to_update, 'country_name'] = country_name

# Extract socioeconomic SSP data

def extract_socioeconomic_ssp_data(ssp_database_filepath, output_directory, 
                                   socioeconomic_variables, SSPs, start_year, end_year, 
                                   frequency):
    print(f"Extracting socioeconomic data from {ssp_database_filepath}")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Read the csv file as a pandas dataframe
    df = pd.read_csv(ssp_database_filepath)
    
    # Define years list (where years are multiples of 5)
    years_list = [str(year) for year in range((((start_year - 1) // 5) * 5), (((end_year // 5) + 1) * 5) + 1, 5)]

    for variable in socioeconomic_variables:
        print(f"Processing {variable}...")
        for ssp in tqdm(SSPs):
        
            # Filter for specified rows
            scenario_df = df[(df['MODEL'] == "OECD Env-Growth") & 
                             (df['SCENARIO'] == f"SSP{ssp}_v9_130325") & 
                             (df['VARIABLE'] == variable)]
            
            # Select the desired columns (years from 2015 to 2023)
            filter_columns = ["REGION", "UNIT"] + years_list
            scenario_df = scenario_df[filter_columns]
            
            # Create a dictionary to store the transformed data
            transformed_data = {'REGION': [], **{year: [] for year in years_list}}
            
            # Define the conversion factors for units
            conversion_factors = {
                'million': 1000000,
                'billion': 1000000000
            }
            
            # Iterate through the rows of the original dataframe
            for index, row in scenario_df.iterrows():
                region = row['REGION']
                unit = row['UNIT']
                
                # Determine the appropriate conversion factor
                factor = 1  # Default factor, no transformation
                for key, value in conversion_factors.items():
                    if key in unit:
                        factor = value
                        break
                
                transformed_data['REGION'].append(region)
                
                # Apply the conversion factor to each year's data and update transformed_data
                for column in transformed_data:
                    if column != 'REGION':
                        transformed_data[column].append(row[column] * factor)
            
            # Create the new transformed dataframe
            scenario_df = pd.DataFrame(transformed_data)
            
            # Set REGION column as the index
            scenario_df.set_index('REGION', inplace=True)
            
            # Transpose the dataframe
            scenario_df = scenario_df.transpose()
            
            # Change data type to integers
            scenario_df = scenario_df.astype(int)
            
            # Convert the index to datetime objects for the final day of the year
            scenario_df.index = pd.to_datetime(scenario_df.index + '-12-31', format='%Y-%m-%d')
            
            # Interpolate values for each year
            scenario_df_interpolated = scenario_df.resample(frequency).interpolate()
            scenario_df_interpolated = scenario_df_interpolated.astype(int)
    
            # Save the dataframe
            variable_name = variable.split("|")[0].lower()
            output_filepath = os.path.join(output_directory, f"ssp{ssp}_{variable_name}.csv")
            scenario_df_interpolated.to_csv(output_filepath)
    print(f"Saved output to {output_directory}")

# Process historical population data

def process_population_historical_data(data_filepath, output_directory, 
                                       frequency):
    print(f"Processing {data_filepath}")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Read the csv file as a pandas dataframe
    historical_df = pd.read_csv(data_filepath)
    
    # Pivot the dataframe to reshape it
    pivot_df = historical_df.pivot(index="Time", columns="LOCATION", values='Value')
    
    # Rename the columns to remove the 'Time' prefix
    pivot_df.columns.name = None
    
    # Change data type of values to integers
    historical_df = pivot_df.astype(int)
    
    # Convert the index to strings
    historical_df.index = historical_df.index.astype(str)
    
    # Convert the index to datetime objects for the final day of the year
    historical_df.index = pd.to_datetime(historical_df.index + '-12-31', format='%Y-%m-%d')
    
    # Interpolate values for each year
    historical_df_interpolated = historical_df.resample(frequency).interpolate()
    historical_df_interpolated = historical_df_interpolated.astype(int)

    # Save the dataframe as a csv file
    output_filepath = os.path.join(output_directory, f"historical_population.csv")
    historical_df_interpolated.to_csv(output_filepath)
    print(f"Output saved to {output_filepath}")

# Process historical GDP data

def process_GDP_historical_data(data_filepath, output_directory, 
                                frequency):
    print(f"Processing {data_filepath}")

    # Create the output folder if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Read the csv file as a pandas dataframe
    historical_df = pd.read_csv(data_filepath)
    
    # Select the desired columns 
    filter_columns = ["LOCATION", "TIME", "MEASURE", "Value"]
    historical_df = historical_df[filter_columns]
    
    # Filter for specified rows
    historical_df = historical_df[(historical_df["MEASURE"] == "MLN_USD")]
    
    # Multiplying values by 1,000,000 when MEASURE is MLN_USD
    historical_df['Value'] = historical_df.apply(lambda row: row['Value'] * 1000000 if row['MEASURE'] == 'MLN_USD' else row['Value'], axis=1)
    
    # Pivoting the DataFrame
    historical_df = historical_df.pivot(index='TIME', columns='LOCATION', values='Value')
    
    # Change index type to string
    historical_df.index = historical_df.index.astype(str)
    
    # Convert the index to datetime objects for the final day of the year
    historical_df.index = pd.to_datetime(historical_df.index + '-12-31', format='%Y-%m-%d')
    
    # Interpolate values for each year
    historical_df_interpolated = historical_df.resample(frequency).interpolate()

    # Save the dataframe as a csv file
    output_filepath = os.path.join(output_directory, f"historical_gdp.csv")
    historical_df_interpolated.to_csv(output_filepath)
    print(f"Saved output to {output_filepath}")

# TRACE 

def get_country_features(csv_files, country_code, date_index, frequency, normalise=True):

    features = []
    
    for file in csv_files:
 
        # Read the csv file
        df = pd.read_csv(file)

        # Convert the date column to datetime format
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        # Set the datetime index and remove the name
        df.set_index(df.columns[0], inplace=True)
        df.index.name = None

        # Trim the dataframe to the date range
        df = df.loc[date_index]

        # Aggregate to annual time steps
        df = df.resample(frequency).mean()

        # Get the features 
        data = df[country_code].values

        if normalise:
            # Calculate the minimum and maximum values in the array
            min_val = np.min(data)
            max_val = np.max(data)
            
            # Scale the array to be between 0 and 1
            output_data = (data - min_val) / (max_val - min_val)
        else:
            output_data = data
        
        # Append to the features list
        features.append(output_data)

    return features    

def get_x0(i, features): # Get features for this time step, t=0
    
    # Create slices to represent x0 and x1
    x0_slices = np.array(features)[:, :-1]  # All features except the last time step
    
    # Transpose the slices to get the desired arrangement
    x0 = x0_slices.T

    return x0[i]

def get_xt1(i, features): # Get features for the next time step, t+1

    # Create slices to represent x0 and x1
    x_slices = np.array(features)[:, 1:]   # All features except the first time step
        
    # Transpose the slices to get the desired arrangement
    x = x_slices.T

    return x[i]

def cumulative_sum_trace(scores):
    '''
    Calculate cumulative average TraCE score
    The final value is an average of the entire trajectory

    '''
    df = pd.DataFrame(scores, columns=['score'])
    return df['score'].expanding().sum().to_numpy()

# PLOTTING

# Plot TraCE scores for each SSP for a country
def plot_country_ssp_trace_scores(country_code, SSPs, start_date, end_date, frequency, data_folder, country_codes_df, angle_weight=0.9, cumulative_sum=True, figsize=(20, 6), title=True):

    country_name = country_codes_df[country_codes_df['code'] == country_code]["name"].values[0]
    
    time_steps_dict = {
        "M": "monthly",
        "Q": "quarterly",
        "A": "annual"}
    
    # Create the date range for analysis
    date_index = pd.date_range(start=start_date, end=end_date, freq=frequency)

    # HISTORICAL
    # Get lists of files for historical data
    historical_csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv") and "ssp" not in f]
    historical_csv_files.sort()
    # Extract features
    try:
        historical_features = get_country_features(historical_csv_files, country_code, date_index, frequency)
    except:
        return

    variables_list = [os.path.split(file)[-1].replace("historical_", "").replace(".csv", "") for file in historical_csv_files]
    variables = ", ".join(variables_list)
    if cumulative_sum:
        description = f"{country_name} {time_steps_dict[frequency]} cumulative TraCE score with variables: {variables}"
    else:
        description = f"{country_name} {time_steps_dict[frequency]} instantaneous TraCE score with variables: {variables}"
    if not title:
        print(description)

    # Create lists to store data for plotting
    plot_data = {}  # Dictionary to store data for each SSP
    
    for ssp in SSPs:
    
        # Get lists of files for a single SSP
        ssp_csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv") and f"ssp{ssp}" in f]
        ssp_csv_files.sort()

        try:
            # Extract features
            ssp_features = get_country_features(ssp_csv_files, country_code, date_index, frequency)
        except Exception as e:
            return  # Skip 
        
        # Calculate TraCE scores
        scores = []
    
        for i in range(len(date_index)-1):
            x0 = get_x0(i, historical_features)
            x1 = get_xt1(i, historical_features)
            x_prime = get_xt1(i, ssp_features)
        
            trace = score(x0, x1, x_prime, func=lambda v : angle_weight) # weight ratio of angle to distance
            
            scores.append(trace)
        
        plot_data[ssp] = (date_index[:-1], scores)
    
    if not plot_data:  # Skip if there's no data to plot
        return

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot using datetime index and numpy array
    for ssp, (x_data, y_data) in plot_data.items():
        if cumulative_sum:
            y_data = cumulative_sum_trace(y_data)
        ax.plot(x_data, y_data, label=f"SSP{ssp}")
    
    plt.xlim(date_index.min(), date_index.max())
    plt.ylim(0, y_data.max()+0.1*y_data.max())
    
    # Add labels
    plt.xticks(rotation=45)
    plt.xlabel('date')
    plt.ylabel('score')
    # plt.ylim((0,12))
    plt.legend()

    # Add title
    if title:
        plt.title(description)
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
# Plot two normalised features against each other, for all the SSPs and historical data
def plot_two_features(feature_x, feature_y, country_code, SSPs, start_date, end_date, frequency, data_folder, country_codes_df, figsize=(10,10), title=True):

    country_name = country_codes_df[country_codes_df['code'] == country_code]["name"].values[0]
    description = f'{feature_y} vs. {feature_x} ({country_name})'
    if not title:
        print(description)
    
    # Create the date range for analysis
    date_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Loop through SSPs
    for ssp in SSPs:
        ssp_csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv") and f"ssp{ssp}" in f and (feature_x in f or feature_y in f)]
        ssp_csv_files.sort()
        
        # Extract features
        ssp_features = get_country_features(ssp_csv_files, country_code, date_index, frequency)
    
        # Get the column indices for the selected features
        variables_list = [os.path.split(file)[-1].replace(f"ssp{ssp}_", "").replace(".csv", "") for file in ssp_csv_files]
        idx_x = variables_list.index(feature_x)
        idx_y = variables_list.index(feature_y)
        x_values_ssp = ssp_features[idx_x]
        y_values_ssp = ssp_features[idx_y]
    
        # Create a line plot connecting the data points for the current SSP scenario
        ax.plot(x_values_ssp, y_values_ssp, marker='o', linestyle='-', label=f"SSP{ssp}", alpha=0.3)
    
    # HISTORICAL
    historical_csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv") and "historical" in f and (feature_x in f or feature_y in f)]
    historical_csv_files.sort()
    
    # Extract features
    historical_features = get_country_features(historical_csv_files, country_code, date_index, frequency)
    
    # Get the column indices for the selected features
    variables_list = [os.path.split(file)[-1].replace("historical_", "").replace(".csv", "") for file in historical_csv_files]
    idx_x = variables_list.index(feature_x)
    idx_y = variables_list.index(feature_y)
    
    # Extract the data for the selected features
    x_values = historical_features[idx_x]
    y_values = historical_features[idx_y]
    
    # Create a line plot connecting the data points for historical data
    ax.plot(x_values, y_values, marker='o', linestyle='-', label="historical")

    # Add labels to the first and final data points with the year
    first_year = date_index[0].year
    final_year = date_index[-1].year
    ax.annotate(first_year, (x_values[0], y_values[0]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(final_year, (x_values[-1], y_values[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    
    # Customise the plot
    ax.set_xlabel(f"{feature_x} (normalised)")
    ax.set_ylabel(f"{feature_y} (normalised)")

    if title:
        ax.set_title(description)
    
    # Show the legend
    ax.legend()
    
    # Show the plot
    plt.show()

# Plot two normalised features against each other, for all the SSPs and historical data
def plot_feature_timeseries(feature, country_code, SSPs, start_date, end_date, frequency, data_folder, country_codes_df, figsize=(20, 6), title=True):

    country_name = country_codes_df[country_codes_df['code'] == country_code]["name"].values[0]
    time_steps_dict = {
        "M": "monthly", 
        "Q": "quarterly",
        "A": "annual"}
    description = f'{time_steps_dict[frequency]} {feature} ({country_name})'
    if not title:
        print(description)
    
    # Create the date range for analysis
    date_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Loop through SSPs
    for ssp in SSPs:
        ssp_csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv") and f"ssp{ssp}" in f and feature in f]
        ssp_csv_files.sort()
        
        # Extract features
        ssp_features = get_country_features(ssp_csv_files, country_code, date_index, frequency)
    
        # Get the column indices for the selected features to extract the values
        variables_list = [os.path.split(file)[-1].replace(f"ssp{ssp}_", "").replace(".csv", "") for file in ssp_csv_files]
        idx = variables_list.index(feature)
        values_ssp = ssp_features[idx]
    
        # Create a line plot connecting the data points for the current SSP scenario
        ax.plot(date_index, values_ssp, marker='o', linestyle='-', label=f"SSP{ssp}", alpha=0.3)
    
    # HISTORICAL
    historical_csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv") and "historical" in f and feature in f]
    historical_csv_files.sort()
    
    # Extract features
    historical_features = get_country_features(historical_csv_files, country_code, date_index, frequency)
    
    # Get the column indices for the selected features
    variables_list = [os.path.split(file)[-1].replace("historical_", "").replace(".csv", "") for file in historical_csv_files]
    idx = variables_list.index(feature)
    
    # Extract the data for the selected features
    values = historical_features[idx]
    
    # Create a line plot connecting the data points for historical data
    ax.plot(date_index, values, marker='o', linestyle='-', label="historical")

    # Customise the plot
    ax.set_xlabel(f"date")
    ax.set_ylabel(f"{feature} (normalised)")
    ax.set_xlim(date_index.min(), date_index.max())
    ax.set_ylim(0, values.max()+0.1*values.max())
    
    if title:
        ax.set_title(description)
    
    # Show the legend
    ax.legend()
    
    # Show the plot
    plt.show()