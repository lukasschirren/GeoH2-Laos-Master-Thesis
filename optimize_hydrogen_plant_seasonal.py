# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:47:57 2023

@author: Claire Halloran, University of Oxford

Includes code from Nicholas Salmon, University of Oxford, for optimizing
hydrogen plant capacity.

"""

from osgeo import gdal
import atlite
import geopandas as gpd
import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import p_H2_aux as aux
from functions import CRF
import numpy as np
import logging
import time

import xarray as xr
from scipy.constants import physical_constants
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.ERROR)


def hydropower_potential(eta,flowrate,head):
    '''
    Calculate hydropower potential in Megawatts

    Parameters
    ----------
    eta : float
        Efficiency of the hydropower plant.
    flowrate : float
        Flowrate calculated with runoff multiplied by the hydro-basin area, in cubic meters per hour.
    head : float
        Height difference at the hydropower site, in meters.

    Returns
    -------
    float
        Hydropower potential in Megawatts (MW).
    '''
    rho = 997 # kg/m3; Density of water
    g = physical_constants['standard acceleration of gravity'][0] # m/s2; Based on the CODATA constants 2018
    Q = flowrate / 3600 # transform flowrate per h into flowrate per second
    return (eta * rho * g * Q * head) / (1000 * 1000) # MW

def hydropower_potential_with_capacity(flowrate, head, capacity, eta):
    '''
    Calculate the hydropower potential considering the capacity limit

    Parameters
    ----------
    flowrate : float
        Flowrate calculated with runoff multiplied by the hydro-basin area, in cubic meters per hour.
    head : float
        Height difference at the hydropower site, in meters.
    capacity : float
        Maximum hydropower capacity in Megawatts (MW).
    eta : float
        Efficiency of the hydropower plant.

    Returns
    -------
    xarray DataArray
        Capacity factor, which is the limited potential divided by the capacity.
    '''
    potential = hydropower_potential(flowrate, head, eta)
    limited_potential = xr.where(potential > capacity, capacity, potential)
    capacity_factor = limited_potential / capacity
    return capacity_factor

#########################################################################################################

# Sizing of electrolysers will need to be based on the average excess electricity available in each time slice
# Input: Minimum load factor of the elctrolyser
def demand_schedule(quantity, transport_state, transport_excel_path,
                             weather_excel_path):
    '''
    calculates hourly hydrogen demand for truck shipment and pipeline transport.

    Parameters
    ----------
    quantity : float
        annual amount of hydrogen to transport in kilograms.
    transport_state : string
        state hydrogen is transported in, one of '500 bar', 'LH2', 'LOHC', or 'NH3'.
    transport_excel_path : string
        path to transport_parameters.xlsx file
    weather_excel_path : string
        path to transport_parameters.xlsx file
            
    Returns
    -------
    trucking_hourly_demand_schedule : pandas DataFrame
        hourly demand profile for hydrogen trucking.
    pipeline_hourly_demand_schedule : pandas DataFrame
        hourly demand profile for pipeline transport.
    '''
    transport_parameters = pd.read_excel(transport_excel_path,
                                         sheet_name = transport_state,
                                         index_col = 'Parameter'
                                         ).squeeze('columns')
    weather_parameters = pd.read_excel(weather_excel_path,
                                       index_col = 'Parameters',
                                       ).squeeze('columns')
    truck_capacity = transport_parameters['Net capacity (kg H2)']
    start_date = weather_parameters['Start date']
    end_date = weather_parameters['End date (not inclusive)']

    # Adjust capacity based on excess
    # schedule for trucking
    annual_deliveries = quantity/truck_capacity
    quantity_per_delivery = quantity/annual_deliveries
    index = pd.date_range(start_date, end_date, periods=annual_deliveries)
    trucking_demand_schedule = pd.DataFrame(quantity_per_delivery, index=index, columns = ['Demand'])
    trucking_hourly_demand_schedule = trucking_demand_schedule.resample('h').sum().fillna(0.)

    # schedule for pipeline
    index = pd.date_range(start_date, end_date, freq = 'h')
    pipeline_hourly_quantity = quantity/index.size
    pipeline_hourly_demand_schedule = pd.DataFrame(pipeline_hourly_quantity, index=index,  columns = ['Demand'])

    return trucking_hourly_demand_schedule, pipeline_hourly_demand_schedule


def optimize_hydrogen_plant(wind_potential, pv_potential, hydro_potential, times, demand_profile, 
                            wind_max_capacity, pv_max_capacity, hydro_max_capacity,
                            country_series, water_limit=None):
    '''
    Optimizes the size of green hydrogen plant components based on renewable potential, hydrogen demand, and country parameters. 

    Parameters
    ----------
    wind_potential : xarray DataArray
        1D dataarray of per-unit wind potential in hexagon.
    pv_potential : xarray DataArray
        1D dataarray of per-unit solar potential in hexagon.
    times : xarray DataArray
        1D dataarray with timestamps for wind and solar potential.
    demand_profile : pandas DataFrame
        hourly dataframe of hydrogen demand in kg.
    country_series : pandas Series
        interest rate and lifetime information.
    water_limit : float, optional
        annual limit on water available for electrolysis in hexagon, in cubic meters. Default is None.
    hydro_potential : xarray DataArray, optional
        1D dataarray of per-unit hydro potential in hexagon. Default is None.
    hydro_max_capacity : float, optional
        maximum hydro capacity in MW. Default is None.

    Returns
    -------
    lcoh : float
        levelized cost per kg hydrogen.
    wind_capacity: float
        optimal wind capacity in MW.
    solar_capacity: float
        optimal solar capacity in MW.
    hydro_capacity: float
        optimal hydro capacity in MW.
    electrolyzer_capacity: float
        optimal electrolyzer capacity in MW.
    battery_capacity: float
        optimal battery storage capacity in MW/MWh (1 hour batteries).
    h2_storage: float
        optimal hydrogen storage capacity in MWh.
    '''
   
    # if a water limit is given, check if hydrogen demand can be met
    if water_limit != None:
        # total hydrogen demand in kg
        total_hydrogen_demand = demand_profile['Demand'].sum()
        # check if hydrogen demand can be met based on hexagon water availability
        water_constraint =  total_hydrogen_demand <= water_limit * 111.57 # kg H2 per cubic meter of water
        if water_constraint == False:
            print('Not enough water to meet hydrogen demand!')
            # return null values
            lcoh = np.nan
            wind_capacity = np.nan
            solar_capacity = np.nan
            hydro_capacity = np.nan
            electrolyzer_capacity = np.nan
            battery_capacity = np.nan
            h2_storage = np.nan
            return lcoh, wind_capacity, solar_capacity, hydro_capacity, electrolyzer_capacity, battery_capacity, h2_storage

    # Set up network
    n = pypsa.Network(override_component_attrs=aux.create_override_components())
    n.set_snapshots(times)

    # Import the design of the H2 plant into the network
    n.import_from_csv_folder("Parameters_3/Basic_H2_plant")

    # Add electrolyzer as a link with unit commitment
    electrolyzer_min_on_time = 3  # Minimum 3 hours of operation
    electrolyzer_min_off_time = 2  # Minimum 2 hours of downtime
    n.add("Link", "Electrolysis",
          bus0="Electricity",
          bus1="Hydrogen",
        #   efficiency=0.7, 
          p_nom_extendable=True,  # Allow capacity to be optimized
          unit_commitment=True)  # Enable unit commitment

    # Set minimum on/off times
    n.links.at["Electrolysis", "min_up_time"] = electrolyzer_min_on_time
    n.links.at["Electrolysis", "min_down_time"] = electrolyzer_min_off_time

    # Import demand profile
    # Note: All flows are in MW or MWh, conversions for hydrogen done using HHVs. Hydrogen HHV = 39.4 MWh/t
    n.add('Load', 'Hydrogen demand', bus='Hydrogen', p_set=demand_profile['Demand'] / 1000 * 39.4)

    # Send the weather data to the model
    n.generators_t.p_max_pu['Wind'] = wind_potential
    n.generators_t.p_max_pu['Solar'] = pv_potential
    n.generators_t.p_max_pu['Hydro'] = hydro_potential

    # Specify maximum capacity based on land use
    n.generators.loc['Hydro', 'p_nom_max'] = hydro_max_capacity
    n.generators.loc['Wind', 'p_nom_max'] = wind_max_capacity * 2  # Rated power - here 2 MW
    n.generators.loc['Solar', 'p_nom_max'] = pv_max_capacity * 1  # Power per node - here 1 MW

    # Specify technology-specific and country-specific WACC and lifetime
    n.generators.loc['Hydro', 'capital_cost'] *= CRF(country_series['Hydro interest rate'], 
                                                    country_series['Hydro lifetime'])
    n.generators.loc['Wind', 'capital_cost'] *= CRF(country_series['Wind interest rate'], 
                                                    country_series['Wind lifetime (years)'])
    n.generators.loc['Solar', 'capital_cost'] *= CRF(country_series['Solar interest rate'], 
                                                     country_series['Solar lifetime (years)'])
    for item in [n.links, n.stores, n.storage_units]:
        item.capital_cost = item.capital_cost * CRF(country_series['Plant interest rate'], country_series['Plant lifetime (years)'])

    # Solve the model with unit commitment
    solver = 'gurobi'
    n.lopf(solver_name=solver,
           solver_options={'LogToConsole': 0, 'OutputFlag': 0},
           pyomo=True,  # Enable Pyomo for unit commitment
           extra_functionality=aux.extra_functionalities)

    # Output results
    lcoh = n.objective / (n.loads_t.p_set.sum().iloc[0] / 39.4 * 1000)  # Convert back to kg H2
    wind_capacity = n.generators.p_nom_opt['Wind']
    solar_capacity = n.generators.p_nom_opt['Solar']
    hydro_capacity = n.generators.p_nom_opt.get('Hydro', np.nan)  # Get hydro capacity if it exists
    electrolyzer_capacity = n.links.p_nom_opt['Electrolysis'] 
    battery_capacity = n.storage_units.p_nom_opt.get('Battery', np.nan)
    h2_storage = n.stores.e_nom_opt.get('Compressed H2 Store', np.nan)
    print(lcoh)
    return lcoh, wind_capacity, solar_capacity, hydro_capacity, electrolyzer_capacity, battery_capacity, h2_storage


if __name__ == "__main__":
    transport_excel_path = "Parameters_3/transport_parameters.xlsx"
    weather_excel_path = "Parameters_3/weather_parameters.xlsx"
    country_excel_path = 'Parameters_3/country_parameters.xlsx'
    country_parameters = pd.read_excel(country_excel_path,
                                        index_col='Country')
    demand_excel_path = 'Parameters_3/demand_parameters.xlsx'
    demand_parameters = pd.read_excel(demand_excel_path,
                                      index_col='Demand center',
                                      ).squeeze("columns")
    demand_centers = demand_parameters.index
    weather_parameters = pd.read_excel(weather_excel_path,
                                       index_col = 'Parameters'
                                       ).squeeze('columns')
    weather_filename = weather_parameters['Filename']

    hexagons = gpd.read_file('Parameters_3/hex_transport.geojson')
    # !!! change to name of cutout in weather
    cutout = atlite.Cutout('Cutouts_23/' + weather_filename +'.nc')
    layout = cutout.uniform_layout()
    
    ###############################################################
    # Added for hydropower
    location_hydro = gpd.read_file('Data_3/hydropower_dams.gpkg')
    location_hydro.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
    # # Monthly generation columns
    # monthly_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # # Hours in each month for the year 2023
    # hours_in_month = {
    #     'Jan': 744, 'Feb': 672, 'Mar': 744, 'Apr': 720,
    #     'May': 744, 'Jun': 720, 'Jul': 744, 'Aug': 744,
    #     'Sep': 720, 'Oct': 744, 'Nov': 720, 'Dec': 744
    # }

    # # Initialize an empty DataFrame to store hourly capacity factors
    # hourly_capacity_factors = pd.DataFrame()

    
    # # Iterate over each plant
    # for index, row in location_hydro.iterrows():
    #     plant_name = row['Name']
    #     plant_capacity = row['Total capacity (MW)']
        
    #     # Generate an empty list to hold hourly capacity factors for the plant
    #     plant_hourly_factors = []

    #     # Spread the generation for each month evenly across the respective hours
    #     for month in monthly_columns:
    #         monthly_generation = row[month] * 1000  # Convert GWh to MWh
    #         hourly_generation = monthly_generation / hours_in_month[month]  # MWh per hour
    #         hourly_capacity_factor = hourly_generation / plant_capacity  # Capacity factor
            
    #         # Repeat this capacity factor for each hour of the month
    #         plant_hourly_factors.extend([hourly_capacity_factor] * hours_in_month[month])

    #     # Add the plant's hourly capacity factors to the DataFrame
    #     hourly_capacity_factors[plant_name] = plant_hourly_factors

    # # Extend the data by adding the last hour's capacity factors for an additional 24 hours
    # last_hour = hourly_capacity_factors.iloc[-1]  # Get the last hour's data
    # extended_hours = pd.DataFrame([last_hour] * 24, columns=hourly_capacity_factors.columns)
    # hourly_capacity_factors = pd.concat([hourly_capacity_factors, extended_hours], ignore_index=True)

    # # Create a time index for the year 2023 plus the first hour of 2024
    # time_index = pd.date_range(start='2023-01-01', end='2024-01-01 23:00', freq='H')

    # # Ensure the time index matches the length of the DataFrame
    # hourly_capacity_factors.index = time_index
    
    # # Convert the DataFrame to an xarray DataArray
    # capacity_factor_array = xr.DataArray(
    #     data=hourly_capacity_factors.values,
    #     dims=['time', 'plant'],
    #     coords={'time': time_index, 'plant': location_hydro['Name'].values}
    # )
    # PRESUMABLY NOT RELEVANT

    capacity_factor_array = xr.open_dataarray("capacity_factors_dry_2025.nc")

    if 'index_left' in location_hydro.columns:
        location_hydro = location_hydro.rename(columns={'index_left': 'index_left_renamed'})
    if 'index_right' in location_hydro.columns:
        location_hydro = location_hydro.rename(columns={'index_right': 'index_right_renamed'})
    if 'index_left' in hexagons.columns:
        hexagons = hexagons.rename(columns={'index_left': 'index_left_renamed'})
    if 'index_right' in hexagons.columns:
        hexagons = hexagons.rename(columns={'index_right': 'index_right_renamed'})

    # Perform the spatial join
    hydro_hex_mapping = gpd.sjoin(location_hydro, hexagons, how='left', predicate='within')
    hydro_hex_mapping = hydro_hex_mapping.drop_duplicates(subset=['Name'])
    hydro_hex_mapping['plant_index'] = hydro_hex_mapping.index

    # Number of hexagons and time steps (from the capacity factor array)
    num_hexagons = len(hexagons)
    num_time_steps = len(capacity_factor_array.time)

    # Initialize the final xarray DataArray to store capacity factors for each hexagon
    hydro_profile = xr.DataArray(
        data=np.zeros((num_hexagons, num_time_steps)),
        dims=['hexagon', 'time'],
        coords={'hexagon': np.arange(num_hexagons), 'time': capacity_factor_array.time}
    )
    
    # Ensure that plants_in_hex contains valid plant names

    for hex_index in range(num_hexagons):
        plants_in_hex = hydro_hex_mapping[hydro_hex_mapping['index_right'] == hex_index]['Name'].tolist()
        valid_plants_in_hex = [plant for plant in plants_in_hex if plant in capacity_factor_array['plant'].values]

        if len(valid_plants_in_hex) > 0:
            # Select the capacity factors for all plants in the current hexagon
            hex_capacity_factors = capacity_factor_array.sel(plant=valid_plants_in_hex)
                
            # Get the capacities for these plants
            plant_capacities = xr.DataArray(location_hydro.set_index('Name').loc[valid_plants_in_hex]['Total capacity (MW)'].values, dims=['plant'])

            # Weighted average based on plant capacities
            weights = plant_capacities / plant_capacities.sum()
            weighted_avg_capacity_factor = (hex_capacity_factors * weights).sum(dim='plant')
            
            # Store the result in the hydro_profile array
            hydro_profile.loc[hex_index] = weighted_avg_capacity_factor    
    ###############################################################

    pv_profile = cutout.pv(
        panel= 'CSi',
        orientation='latitude_optimal',
        layout = layout,
        shapes = hexagons,
        per_unit = True
        )
    pv_profile = pv_profile.rename(dict(dim_0='hexagon'))

    wind_profile = cutout.wind(
        # Changed turbine type - was Vestas_V80_2MW_gridstreamer in first run
        # Other option being explored: NREL_ReferenceTurbine_2020ATB_4MW, Enercon_E126_7500kW
        turbine = 'Vestas_V80_2MW_gridstreamer',
        layout = layout,
        shapes = hexagons,
        per_unit = True
        )
    wind_profile = wind_profile.rename(dict(dim_0='hexagon'))


# wind_potential, pv_potential, hydro_potential, times, demand_profile, 
# wind_max_capacity, pv_max_capacity, hydro_max_capacity,
# country_series, water_limit=None

    for location in demand_centers:
        # Trucking optimization
        lcohs_trucking = np.zeros(len(pv_profile.hexagon))
        solar_capacities = np.zeros(len(pv_profile.hexagon))
        wind_capacities = np.zeros(len(pv_profile.hexagon))
        hydro_capacities = np.zeros(len(pv_profile.hexagon))
        electrolyzer_capacities = np.zeros(len(pv_profile.hexagon))
        battery_capacities = np.zeros(len(pv_profile.hexagon))
        h2_storages = np.zeros(len(pv_profile.hexagon))
        start = time.process_time()
        
        for hexagon in pv_profile.hexagon.data:
            hydrogen_demand_trucking, _ = demand_schedule(
                demand_parameters.loc[location,'Annual demand [kg/a]'],
                hexagons.loc[hexagon,f'{location} trucking state'],
                transport_excel_path,
                weather_excel_path)
            country_series = country_parameters.loc[hexagons.country[hexagon]]
            lcoh, wind_capacity, solar_capacity, hydro_capacity, electrolyzer_capacity, battery_capacity, h2_storage =\
                optimize_hydrogen_plant(wind_profile.sel(hexagon = hexagon),
                                    pv_profile.sel(hexagon = hexagon),
                                    hydro_profile.sel(hexagon = hexagon),
                                    wind_profile.time,
                                    hydrogen_demand_trucking,
                                    hexagons.loc[hexagon,'theo_turbines'],
                                    hexagons.loc[hexagon,'theo_pv'],
                                    hexagons.loc[hexagon,'hydro'],
                                    country_series,
                                    # water_limit = hexagons.loc[hexagon,'delta_water_m3']
                                    )
            lcohs_trucking[hexagon] = lcoh
            solar_capacities[hexagon] = solar_capacity
            wind_capacities[hexagon] = wind_capacity
            hydro_capacities[hexagon] = hydro_capacity
            electrolyzer_capacities[hexagon] = electrolyzer_capacity
            battery_capacities[hexagon] = battery_capacity
            h2_storages[hexagon] = h2_storage
        trucking_time = time.process_time()-start

        hexagons[f'{location} trucking solar capacity'] = solar_capacities
        hexagons[f'{location} trucking wind capacity'] = wind_capacities
        hexagons[f'{location} trucking hydro capacity'] = hydro_capacities
        hexagons[f'{location} trucking electrolyzer capacity'] = electrolyzer_capacities
        hexagons[f'{location} trucking battery capacity'] = battery_capacities
        hexagons[f'{location} trucking H2 storage capacity'] = h2_storages
        hexagons[f'{location} trucking production cost'] = lcohs_trucking

        print(f"Trucking optimization for {location} completed in {trucking_time} s")

        # Pipeline optimization
        lcohs_pipeline = np.zeros(len(pv_profile.hexagon))
        solar_capacities = np.zeros(len(pv_profile.hexagon))
        wind_capacities = np.zeros(len(pv_profile.hexagon))
        hydro_capacities = np.zeros(len(pv_profile.hexagon))
        electrolyzer_capacities = np.zeros(len(pv_profile.hexagon))
        battery_capacities = np.zeros(len(pv_profile.hexagon))
        h2_storages = np.zeros(len(pv_profile.hexagon))
        start = time.process_time()

        for hexagon in pv_profile.hexagon.data:
            _, hydrogen_demand_pipeline = demand_schedule(
                demand_parameters.loc[location,'Annual demand [kg/a]'],
                hexagons.loc[hexagon,f'{location} trucking state'],
                transport_excel_path,
                weather_excel_path)
            country_series = country_parameters.loc[hexagons.country[hexagon]]
            lcoh, wind_capacity, solar_capacity, hydro_capacity, electrolyzer_capacity, battery_capacity, h2_storage =\
                optimize_hydrogen_plant(wind_profile.sel(hexagon = hexagon),
                                    pv_profile.sel(hexagon = hexagon),
                                    hydro_profile.sel(hexagon = hexagon),
                                    wind_profile.time,
                                    hydrogen_demand_pipeline,
                                    hexagons.loc[hexagon,'theo_turbines'],
                                    hexagons.loc[hexagon,'theo_pv'],
                                    hexagons.loc[hexagon,'hydro'],
                                    country_series,
                                    # water_limit = hexagons.loc[hexagon,'delta_water_m3']
                                    )
            lcohs_pipeline[hexagon] = lcoh
            solar_capacities[hexagon] = solar_capacity
            wind_capacities[hexagon] = wind_capacity
            hydro_capacities[hexagon] = hydro_capacity
            electrolyzer_capacities[hexagon] = electrolyzer_capacity
            battery_capacities[hexagon] = battery_capacity
            h2_storages[hexagon] = h2_storage

        pipeline_time = time.process_time() - start

        hexagons[f'{location} pipeline solar capacity'] = solar_capacities
        hexagons[f'{location} pipeline wind capacity'] = wind_capacities
        hexagons[f'{location} pipeline hydro capacity'] = hydro_capacities
        hexagons[f'{location} pipeline electrolyzer capacity'] = electrolyzer_capacities
        hexagons[f'{location} pipeline battery capacity'] = battery_capacities
        hexagons[f'{location} pipeline H2 storage capacity'] = h2_storages
        hexagons[f'{location} pipeline production cost'] = lcohs_pipeline

        print(f"Pipeline optimization for {location} completed in {pipeline_time} s")

    hexagons.to_file('Resources/0Seasonality/2019/hex_lcoh.geojson', driver='GeoJSON', encoding='utf-8')
    # hexagons.to_file('Resources/hex_lcoh_base_case.geojson', driver='GeoJSON', encoding='utf-8')
    