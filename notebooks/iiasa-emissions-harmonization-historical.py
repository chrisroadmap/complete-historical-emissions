# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

# %%
emissions = pd.read_feather('../data/input/harmonisation-history_0002_0002_0002_0002_0003_0002_0002_0002_0002_95b5f2c9fb62e32a4d08fe2ffc5b4a6ff246ad2d/db/0.feather')

# %%
emissions

# %%
emissions.index.get_level_values('purpose').unique()

# %%
global_historical = emissions.xs('global_workflow_emissions', level='purpose').loc[:, 1750:2024]
global_historical.drop('Emissions|Halon1202', level='variable', inplace=True)

# %%
global_historical

# %%
# CO2 and N2O are in silly units, so let's fix
global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|Energy and Industrial Processes', :] = (
    global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|Energy and Industrial Processes', :] * 0.001
)

global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|AFOLU', :] = (
    global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|AFOLU', :] * 0.001
)

global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|N2O', :] = (
    global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|N2O', :] * 0.001
)

# %%
idx = global_historical.index
idx

# %%
idx.get_level_values('unit')=='Mt CO2/yr'

# %%
global_historical.index = global_historical.index.set_levels(global_historical.index.levels[4].str.replace('Mt CO2/yr', 'Gt CO2/yr'), level=4)

# %%
global_historical.index = global_historical.index.set_levels(global_historical.index.levels[4].str.replace('kt N2O/yr', 'Mt N2O/yr'), level=4)

# %%
global_historical

# %%
global_historical

# %%
# let's check that the emissions look ok
fig, ax = pl.subplots(13, 4, figsize = (16, 40))
ivar = 0
for variable in global_historical.index.get_level_values('variable'):
    ax[ivar//4, ivar%4].plot(
        np.arange(1750, 2025), 
        global_historical.xs(variable, level='variable').loc[:, 1750:2024].T
    )
    ax[ivar//4, ivar%4].set_title(variable)
    ivar = ivar + 1
fig.tight_layout()

# %%
# CO2 FFI is 0.8% higher than 2023 (provisional estimate, GCB)
global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|Energy and Industrial Processes', 2024] = (
    global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|Energy and Industrial Processes', 2023] * 1.008
)

# %%
# in land use, preliminary estimates are 4.2 GtCO2 (https://essd.copernicus.org/articles/17/965/2025/essd-17-965-2025.html)
# the GtC to GtCO2 conversion isn't quite right - perhaps both numbers are rounded
# in which case use the GtCO2 value since the 2 sf precision is a smaller fraction of the estimate for 4.2 versus 1.2
global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|AFOLU', 2024] = 4.2

# %%
# let's see what we still need to extrapolate
# Find all the species that don't have a 2024 value; these need extrapolating
last_valid_year = {}
for variable in global_historical.index.get_level_values('variable'):
    lvy = global_historical.loc[global_historical.index.get_level_values('variable')==variable, 1750:].squeeze(axis=0).last_valid_index()
    if lvy != '2024':  # ignore complete variables
        last_valid_year[variable] = lvy

# %%
last_valid_year

# %%
# I suggest it is probably fine to extrapolate forward using the trend of the last 5 years
fig, ax = pl.subplots(13, 4, figsize = (16, 30))
ivar = 0
for variable in last_valid_year:
    ax[ivar//4, ivar%4].plot(
        np.arange(2000, int(last_valid_year[variable]) + 1), 
        global_historical.loc[global_historical.index.get_level_values('variable')==variable, 2000:last_valid_year[variable]].T,
        color='sienna',
        label='ScenarioMIP'
    )
    trend = (
        global_historical.loc[global_historical.index.get_level_values('variable')==variable, last_valid_year[variable]] -
        global_historical.loc[global_historical.index.get_level_values('variable')==variable, last_valid_year[variable]-1]
    ).values[0]
    ax[ivar//4, ivar%4].plot(
        np.arange(last_valid_year[variable], 2025, 1), 
        (
            global_historical.loc[global_historical.index.get_level_values('variable')==variable, last_valid_year[variable]].values[0] + 
            trend * (np.arange(2025-int(last_valid_year[variable])))
        ),
        color='skyblue',
        label='cubic spline'
    )
    ax[ivar//4, ivar%4].set_title(variable)
    ivar = ivar + 1
fig.tight_layout()

# %%
# let's implement
fig, ax = pl.subplots(13, 4, figsize = (16, 30))
ivar = 0
for variable in last_valid_year:
    trend = (
        global_historical.loc[global_historical.index.get_level_values('variable')==variable, last_valid_year[variable]] -
        global_historical.loc[global_historical.index.get_level_values('variable')==variable, last_valid_year[variable]-1]
    ).values[0]
    global_historical.loc[global_historical.index.get_level_values('variable')==variable, last_valid_year[variable]:2024] = (
        global_historical.loc[global_historical.index.get_level_values('variable')==variable, last_valid_year[variable]].values[0] + 
        trend * (np.arange(2025-last_valid_year[variable]))
    )
    ax[ivar//4, ivar%4].plot(
        np.arange(2000, 2025), 
        global_historical.loc[global_historical.index.get_level_values('variable')==variable, 2000:2024].T,
        color='skyblue',
    )
    ax[ivar//4, ivar%4].set_title(variable)
    ivar = ivar + 1
fig.tight_layout()

# %%
# too painful to automate
scenariomip_to_fair = {
    'Emissions|HFC|HFC23': 'HFC-23',
    'Emissions|BC': 'BC',
    'Emissions|CH4': 'CH4', 
    'Emissions|CO': 'CO',
    'Emissions|N2O': 'N2O',
    'Emissions|NH3': 'NH3',
    'Emissions|NOx': 'NOx',
    'Emissions|OC': 'OC',
    'Emissions|Sulfur': 'Sulfur',
    'Emissions|VOC': 'VOC',
    'Emissions|CO2|Energy and Industrial Processes': 'CO2 FFI',
    'Emissions|C3F8': 'C3F8',
    'Emissions|C4F10': 'C4F10',
    'Emissions|C5F12': 'C5F12',
    'Emissions|C6F14': 'C6F14',
    'Emissions|C7F16': 'C7F16',
    'Emissions|C8F18': 'C8F18',
    'Emissions|CH2Cl2': 'CH2Cl2',
    'Emissions|CH3Br': 'CH3Br',
    'Emissions|CH3Cl': 'CH3Cl',
    'Emissions|CHCl3': 'CHCl3',
    'Emissions|NF3': 'NF3',
    'Emissions|SO2F2': 'SO2F2',
    'Emissions|cC4F8': 'c-C4F8',
    'Emissions|CO2|AFOLU': 'CO2 AFOLU',
    'Emissions|HFC|HFC125': 'HFC-125',
    'Emissions|HFC|HFC134a': 'HFC-134a',
    'Emissions|HFC|HFC143a': 'HFC-143a',
    'Emissions|HFC|HFC152a': 'HFC-152a',
    'Emissions|HFC|HFC227ea': 'HFC-227ea',
    'Emissions|HFC|HFC236fa': 'HFC-236fa',
    'Emissions|HFC|HFC245fa': 'HFC-245fa',
    'Emissions|HFC|HFC32': 'HFC-32',
    'Emissions|HFC|HFC365mfc': 'HFC-365mfc',
    'Emissions|HFC|HFC43-10': 'HFC-4310mee',
    'Emissions|CCl4': 'CCl4',
    'Emissions|CFC11': 'CFC-11',
    'Emissions|CFC113': 'CFC-113',
    'Emissions|CFC114': 'CFC-114',
    'Emissions|CFC115': 'CFC-115',
    'Emissions|CFC12': 'CFC-12',
    'Emissions|CH3CCl3': 'CH3CCl3',
    'Emissions|HCFC141b': 'HCFC-141b',
    'Emissions|HCFC142b': 'HCFC-142b',
    'Emissions|HCFC22': 'HCFC-22',
    'Emissions|Halon1211': 'Halon-1211',
    'Emissions|Halon1301': 'Halon-1301',
    'Emissions|Halon2402': 'Halon-2402',
    'Emissions|C2F6': 'C2F6',
    'Emissions|CF4': 'CF4',
    'Emissions|SF6': 'SF6'
}

# %%
for variable in global_historical.index.get_level_values('variable'):
    global_historical.index = global_historical.index.set_levels(global_historical.index.levels[3].str.replace(variable, scenariomip_to_fair[variable]), level=3)

# %%
# gets caught by CO
global_historical.index = global_historical.index.set_levels(global_historical.index.levels[3].str.replace('CO2|Energy and Industrial Processes', 'CO2 FFI'), level=3)
global_historical.index = global_historical.index.set_levels(global_historical.index.levels[3].str.replace('CO2|AFOLU', 'CO2 AFOLU'), level=3)

# %%
global_historical

# %%
# one final thing: add a total CO2, which is useful for harmonization
co2_total = (
    global_historical.loc[global_historical.index.get_level_values('variable').str.startswith('CO2')]
).sum(axis=0)
# global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|AFOLU', :] = (
#     global_historical.loc[global_historical.index.get_level_values('variable')=='Emissions|CO2|AFOLU', :] * 0.001
# )
global_historical.loc[('reconstructed', 'historical', 'World', 'CO2', 'Gt CO2/yr'), :] = co2_total

# %%
global_historical

# %%
os.makedirs('../data/output', exist_ok=True)

# %%
global_historical.to_csv('../data/output/historical_emissions_1750-2024.csv', index=False)

# %%
