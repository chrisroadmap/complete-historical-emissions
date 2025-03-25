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

# %% [markdown]
# # Complete historical emissions
#
# This script makes a file of historical emissions from 1750-2024 that is suitable for harmonization to future scenarios.
#
# The input data is from IIASA's emissions harmonization historical at https://github.com/iiasa/emissions_harmonization_historical 
#
# Do we want two versions?
# 1. historical including biomass burning variability to 2024
# 2. harmonization for future scenarios, by smoothing out the biomass burning by using e.g. a 5- or 10-year smoothing filter.
#
# The starting point is the historical emissions from CMIP7 ScenarioMIP: we want these to be exactly consistent with the ScenarioMIP final emissions from 1990 to 2021 for all species. This is the period which harmonization will occur. Where datasets are complete and final, extend before 1990 or after 2021, and present in ScenarioMIP, we will also use these. Where this is not the case, we will make some justifyable assumptions/extrapolations/extensions that ScenarioMIP would consider too fruity, but are Good Enough For Government Work.
#
# The historical emissions will be used to produce v1.5.0 of the fair calibration.
#
# This will use several data sources; we will try to clearly label them!

# %%
import warnings
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import CubicSpline

# %%
# the ScenarioMIP state of play
df_scenariomip = pd.read_csv('../data/input/cmip7_history_world_0022.csv')

# %%
df_scenariomip

# %%
# we can crib this to make our actual emissions dataset
df_emissions = df_scenariomip.copy()

# %%
# firstly, we don't want to extrapolate past 2024
df_emissions = df_emissions.loc[:, :'2024']
df_emissions

# %%
# secondly, CO2 and N2O are in silly* units, so let's fix
df_emissions.loc[df_emissions['variable']=='Emissions|N2O', '1750':] = df_emissions.loc[df_emissions['variable']=='Emissions|N2O', '1750':] / 1000
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|Energy and Industrial Processes', '1750':] = (
    df_emissions.loc[df_emissions['variable']=='Emissions|CO2|Energy and Industrial Processes', '1750':] / 1000
)
# we haven't gone before 1850 yet for CO2 AFOLU
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|AFOLU', '1850':] = (
    df_emissions.loc[df_emissions['variable']=='Emissions|CO2|AFOLU', '1850':] / 1000
)

df_emissions.loc[df_emissions['variable']=='Emissions|N2O', 'unit'] = 'Mt N2O/yr'
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|Energy and Industrial Processes', 'unit'] = 'Gt CO2/yr'
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|AFOLU', 'unit'] = 'Gt CO2/yr'
df_emissions
# *ok, I can see an argument for CO2 in Mt, but N2O in kt is batshit.

# %%
# thirdly, Halon-1202 isn't included in any CMIP6 scenarios, and its historical emissions were tiny and are now pretty much zero
# so let's remove it
df_emissions.drop(df_emissions.loc[df_emissions['variable']=='Emissions|Montreal Gases|Halon1202'].index, inplace=True)
df_emissions

# %%
# Find all the species that don't have a 1750 value; these need backfilling
first_valid_year = {}
for variable in df_emissions.variable:
    fvy = df_emissions.loc[df_emissions['variable']==variable, '1750':].squeeze(axis=0).first_valid_index()
    if fvy != '1750':  # ignore complete variables
        first_valid_year[variable] = fvy

# %%
first_valid_year

# %%
# so, everything other than CO2 AFOLU can be filled in with inversions
df_inverse = pd.read_csv('../data/input/inverse_emissions_0012.csv')

# %%
len(first_valid_year) - 1

# %%
df_inverse

# %%
# let's check that the emissions look ok
fig, ax = pl.subplots(7, 4, figsize = (16, 30))
ivar = 0
for variable in first_valid_year:
    if variable != 'Emissions|CO2|AFOLU':
        ax[ivar//4, ivar%4].plot(
            np.arange(1900, 2023), 
            df_inverse.loc[df_inverse['variable']==variable, '1900':].T,
            label='inverse',
            color='indigo',
        )
        ax[ivar//4, ivar%4].plot(
            np.arange(int(first_valid_year[variable]), 2025), 
            df_emissions.loc[df_emissions['variable']==variable, first_valid_year[variable]:].T,
            color='sienna',
            label='ScenarioMIP'
        )
        ax[ivar//4, ivar%4].set_title(variable)
        ivar = ivar + 1
fig.tight_layout()

# %% [markdown]
# Joining logic:
#
# - HFC23: join with scale factor
# - HFC125: join
# - HFC134a: join
# - HFC143a: join
# - HFC152a: join
# - HFC227ea: join
# - HFC236fa: join
# - HFC245fa: join
# - HFC32: join
# - HFC365mfc: join
# - HFC43-10: join
# - CCl4: join with scale factor
# - CFC11: join
# - CFC113: join
# - CFC114: join
# - CFC115: join
# - CFC12: join
# - CH3CCl3: join
# - HCFC141b: join
# - HCFC142b: join
# - HCFC22: join
# - Halon1211: join
# - Halon1301: join
# - Halon2402: join
# - C2F6: join from about 1980
# - CF4: join from about 1980
# - SF6: join from about 1980

# %%
# special cases: scale factors

# HFC23 - no obvious trend so use period mean
pl.plot(
    df_emissions.loc[df_emissions['variable']=='Emissions|HFC|HFC23', first_valid_year['Emissions|HFC|HFC23']:'2022'].values.squeeze()
    /df_inverse.loc[df_inverse['variable']=='Emissions|HFC|HFC23', first_valid_year['Emissions|HFC|HFC23']:'2022'].values.squeeze()
)
hfc23_sf = (
    df_emissions.loc[df_emissions['variable']=='Emissions|HFC|HFC23', first_valid_year['Emissions|HFC|HFC23']:'2022'].values.squeeze()
    /df_inverse.loc[df_inverse['variable']=='Emissions|HFC|HFC23', first_valid_year['Emissions|HFC|HFC23']:'2022'].values.squeeze()
).mean()
hfc23_sf

# %%
df_emissions.loc[df_emissions['variable']=='Emissions|HFC|HFC23', '1750':str(int(first_valid_year['Emissions|HFC|HFC23'])-1)] = (
    hfc23_sf * df_inverse.loc[df_inverse['variable']=='Emissions|HFC|HFC23', '1750':str(int(first_valid_year['Emissions|HFC|HFC23'])-1)].values
)

# %%
# CCl4 - looks like a trend so regress it
ccl4_ratio = (
    df_emissions.loc[df_emissions['variable']=='Emissions|Montreal Gases|CCl4', first_valid_year['Emissions|Montreal Gases|CCl4']:'2022'].values.squeeze()
    /df_inverse.loc[df_inverse['variable']=='Emissions|Montreal Gases|CCl4', first_valid_year['Emissions|Montreal Gases|CCl4']:'2022'].values.squeeze()
)

pl.plot(ccl4_ratio)

lreg = linregress(np.arange(len(ccl4_ratio)), ccl4_ratio)

pl.plot(np.arange(len(ccl4_ratio)), lreg.slope * np.arange(len(ccl4_ratio)) + lreg.intercept)

ccl4_sf = lreg.intercept
ccl4_sf

# %%
df_emissions.loc[df_emissions['variable']=='Emissions|Montreal Gases|CCl4', '1750':str(int(first_valid_year['Emissions|Montreal Gases|CCl4'])-1)] = (
    ccl4_sf * df_inverse.loc[df_inverse['variable']=='Emissions|Montreal Gases|CCl4', '1750':str(int(first_valid_year['Emissions|Montreal Gases|CCl4'])-1)].values
)

# %%
# join with inverse: most others
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for variable in [
        'Emissions|HFC|HFC125',
        'Emissions|HFC|HFC134a',
        'Emissions|HFC|HFC143a',
        'Emissions|HFC|HFC152a',
        'Emissions|HFC|HFC227ea',
        'Emissions|HFC|HFC236fa',
        'Emissions|HFC|HFC245fa',
        'Emissions|HFC|HFC32',
        'Emissions|HFC|HFC365mfc',
        'Emissions|HFC|HFC43-10',
        'Emissions|Montreal Gases|CFC|CFC11',
        'Emissions|Montreal Gases|CFC|CFC113',
        'Emissions|Montreal Gases|CFC|CFC114',
        'Emissions|Montreal Gases|CFC|CFC115',
        'Emissions|Montreal Gases|CFC|CFC12',
        'Emissions|Montreal Gases|CH3CCl3',
        'Emissions|Montreal Gases|Halon1211',
        'Emissions|Montreal Gases|Halon1301',
        'Emissions|Montreal Gases|Halon2402',
        'Emissions|Montreal Gases|HCFC141b',
        'Emissions|Montreal Gases|HCFC142b',
        'Emissions|Montreal Gases|HCFC22'
    ]:
        # handle duplicates
        # print(df_inverse.loc[df_inverse['variable']==variable])
        # row_select = df_inverse.loc[df_inverse['variable']==variable].iloc[0, :]
        df_emissions.loc[df_emissions['variable']==variable, '1750':str(int(first_valid_year[variable])-1)] = (
            df_inverse.loc[df_inverse['variable']==variable, '1750':str(int(first_valid_year[variable])-1)].values
        )
        #print(df_inverse.loc[df_inverse['variable']==variable, '1750':str(int(first_valid_year[variable])-1)])
        # df_emissions.loc[df_emissions['variable']==variable, '1750':str(int(first_valid_year[variable])-1)] = (
        #     row_select['1750':str(int(first_valid_year[variable])-1)].to_frame().T
        # )

# %%
# join about 1980
df_inverse.loc[df_inverse['variable']=='Emissions|SF6', "1750":"1979"]

# %%
# join about 1980
for variable in ['Emissions|SF6', 'Emissions|C2F6', 'Emissions|CF4']:
    df_emissions.loc[df_emissions['variable']==variable, '1750':'1979'] = (
        df_inverse.loc[df_inverse['variable']==variable, "1750":"1979"].values
    )

# %%
# CO2 AFOLU: Zeb wants more of a spline cut in 1750-1850; but it's such a piddly small forcing and very uncertain so linear is fine IMO
# 30 GtC cumulative is what GCB say for 1750-1850
linear_ramp_up = np.arange(0.003, 0.6, 0.006)

# %%
# check adds up to 30 GtC
np.cumsum(linear_ramp_up)[-1]

# %%
# and is 100 years long
len(linear_ramp_up)

# %%
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|AFOLU', '1750':'1849'] = linear_ramp_up * 44.009/12.011  # put in GtCO2 units

# %%
# implement IGCC2024 extensions of SLCFs for 2024
# in FFI & ag., this is based on CEDS (soon to be published) for 2023, and CAMS for 2024
# for biomass burning, this is based on GFED (public until 2023, 2024 version supplied by Guido van der Werf).
# see https://github.com/ClimateIndicator/forcing-timeseries/tree/ceds2025/notebooks (to update once merged)
df_slcf = pd.read_csv('../data/input/slcf_emissions_1750-2024.csv', index_col=0)

name_map = {species: f'Emissions|{species}' for species in df_slcf.columns}
name_map['SO2'] = 'Emissions|Sulfur'
name_map['NMVOC'] = 'Emissions|VOC'

for species in df_slcf.columns:
    df_emissions.loc[df_emissions['variable']==name_map[species], '2024'] = df_slcf.loc[2024, species]

# %%
# include the CO2 emissions from 2024 in GCB in here too
# we assume that CEDS does not include cement carbonation (think explicitly mentioned somewhere)
# GCB: preliminary estimates are 0.8% above 2023 for 2024 (https://essd.copernicus.org/articles/17/965/2025/essd-17-965-2025.html)
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|Energy and Industrial Processes', '2024'] = (
    1.008 *
    df_emissions.loc[df_emissions['variable']=='Emissions|CO2|Energy and Industrial Processes', '2023']
)
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|Energy and Industrial Processes', '2015':]


# %%
# in land use, preliminary estimates are 4.2 GtCO2 (https://essd.copernicus.org/articles/17/965/2025/essd-17-965-2025.html)
# the GtC to GtCO2 conversion isn't quite right - perhaps both numbers are rounded
# in which case use the GtCO2 value since the 2 sf precision is a smaller fraction of the estimate for 4.2 versus 1.2
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|AFOLU', '2024'] = 4.2
df_emissions.loc[df_emissions['variable']=='Emissions|CO2|AFOLU', '2015':]

# %%
# For CH4 and N2O, we can use CAMS estimate for 2024 plus GFED from Guido
# This is prepared for the Climate Indicators 2024
df_emissions.loc[df_emissions['variable']=='Emissions|CH4', '2024'] = df_slcf.loc[2024, 'CH4']
df_emissions.loc[df_emissions['variable']=='Emissions|N2O', '2024'] = df_slcf.loc[2024, 'N2O']

# %%
df_emissions#.loc[df_emissions['variable']==f'Emissions|{species}', str(year)]

# %%
# let's see what we still need to extrapolate
# Find all the species that don't have a 2024 value; these need extrapolating
last_valid_year = {}
for variable in df_emissions.variable:
    lvy = df_emissions.loc[df_emissions['variable']==variable, '1750':].squeeze(axis=0).last_valid_index()
    if lvy != '2024':  # ignore complete variables
        last_valid_year[variable] = lvy

# %%
last_valid_year

# %%
len(last_valid_year)

# %%
# let's check that the emissions look ok
# I suggest it is probably fine to extrapolate forward using the trend of the last 5 years
fig, ax = pl.subplots(5, 4, figsize = (16, 25))
ivar = 0
for variable in last_valid_year:
    ax[ivar//4, ivar%4].plot(
        np.arange(2000, int(last_valid_year[variable]) + 1), 
        df_emissions.loc[df_emissions['variable']==variable, '2000':last_valid_year[variable]].T,
        color='sienna',
        label='ScenarioMIP'
    )
    trend = (
        df_emissions.loc[df_emissions['variable']==variable, last_valid_year[variable]] -
        df_emissions.loc[df_emissions['variable']==variable, str(int(last_valid_year[variable])-1)]
    ).values[0]
    ax[ivar//4, ivar%4].plot(
        np.arange(int(last_valid_year[variable]), 2025, 1), 
        (
            df_emissions.loc[df_emissions['variable']==variable, last_valid_year[variable]].values[0] + 
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
fig, ax = pl.subplots(5, 4, figsize = (16, 25))
ivar = 0
for variable in last_valid_year:
    trend = (
        df_emissions.loc[df_emissions['variable']==variable, last_valid_year[variable]] -
        df_emissions.loc[df_emissions['variable']==variable, str(int(last_valid_year[variable])-1)]
    ).values[0]
    df_emissions.loc[df_emissions['variable']==variable, str(int(last_valid_year[variable])):'2024'] = (
        df_emissions.loc[df_emissions['variable']==variable, last_valid_year[variable]].values[0] + 
        trend * (np.arange(2025-int(last_valid_year[variable])))
    )
    ax[ivar//4, ivar%4].plot(
        np.arange(2000, 2025), 
        df_emissions.loc[df_emissions['variable']==variable, '2000':'2024'].T,
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
    'Emissions|Montreal Gases|CH2Cl2': 'CH2Cl2',
    'Emissions|Montreal Gases|CH3Br': 'CH3Br',
    'Emissions|Montreal Gases|CH3Cl': 'CH3Cl',
    'Emissions|Montreal Gases|CHCl3': 'CHCl3',
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
    'Emissions|Montreal Gases|CCl4': 'CCl4',
    'Emissions|Montreal Gases|CFC|CFC11': 'CFC-11',
    'Emissions|Montreal Gases|CFC|CFC113': 'CFC-113',
    'Emissions|Montreal Gases|CFC|CFC114': 'CFC-114',
    'Emissions|Montreal Gases|CFC|CFC115': 'CFC-115',
    'Emissions|Montreal Gases|CFC|CFC12': 'CFC-12',
    'Emissions|Montreal Gases|CH3CCl3': 'CH3CCl3',
    'Emissions|Montreal Gases|HCFC141b': 'HCFC-141b',
    'Emissions|Montreal Gases|HCFC142b': 'HCFC-142b',
    'Emissions|Montreal Gases|HCFC22': 'HCFC-22',
    'Emissions|Montreal Gases|Halon1211': 'Halon-1211',
    'Emissions|Montreal Gases|Halon1301': 'Halon-1301',
    'Emissions|Montreal Gases|Halon2402': 'Halon-2402',
    'Emissions|C2F6': 'C2F6',
    'Emissions|CF4': 'CF4',
    'Emissions|SF6': 'SF6'
}

# %%
# for ease, give all "model" column the same name and make the variable name consistent with fair
df_emissions['model']='reconstructed'
df_emissions['variable'] = df_emissions['variable'].replace(scenariomip_to_fair)

# %%
df_emissions

# %%
os.makedirs('../data/output', exist_ok=True)

# %%
# save out the first raw version
df_emissions.to_csv('../data/output/historical_emissions_1750-2024.csv', index=False)

# %%
# version for historical harmonizations:
# rename variable
# dump everything before 2015 (we don't harmonize before then)
# take 5 year running mean of all SLCFs (NOT SULFUR BECAUSE MOSTLY EEI) and CO2 AFOLU as the harmonization value
# extrapolate 2023 and 2024 using trend of 5-year running means
# make same model (not sure if needed, but will help)
df_historical_harmonization = df_emissions.copy()
variables_to_mean = ['BC', 'OC', 'CO', 'NH3', 'NOx', 'VOC', 'CO2 AFOLU']
for variable in variables_to_mean:
    for year in range(2015, 2023):
        df_historical_harmonization.loc[df_historical_harmonization['variable']==variable, str(year)] = (
            df_emissions.loc[df_emissions['variable']==variable, str(year-2):str(year+2)].mean(axis=1).values
        )
    lrr = (
        linregress(
            np.arange(2018, 2023),
            df_historical_harmonization.loc[df_historical_harmonization['variable']==variable, '2018':'2022']
        )
    )
    df_historical_harmonization.loc[df_historical_harmonization['variable']==variable, '2023':'2024'] = (
        lrr.slope * np.array((2023, 2024)) + lrr.intercept
    )
to_drop = [str(year) for year in range(1750, 2014)]
df_historical_harmonization.drop(columns=to_drop, inplace=True)
#df_historical_harmonization['model']='reconstructed'
#df_historical_harmonization['variable'] = df_historical_harmonization['variable'].replace(scenariomip_to_fair)
    #list(df_historical_harmonization['variable'].values)
#df_historical_harmonization.replace(to_replace='Emissions|*', value='', regex=True)
df_historical_harmonization

# %%
# save out the version for historical harmonization
df_historical_harmonization.to_csv('../data/output/historical_harmonization_5yr_running_means_2014-2024.csv', index=False)

# %%
# # finally, make into a fair format emissions file
# # drop model and put on half-years


# this is not modified from above yet


# df_fair_calibrate = df_emissions.copy()
# variables_to_mean = ['BC', 'OC', 'CO', 'NH3', 'NOx', 'VOC', 'CO2|AFOLU']
# for variable in variables_to_mean:
#     for year in range(2015, 2023):
#         df_historical_harmonization.loc[df_historical_harmonization['variable']==f'Emissions|{variable}', str(year)] = (
#             df_emissions.loc[df_emissions['variable']==f'Emissions|{variable}', str(year-2):str(year+2)].mean(axis=1).values
#         )
#     lrr = (
#         linregress(
#             np.arange(2018, 2023),
#             df_historical_harmonization.loc[df_historical_harmonization['variable']==f'Emissions|{variable}', '2018':'2022']
#         )
#     )
#     df_historical_harmonization.loc[df_historical_harmonization['variable']==f'Emissions|{variable}', '2023':'2024'] = (
#         lrr.slope * np.array((2023, 2024)) + lrr.intercept
#     )
# to_drop = [str(year) for year in range(1750, 2014)]
# df_historical_harmonization.drop(columns=to_drop, inplace=True)
# df_historical_harmonization['model']='reconstructed'
# df_historical_harmonization['variable'] = df_historical_harmonization['variable'].replace(scenariomip_to_fair)
#     #list(df_historical_harmonization['variable'].values)
# #df_historical_harmonization.replace(to_replace='Emissions|*', value='', regex=True)
# df_historical_harmonization

# %%
# make into a fair-format emissions input file
# rename variable
# drop model
# put on half years
