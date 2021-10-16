from tabulate import tabulate

from tabulate import tabulate

table = [['Name','Data Type','Units','Description'],
         ['sy_snum', 'Numeric', 'NA', 'Number of stars in the system'],
         ['sy_pnum', 'Numeric', 'NA', 'Number of planets in the system'],
         ['cb_flag', 'Nominal categorical', 'NA', 'Circumbinary flag: whether the planet orbits 2 stars'],
         ['pl_orbper', 'Numeric', 'Earth days', 'Orbital period (Time it takes planet to complete an orbit'],
         ['pl_orbsmax', 'Numeric', 'au', 'Orbit semi-Major Axis. au is the distance from Earth to sun.'],
         ['pl_rade', 'Numeric', 'Earth radius', 'Planet radius, where 1.0 is Earth\'s radius'],
         ['pl_bmasse', 'Numeric', 'Earth Mass', 'Planetary Mass, where 1.0 is Earth\'s mass'],
         ['pl_orbeccen', 'Numeric', 'NA', 'Planet\s orbital eccentricity'],
         ['pl_eqt', 'Numeric', 'Kelvin', 'Equilibrium Temperature: (The planetary equilibrium temperature is a the theoretical temperature that a planet would be a black body being heated only by its parent star)'],
         ['st_teff', 'Numeric', 'Kelvin', 'Stellar Effective Temperature'],
         ['st_rad', 'Numeric', 'Solar Radius', 'Stellar Radius, where 1.0 is 1 of our Sun\'s radius'],
         ['st_mass', 'Numeric', 'Solar Mass', 'Stellar Mass, where 1.0 is 1* our Sun\'s mass'],
         ['st_lum', 'Numeric', 'log(Solar luminosity)', 'Stellar Luminosity'],
         ['st_age', 'Numeric', 'gyr (Gigayear)', 'Stellar Age'],
         ['glat', 'Numeric', 'degrees', 'Galactic Latitude'],
         ['glon', 'Numeric', 'degrees', 'Galactic Longitude'],
         ['sy_dist', 'Numeric', 'parsec', 'Distance'],
         ['sy_plx', 'Numeric', 'mas (miliarcseconds)', 'Parallax: Distance the star moves in relation to other objects in the night sky'],
        ]

print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import io
import requests

pd.set_option('display.max_columns', None) 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic("matplotlib", " inline ")
get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")
plt.style.use("ggplot")

# For mac users
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


url = 'https://raw.githubusercontent.com/RafDingo/Maths40/main/data/exoPlanets.csv'
df_main = pd.read_csv(url, error_bad_lines=False)


print('Shape:', df_main.shape)
df_main.sample(10, random_state=999)


# Copy orignal data to a a working dataframe
df = df_main.copy()


# View all columns to find the ones required for regression
df.columns


# Drop id and irrelevant columns
del df['loc_rowid']
del df['pl_dens']
del df['pl_insol']
del df['ttv_flag']
del df['st_spectype']
del df['st_metratio']
del df['st_logg']
del df['st_rotp']
del df['sy_pmdec']
del df['sy_mnum']
# data check
df.columns


# Change to readable column names.
df.rename({
      'sy_snum': 'num_star',
      'sy_pnum': 'num_planet',
      'cb_flag': '2_stars',
      'pl_orbper': 'orbital_period',
      'pl_orbsmax': 'semi-major_axis',
      'pl_rade': 'planet_radius',
      'pl_bmasse': 'planet_mass',
      'pl_orbeccen': 'planet_eccen',
      'pl_eqt': 'planet_temp',
      'st_teff': 'star_temp',
      'st_rad': 'star_radius',
      'st_mass': 'star_mass',
      'st_lum': 'star_bright',
      'st_age': 'star_age',
      'glat': 'latitude_gal',
      'glon': 'longitude_gal',
      'sy_dist': 'distance',
      'sy_plx': 'parallax'
      }, 
      axis=1, inplace=True
)
# data check
df.head()


# Overview into data types and uniqueness
print('Unique rows =', df.shape[0], '| Unique columns =', df.shape[1])
print('-----')
print('Data types: ', df.dtypes)
print('-----')
print('Unique values per column: ', df.nunique())


# Find outliers of every column and store them into dictionary
# dict = {}
excluded_columns = [
                    'num_star',
                    'num_planet',
                    '2_stars',
]
for column_name in df.columns: 
    # conditional to exclude certain columns from outlier check
    if column_name in excluded_columns:
        continue
    else:
        column = df[column_name]
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        iqr = column.quantile(0.75) - column.quantile(0.25)

        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        num_column_outliers = df[(column > upper) | (column < lower)].shape[0]

        # Add to list as dict
        # dict[column] = [lower, upper, num_column_outliers]

        # drop rows that exceeds outlier parameters
        df = df[(column < upper) | (column > lower)]

df


# print the dictionary
# for key, value in dict.items():
#     print(f'''{key.upper()} | lower: {value[0]}, upper: {value[1]}, 
#     num of outliers: {value[2]}
#     ''')


# Overview of null values
df.isna().sum()


df = df.dropna()
df.isna().sum()



