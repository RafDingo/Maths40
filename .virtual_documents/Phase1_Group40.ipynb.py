import warnings
warnings.filterwarnings("ignore")

from tabulate import tabulate

table = [['Name','Data Type','Units','Description'],
         ['sy_snum', 'Nominal categorical', 'NA', 'Number of stars in the system'],
         ['sy_pnum', 'Nominal categorical', 'NA', 'Number of planets in the system'],
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

print(tabulate(table, headers='firstrow', tablefmt='simple'))


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
sns.set(rc={"figure.dpi":150, 'savefig.dpi':300})

# For mac users
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


df_main = pd.read_csv("Phase1_Group40.csv", error_bad_lines=False, index_col=False)


print('Shape:', df_main.shape)
df_main.sample(10, random_state=999)


# Copy orignal data to a a working dataframe
df = df_main.copy()


# View all columns to find ones required for regression
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


EARTH_MASS = 5.972 * (10**24)
SUN_MASS = 1.989 * (10**30)
EARTH_RADIUS = 6371 
SUN_RADIUS = 696340 
# mass ratio in kgs
df['mass_ratio_sys'] = (df['planet_mass'] * EARTH_MASS) / (df['star_mass'] * SUN_MASS)
# radius in km

df['radius_ratio_sys'] = (df['planet_radius'] * EARTH_RADIUS) / (df['star_radius'] * SUN_RADIUS)
# data check
df[['mass_ratio_sys', 'radius_ratio_sys']].sample(15)


# Overview into data types and uniqueness
print('Unique rows =', df.shape[0], '| Unique columns =', df.shape[1])
print('-----')
print('Data types: ', df.dtypes)
print('-----')
print('Unique values per column: ', df.nunique())


# def set_outlier_nan(df):
#     """
#     - Finds outliers and sets their values to NaN to be processed later.
#     - Excluded columns involves categories to be excluded from the outlier check
#     """
#     excluded_columns = [
#                         'num_star',
#                         'num_planet',
#                         '2_stars',
#                         'longitude_gal',
#                         'latitude_gal',
#                         'parallax',
#                         'distance',
#     ]
#     for column_name in df.columns: 
#         # conditional to exclude certain columns from the outlier check
#         if column_name in excluded_columns:
#             continue
#         else:
#             column = df[column_name]
#             q1 = column.quantile(0.25)
#             q3 = column.quantile(0.75)
#             iqr = column.quantile(0.75) - column.quantile(0.25)

#             lower = q1 - 3 * iqr
#             upper = q3 + 3 * iqr
#             num_column_outliers = df[(column > upper) | (column < lower)]\
#             .shape[0]
#             # set rows that exceeds outlier parameters to none
#             df[(column > upper) | (column < lower)] = np.nan

#     return df

# df = set_outlier_nan(df=df)
# df.isna().sum()
# df.sample(25)


def drop_nan(df):
    df = df.dropna()

    nan_values = False
    for column in df.isna().sum():
        if not column == 0:
            nan_values = True
            break
    return df, nan_values

# Overview of null values
print('Before: \n', df.isna().sum())

df, nan_values = drop_nan(df=df)

print(f'''
After:
NaN Values?, {nan_values}

Shape: {df.shape}''')


temp = df['num_planet']
del df['num_planet']
df['num_planet'] = temp
df.head(5)


# number of stars 
sns.distplot(a=df['num_star'], bins=4, kde=False)
plt.xlabel('Number of Stars\n in the same system')
plt.ylabel('Popularity')
plt.xticks([1, 2, 3, 4])


sns.distplot(x=df['num_planet'], bins=8, kde=False)
plt.xlabel('Number of Planets\n within the same system')
plt.ylabel('Popularity')


# Randomisation of colours
from random import choice
colors = [
            'b', 'g', 'r', 
            'c', 'm', 'y',
            'b',
]



sns.distplot(x=df['planet_radius'], color=choice(colors))
plt.title(f'Planets discovered over the planet radius property')
plt.xlabel('Planet Radius (per Earth radius)')
plt.xlim(0, None)
plt.show()


sns.distplot(x=df['planet_temp'], color=choice(colors))
plt.title(f'Planets discovered over the planet temperature property')
plt.xlabel('Planet Temperature (K)')
plt.xlim(0, None)

plt.show()


sns.distplot(x=df['star_temp'], color=choice(colors))
plt.title(f'Planets discovered over the Star Temperature property')
plt.xlabel('Star Temperature (K)')
plt.xlim(0, None)
plt.show()


sns.distplot(x=df['star_radius'], color=choice(colors))
plt.title(f'Planets discovered over the star radius property')
plt.xlabel('Star Radius (per Sol radius)')
plt.xlim(0, None)
plt.show()


sns.distplot(x=df['star_bright'], color=choice(colors))
plt.title(f'Planets discovered over the Star Brightness property')
plt.xlabel('Star Brightness\n (Log of Solar Luminosity)')
plt.show()


sns.distplot(x=df['star_age'], color=choice(colors))
plt.title(f'Planets discovered over the Star Age property')
plt.xlabel('Star Age (GigaYears)')
plt.xlim(0, None)
plt.show()


sns.distplot(x=df['distance'], color=choice(colors))
plt.title(f'Planets discovered over the distance property')
plt.xlabel('Distance (Parsecs)')
plt.xlim(0, None)
plt.show()


sns.barplot(y='num_planet', x='num_star', data=df)
plt.title('Number of Discovered Planets\n vs Number of Stars\n (belonging to the same star system)')
plt.xlabel('Number of Star(s)')
plt.ylabel('Number of Discovered Planet(s)')
plt.show


plt.xscale('log')
plt.yscale('log')
sns.scatterplot(data=df, x="orbital_period", y="semi-major_axis", alpha=1, hue="planet_radius")
plt.ylabel('Orbital Radius (AU)')
plt.xlabel('Orbital Period\n (per Earth days)')


plt.xscale('log')
plt.yscale('log')
sns.histplot(data=df, x="planet_mass", y="orbital_period", color='r')
plt.title('Orbital Period vs.\n Planetary Mass (fig. 1)')
plt.xlabel('Planetary Mass\n (per 1 Earth)')
plt.ylabel('Orbital Period (Earth days)')
plt.show()
print('========')
plt.xscale('log')
plt.yscale('log')
sns.histplot(data=df, x="planet_mass", y="semi-major_axis", color='g')
plt.title('Orbital Distance vs.\n Planetary Mass (fig. 2)')
plt.xlabel('Planetary Mass\n (per 1 Earth)')
plt.ylabel('Orbital Distance (AU)')
plt.show()


plt.yscale("log")
sns.scatterplot(y=df["planet_mass"], x=df["planet_radius"], alpha=0.2, color='g')
plt.title('Mass vs. Radius\n of\n Discovered Planets')
plt.ylabel('Planetary Mass\n (per Earth mass)')
plt.xlabel('Planetary Radius (per Earth radius)')


plot = sns.histplot(data=df, x="distance", hue="num_planet", multiple="fill", bins=10, binrange=(0, 3000), palette="rocket", legend="full")
plot.set(xlabel="Distance from Sol (Parsec(s))", ylabel="Proportion of planet count", title="Planet count per system over Distance")
plot.legend(title="Planet Count", loc="upper right", labels=[8,7,6,5,4,3,2,1])


plot = sns.scatterplot(data=df, x="distance", y="planet_radius")
plot.set(xlabel="Distance from Sol (Parsec(s))", ylabel="Planet Radius", title="Planet radius over Distance", xlim=(0, None))
plot


plot = sns.scatterplot(data=df, x="distance", y="planet_radius")
plot.set(xlabel="Distance from Sol (Parsec(s))", ylabel="Planet Radius", title="Planet radius over Distance, lower bounds", xlim=(0, None), ylim=(0,2))
plot


plot = sns.scatterplot(x=df["distance"], y=df["parallax"], hue=df["num_planet"], size=df["num_planet"], legend="full", alpha=0.7)
plot.set(xlim = (-10, 1000), ylim=(0, 200), xlabel="Distance from Sol", ylabel="Parallax amount")
plt.title('Distance vs. Parallax relationship')
plt.xlabel('Distance (Parsec(s))')
plt.ylabel('Parallax (miliarcseconds)')
plt.show


plot = sns.scatterplot(data=df, x="longitude_gal", y="latitude_gal", size="num_planet", palette='plasma', legend=False, alpha=0.2, color="black")
plot.set(xlabel="Galactic Longitude (deg)", ylabel="Galactic Latitude (deg)", title = "Exo-planet location projection\n in relation to the number of planets\n in its system")
plt.show()


plot = sns.scatterplot(data=df, x="longitude_gal", y="latitude_gal", size="num_planet", palette='plasma', legend=False, alpha=0.2, color="black")
plot.set(xlabel="Galactic Longitude (deg)", ylabel="Galactic Latitude (deg)", title = "Exo-planet locations from Keplar mission", xlim=(67.5, 85), ylim=(5, 22.5))

plt.show()


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='polar')
ax.scatter(x=df["longitude_gal"], y=df["distance"], sizes=30.488*0.4*(df["planet_radius"]), c="black", alpha=0.1)
ax.set(ylim=(0, 5000))
ax.axes.yaxis.set_visible(False)
ax.scatter(x=0, y=0, s=0.4, color='white')
plt.title('Exo-planet position, and relative size, up to 5000 parsecs')
plt.xlabel('Center white dot represents earth.\n Exo-planet sizes to scale')


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='polar')
ax.scatter(x=df["longitude_gal"], y=df["distance"], sizes=30.488*0.4*(df["planet_radius"]), c="black", alpha=0.1)
ax.set(ylim=(0, 2000))
ax.axes.yaxis.set_visible(False)
ax.scatter(x=0, y=0, s=0.4, color='white')
plt.title('Exo-planet position, and relative size, up to 2000 parsecs')
plt.xlabel('Center white dot represents earth.\n Exo-planet sizes to scale')


df.to_csv('Phase2_Group40.csv', index=False)



