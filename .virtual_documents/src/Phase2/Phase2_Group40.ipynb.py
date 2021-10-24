import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import patsy

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None) 

get_ipython().run_line_magic("matplotlib", " inline ")
get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")
plt.style.use("ggplot")

df = pd.read_csv('Phase2_Group40.csv')
print('Before Rename:', df.columns.to_list())

# Rename columns to be compatible with patsy
df.rename({'semi-major_axis': 'semi_major_axis', '2_stars': 'two_stars'}, axis=1, inplace=True)
print("After Rename:")

# Remove particular variables for better study
df = df.drop(['planet_mass', 'latitude_gal', 'longitude_gal', 'mass_ratio_sys', 'radius_ratio_sys'], axis=1 )
df.head()


# Generate a copy for data modification
data_encoded = df.copy()


categorical_vars = [ "num_star", "two_stars", "num_planet"]

for var in categorical_vars:
    data_encoded = data_encoded.astype({var: object})

# Categorical encoding for less than 2 values
for col in data_encoded.columns:
    q = len(data_encoded[col].unique())
    if (q == 2):
        data_encoded[col] = pd.get_dummies(data_encoded[col], drop_first=True)
# For categorical features > 2 levels
data_encoded = pd.get_dummies(data_encoded)

print(f"There are {data_encoded.shape[1]} columns with the column names {data_encoded.columns.to_list()} after one hot encoding")


"""
Due to the nature of our dataset, all uint8 types are considered categorical.
"""
# Perform normalisation on only the float types in df_float.
df_float = data_encoded.select_dtypes(include=['float64'])
df_float.drop('planet_radius', inplace=True, axis=1)
print(df_float.columns.to_list())
# TODO: Check if RobustScaler gives out better results
df_norm = MinMaxScaler().fit_transform(df_float)

print(f"The mean of each column in the df_norm dataframe is {np.round(df_norm.mean(axis=0),3)}")


data_encoded.loc[:, 
        ['orbital_period', 'semi_major_axis', 'planet_eccen',
         'planet_temp', 'star_temp', 'star_radius', 'star_mass', 'star_bright',
         'star_age', 'distance', 'parallax'
        ]
        ] = pd.DataFrame(df_norm, columns=[
                                            'orbital_period', 'semi_major_axis', 
                                             'planet_eccen',
                                            'planet_temp', 'star_temp',
                                            'star_radius', 'star_mass',
                                            'star_bright', 'star_age',
                                            'distance', 'parallax'
                                            ])
data_encoded.sample(3)


formula_string_indep_vars_encoded = ' + '.join(data_encoded.drop(columns='planet_radius').columns)
formula_string_encoded = 'planet_radius ~ ' + formula_string_indep_vars_encoded
print('formula_string_encoded: ', formula_string_encoded)


model_full = sm.formula.ols(formula=formula_string_encoded, data=data_encoded)

model_full_fitted = model_full.fit()

print(model_full_fitted.summary())


residuals_full = pd.DataFrame({'actual': df['planet_radius'], 
                            'predicted': model_full_fitted.fittedvalues, 
                            'residual': model_full_fitted.resid})

def plot_line(axis, slope, intercept, **kargs):
    xmin, xmax = axis.get_xlim()
    plt.plot([xmin, xmax], [xmin*slope+intercept, xmax*slope+intercept], **kargs)

plt.scatter(residuals_full['actual'], residuals_full['predicted'], alpha=0.3);
plot_line(axis=plt.gca(), slope=1, intercept=0, c="red");
plt.xlabel('Actual Radius');
plt.ylabel('Predicted Radius');
plt.title('Figure 9: Scatter plot of actual vs. predicted radius for the full Model', fontsize=15);
plt.show();


plt.scatter(residuals_full['predicted'], residuals_full['residual'], alpha=0.3);
plt.xlabel('Predicted Radius');
plt.ylabel('Residuals')
plt.title('Figure 10(a): Scatterplot of residuals vs. predicted Radius for Full Model', fontsize=15)
plt.show();


plt.scatter(residuals_full['actual'], residuals_full['residual'], alpha=0.3);
plt.xlabel('Actual Radius');
plt.ylabel('Residuals')
plt.title('Figure 10(b): Scatterplot of residuals vs. actual Radius for Full Model', fontsize=15)
plt.show();


plt.hist(residuals_full['actual'], label='actual', bins=20, alpha=0.7);
plt.hist(residuals_full['predicted'], label='predicted', bins=20, alpha=0.7);
plt.xlabel('Radius');
plt.ylabel('Frequency');
plt.title('Figure 11: Histograms of actual Radius vs. predicted Radius for Full Model', fontsize=15);
plt.legend()
plt.show();


plt.hist(residuals_full['residual'], bins = 20);
plt.xlabel('Residual');
plt.ylabel('Frequency');
plt.title('Figure 12: Histogram of residuals for Full Model', fontsize=15);
plt.show();


## create the patsy model description from formula
patsy_description = patsy.ModelDesc.from_formula(formula_string_encoded)

# initialize feature-selected fit to full model
linreg_fit = model_full_fitted

# do backwards elimination using p-values
p_val_cutoff = 0.05

## WARNING 1: The code below assumes that the Intercept term is present in the model.
## WARNING 2: It will work only with main effects and two-way interactions, if any.

print('\nPerforming backwards feature selection using p-values:')
to_remove = []
while True:

    # uncomment the line below if you would like to see the regression summary
    # in each step:
    # print(linreg_fit.summary())

    pval_series = linreg_fit.pvalues.drop(labels='Intercept')
    pval_series = pval_series.sort_values(ascending=False)
    term = pval_series.index[0]
    pval = pval_series[0]
    if (pval < p_val_cutoff):
        break
    term_components = term.split(':')
    print(f'\nRemoving term "{term}" with p-value {pval:.4}')
    to_remove.append(str(term))
    if (len(term_components) == 1): ## this is a main effect term
        patsy_description.rhs_termlist.remove(patsy.Term([patsy.EvalFactor(term_components[0])]))    
    else: ## this is an interaction term
        patsy_description.rhs_termlist.remove(patsy.Term([patsy.EvalFactor(term_components[0]), 
                                                        patsy.EvalFactor(term_components[1])]))    
        
    linreg_fit = smf.ols(formula=patsy_description, data=data_encoded).fit()
    
###
## this is the clean fit after backwards elimination
model_reduced_fitted = smf.ols(formula=patsy_description, data=data_encoded).fit()
print("To remove list:", to_remove, "\n")
###
    
#########
print("\n***")
print(model_reduced_fitted.summary())
print("***")
print(f"Regression number of terms: {len(model_reduced_fitted.model.exog_names)}")
print(f"Regression F-distribution p-value: {model_reduced_fitted.f_pvalue:.4f}")
print(f"Regression R-squared: {model_reduced_fitted.rsquared:.4f}")
print(f"Regression Adjusted R-squared: {model_reduced_fitted.rsquared_adj:.4f}")


residuals_reduced = pd.DataFrame({'actual': df['planet_radius'], 
                            'predicted': model_reduced_fitted.fittedvalues, 
                            'residual': model_reduced_fitted.resid})


# Creating scatter plot for reduced model
plt.scatter(residuals_reduced['actual'], residuals_reduced['predicted'], alpha=0.3);
plot_line(axis=plt.gca(), slope=1, intercept=0, c="red");
plt.xlabel('Actual Radius');
plt.ylabel('Predicted Radius');
plt.title('Figure 9: Scatter plot of actual vs. predicted radius for the Reduced Model', fontsize=15);
plt.show();


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


orig_data_dnn = df.copy()
categorical_vars = [ "num_star", "two_stars", "num_planet"]

for var in categorical_vars:
    orig_data_dnn = orig_data_dnn.astype({var: object})

# Categorical encoding for less than 2 values
for col in orig_data_dnn.columns:
    q = len(orig_data_dnn[col].unique())
    if (q == 2):
        orig_data_dnn[col] = pd.get_dummies(orig_data_dnn[col], drop_first=True)
# For categorical features > 2 levels
orig_data_dnn = pd.get_dummies(orig_data_dnn)

print(f"There are {orig_data_dnn.shape[1]} columns with the column names {orig_data_dnn.columns.to_list()} after one hot encoding")
orig_data_dnn.shape


# # Outlier filter
# def set_outlier_nan(df):
#     """
#     - Finds outliers and sets their values to NaN to be processed later.
#     - Excluded columns involves categories to be excluded from the outlier check
#     """
# #     excluded_columns = [
# #                         'num_star',
# #                         'num_planet',
# #                         'two_stars',
# #                         'longitude_gal',
# #                         'latitude_gal',
# #                         'parallax',
# #                         'distance',
# #     ]
#     for column_name in df.columns: 
#     # conditional to exclude certain columns from the outlier check
# #     if column_name in excluded_columns:
# #         continue
# #     else:
#         column = df[column_name]
#         q1 = column.quantile(0.25)
#         q3 = column.quantile(0.75)
#         iqr = column.quantile(0.75) - column.quantile(0.25)

#         lower = q1 - 3 * iqr
#         upper = q3 + 3 * iqr
#         num_column_outliers = df[(column > upper) | (column < lower)]\
#         .shape[0]
#         # set rows that exceeds outlier parameters to none
#         df[(column > upper) | (column < lower)] = np.nan

#     return df

# orig_data_dnn = set_outlier_nan(df=orig_data_dnn)
# print(
# f"""
# The outlier check will get rid of {orig_data_dnn["planet_radius"].isna().sum()} planets.
# """)
# orig_data_dnn = orig_data_dnn.dropna()
# print(f"The dataset now has {orig_data_dnn.shape[0]} planets")


data_dnn = orig_data_dnn.copy()
target_df = data_dnn['planet_radius'].values.reshape(-1, 1)
target_norm = MinMaxScaler().fit_transform(target_df)

data_dnn.loc[:, 
        ['planet_radius']
                ] = pd.DataFrame(target_norm, columns=['planet_radius'])

# Apply feature selection
for col_name in to_remove:
    data_dnn.drop(col_name, axis=1, inplace=True)
data_dnn.sample(3)


train_dataset = data_dnn.sample(frac=0.8, random_state=0)
test_dataset = data_dnn.drop(train_dataset.index)
print(
f"""
--- Dataset Sizes ---
Original Dataset: {data_encoded.shape}
Training Dataset: {train_dataset.shape}
Testing Dataset: {test_dataset.shape}
"""
)



train_dataset.describe().transpose()


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('planet_radius')
test_labels = test_features.pop('planet_radius')


train_dataset.describe().transpose()[['mean', 'std']]


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
#     plt.ylim([-100, 100])
    plt.title("Performance Analysis Before and after DNN Enhancement")
    plt.xlabel('Epoch')
    plt.ylabel('Error [Planet Radius]')
    plt.legend()
    plt.grid(True)





def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(24, activation='relu', name='layer1'),
      layers.Dense(24, activation='relu', name='layer2'),
      layers.Dense(24, activation='relu', name='layer3'),
      layers.Dense(1, name='output_layer')
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


get_ipython().run_cell_magic("time", "", """history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)""")


plot_loss(history)


import numpy as np
import pandas as pd
import io
import requests
import warnings
warnings.filterwarnings("ignore")

# so that we can see all the columns
pd.set_option('display.max_columns', None) 

# setup matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline ")
get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")
plt.style.use("ggplot")


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data_dnn = data_encoded.copy()
target_df = data_dnn['planet_radius'].values.reshape(-1, 1)
target = MinMaxScaler().fit_transform(target_df)

# Apply feature selection
for col_name in to_remove:
    data_dnn.drop(col_name, axis=1, inplace=True)
Data = data_dnn.drop(columns=['planet_radius'])
Data.sample(3)


D_train, D_test, t_train, t_test, idx_train, idx_test = \
   train_test_split(Data, target, Data.index, test_size=0.3, random_state=999)


# size of the network is determined by the number of neural units in each hidden layer
layer1_units = 64
layer2_units = 64


from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.optimizers import SGD, Adam


loss = 'binary_crossentropy' 
# during training, we would like to monitor accuracy
metrics = ['accuracy'] 


epochs = 500
# batch_size = 100


layer1_activation = 'relu'
layer2_activation = 'relu'
# output_activation = 'sigmoid'


layer1_dropout_rate = 0.05
layer2_dropout_rate = 0.00


learning_rate=0.01
decay=1e-6
momentum=0.5
# SGD stands for stochastic gradient descent
optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum)
# optimizer = Adam(0.001)


# set up an empty deep learning model
def model_factory(input_dim, layer1_units, layer2_units):
    model = Sequential()
    
    model.add(Dense(layer1_units, input_dim=input_dim, activation=layer1_activation))
    model.add(Dropout(layer1_dropout_rate))
    
    model.add(Dense(layer2_units, activation=layer2_activation))
    model.add(Dropout(layer2_dropout_rate))
    
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


# define plot function for the fit
# we will plot the accuracy here
def plot_history(history): 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()


model_test = model_factory(Data.shape[1], layer1_units, layer2_units)

# in the summary, notice the LARGE number of total parameters in the model
model_test.summary()


get_ipython().run_cell_magic("time", "", """history_test = model_test.fit(D_train, 
                              t_train,
                              epochs=epochs,
#                               batch_size=batch_size,
                              verbose=0, # set to 1 for iteration details, 0 for no details
                              shuffle=True,
                              validation_data=(D_test, t_test))""")


# here are the keys in the history attribute of the fitted model object
history_test.history.keys()


plot_history(history_test)




# plt.hist(residuals_full['actual'], label='actual', bins=20, alpha=0.7)
# plt.hist(residuals_full['predicted'], label='predicted', bins=20, alpha=0.7)
# plt.xlabel('Radius')
# plt.ylabel('Frequency')
# plt.title('Figure 11: Histograms of actual Radius vs. predicted Radius for Full Model', fontsize=15)
# plt.legend()
residuals_full

standard_linear = residuals_reduced['predicted']
standard_linear.std()
history_test.history['accuracy'].std()


# compute prediction performance on test data
model_output = model_test.predict(D_test).astype(float)

# decide classification based on threshold of 0.5
t_pred = np.where(model_output < 0.5, 0, 1)

# set up the results data frame
result_test = pd.DataFrame()
result_test['target'] = t_test.flatten()
result_test['fit'] = t_pred
# residuals will be relevant for regression problems
# result_test['abs_residual'] = np.abs(result_test['target'] - result_test['fit'])
result_test.head()


acc = accuracy_score(result_test['target'], result_test['fit'])
auc = roc_auc_score(result_test['target'], result_test['fit'])
print(f"validation data accuracy_score = {acc:.3f}")
print(f"validation data roc_auc_score = {auc:.3f}")


get_ipython().getoutput("pip install -q seaborn")


from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# NEURON PROPERTIES
layer1_units = 64
layer2_units = 64
layer1_activation = 'relu'
layer2_activation = 'relu'
# output_activation = None  # Use linear type
layer1_dropout_rate = 0.05
layer2_dropout_rate = 0.00

# NETWORK PROPERTIES
loss = 'binary_crossentropy' # 'mean_absolute_error'?
# during training, we would like to monitor accuracy
metrics = ['accuracy']
epochs = 100
batch_size = 100

# OPTIMIZER SETTINGS 
# (SGD)
learning_rate=0.01
decay=1e-6
momentum=0.5
optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum)
# (Adam) ?
# optimizer = Adam(0.001)


dl_data = df.copy()

train_data = dl_data.sample(frac=0.8, random_state=0)
test_data = dl_data.drop(train_data.index)


train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('planet_radius')
test_labels = test_features.pop('planet_radius')


# set up an empty deep learning model
def model_factory(input_dim, layer1_units, layer2_units):
    model = Sequential()
    model.add(Dense(layer1_units, input_dim=input_dim, activation=layer1_activation))
    model.add(Dropout(layer1_dropout_rate))
    
    model.add(Dense(layer2_units, activation=layer1_activation))
    model.add(Dropout(layer2_dropout_rate))
    
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


# define plot function for the fit
# we will plot the accuracy here
def plot_history(history): 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()


model_test = model_factory(dl_data.shape[1], layer1_units, layer2_units)

# in the summary, notice the LARGE number of total parameters in the model
model_test.summary()


get_ipython().run_cell_magic("time", "", """history_test = model_test.fit(train_features, 
                              train_labels,
                              epochs=100,
#                               batch_size=batch_size,
                              validation_split=0.2,
                              verbose=0, # set to 1 for iteration details, 0 for no details
#                               shuffle=True,
#                               validation_data=(train_features, train_labels)
                             )""")



