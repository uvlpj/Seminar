#%%
import numpy as np
import pandas as pd
import sdv
import matplotlib.pyplot as plt
import seaborn as sns
import random
seed_value = 42
#%%
data_por = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Seminar/Daten/student/student-por.csv', sep = ';')

#%%
# Histogram --
# Plotting the Distribution of the Target variable
# Student End of year grade G3
plt.figure(figsize=(10, 6))
sns.histplot(data_por['G3'], kde=True, bins=range(0, 21), discrete=True)
plt.xlabel('Final grade G3', fontsize = 16)
plt.ylabel('Frequency', fontsize = 16)
plt.title('Distribution Portuguese language final grade G3', fontsize = 18)
plt.xticks(range(0, 20), fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
#%%
data_por_binary = data_por.copy()
# Pass if G3 ≥ 10, else Failure
data_por_binary ['Results_G3'] = data_por_binary ['G3'].apply(lambda x: 'pass' if x >=10 else 'fail')
# %%
# Creating Results into binary values
mapping = {'pass': 1, 'fail': 0}
data_por_binary['Results_G3'] = data_por_binary['Results_G3'].map(mapping)

#%%
plt.figure(figsize=(10, 6))
plt.hist(data_por_binary['Results_G3'], bins=2, edgecolor='black', alpha=0.7, rwidth=0.7)
plt.xticks([0.25, 0.75], ['fail', 'pass'], fontsize = 14)
plt.xlabel('G3', fontsize = 16)
plt.ylabel('Frequency', fontsize = 16)
plt.title('Portuguese pass/fail grade G3', fontsize = 18)
plt.show()

#%%
# Counting the number of target observations 
counts_target = data_por['G3'].value_counts().sort_index()
print(counts_target)

#%%
# Calculating the mean and the mode of the target variable G3 of the final portuguese language grade G3
mean_G3 = data_por['G3'].mean()
print(f"Mean of G3: {mean_G3}")

mode_G3 = data_por['G3'].mode()
print(f"Mode of G3:")
print(mode_G3)
# %%
# Gaussin Copula Synthesizer ---
from sdv.single_table import GaussianCopulaSynthesizer
# %%
# CTGAN Synthetsizer ---
from sdv.single_table import CTGANSynthesizer
#%%
from sdv.metadata import SingleTableMetadata



#%%
metadata_porr = SingleTableMetadata()


metadata_porr.detect_from_dataframe(data_por)

synthesizer_g = GaussianCopulaSynthesizer(metadata_porr)

my_constraint = {
    'constraint_class': 'ScalarInequality',
    'constraint_parameters': {
        'column_name': 'G1',
        'relation': '>=',
        'value': 0
    }
}

synthesizer_g.add_constraints(
    constraints = [my_constraint]
)

synthesizer_g.fit(data_por)


#%%
# Creating synthesised Data from the original unproccessed math data with 33 columns
metadata_por = SingleTableMetadata()
metadata_por.detect_from_dataframe(data_por)
# %%
synthesizer_gaussian = GaussianCopulaSynthesizer(metadata_por)
synthesizer_ctgan = CTGANSynthesizer(metadata_por)
# %%
synthesizer_gaussian.fit(data_por)
synthesizer_ctgan.fit(data_por)
# %%
# Gaussian Coupla Data
# creating 1000 fake data points
synthetic_por_gaussian = synthesizer_g.sample(num_rows=1000) 
#%%
from sdv.evaluation.single_table import evaluate_quality

# 1000 data points created with gaussain coupla 
quality_report_g1 = evaluate_quality(
    data_por,
    synthetic_por_gaussian,
    metadata_porr
)
# 93.92%

#%%
quality_report_g1.get_details(property_name='Column Shapes')

#%%
data_processed_por = data_por.copy()
# %%
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'guardian', 'Mjob', 'Fjob', 'reason']
data_processed_por = pd.get_dummies(data_por, columns=categorical_columns, dtype=int)
#%%
binary_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
binary_mapping = {'no': 0, 'yes': 1}

for column in binary_columns:
    data_processed_por[column] = data_processed_por[column].map(binary_mapping)

#%%
from sklearn.preprocessing import LabelEncoder

ordinal_columns = ['traveltime', 'Medu', 'Fedu', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']

label_encoder = LabelEncoder()

for column in ordinal_columns:
    data_processed_por[column] = label_encoder.fit_transform(data_processed_por[column])


#%%
# Correlation matrx
correlation_matrix = data_processed_por.corr()
print(correlation_matrix["G3"].sort_values(ascending=False))
#%%
# Correlation matrix 
# ==============================

correlation_matrix = data_processed_por.corr()

# Definiere die interessanten Features, inklusive 'Results_G3'
#interesting_features = ['Results_G3', 'G1', 'G2', 'higher', 'failures', 'goout', 'age']
interesting_features = ['G3', 'G1', 'G2', 'higher', 'failures', 'studytime']

# Extrahiere die Korrelationsmatrix nur für die interessanten Features
correlation_subset = correlation_matrix.loc[interesting_features, interesting_features]

# Plotten der Korrelationsmatrix als Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlationmatrix Portuguese dataset')
plt.show()

#%%
'''
___________________________________________________________________________________
Korrelationen ---
Die 10 stärksten Korrelationen mit der Zielvariable 
-----------------------------------------------------------------------------------
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Funktion zur Berechnung der Korrelationen und p-Werte
def calculate_correlations(data, target_variable):
    correlations = {}
    
    for feature in data.columns:
        if feature != target_variable:
            corr, p_value = pearsonr(data[target_variable], data[feature])
            correlations[feature] = (corr, p_value)
    
    return correlations

# Korrelationen und p-Werte für 'G3'
correlations = calculate_correlations(data_processed_por, 'G3')

# Sortiere nach dem absoluten Wert der Korrelationen und wähle die 10 stärksten
sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1][0]), reverse=True)
top_10_correlations = dict(sorted_correlations[:10])

# Zeige die 10 stärksten Korrelationen und p-Werte
print("Die 10 stärksten Korrelationen mit G3:")
for feature, (corr, p_value) in top_10_correlations.items():
    print(f"{feature}: Korrelation = {corr:.3f}, p-Wert = {p_value:.3f}")

# Visualisierung der 10 stärksten Korrelationen
interesting_features = ['G3'] + [feature for feature in top_10_correlations.keys()]
correlation_subset = data_processed_por[interesting_features].corr()

plt.figure(figsize=(13, 11))

# Plotten der Korrelationsmatrix als Heatmap
plt.title('Top 10 Strongest Correlations with G3 (Portuguese dataset)', fontsize=18)
plt.xticks(fontsize=14, rotation = 90)
plt.yticks(fontsize=14)
plt.show()










#%%
'''
----------------------------
Synthetische Daten erstellen
    Daten in Train, Test, Validation teilen
    Dann um Data leakage zu vermeiden die synthetischen Daten von den Trainingsdaten erstellen
---------------------------    
'''
from sklearn.model_selection import train_test_split
#%%
df_processed_por = data_processed_por.copy()
#%%
X = df_processed_por.drop(columns=['G3'])
y = df_processed_por['G3']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%

train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
print(train_data.shape)

test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
print(test_data.shape)

#%%
from sdv.metadata import SingleTableMetadata
#%%
from sdv.single_table import GaussianCopulaSynthesizer
#%%
metadata_por = SingleTableMetadata()
#%%
metadata_por.detect_from_dataframe(train_data)
#%%
synthesizer_gaussian = GaussianCopulaSynthesizer(
    metadata_por)

synthesizer_gaussian.fit(train_data)

#%%
# Gaussian Coupla Data
synthetic_data_gaussian_1 = synthesizer_gaussian.sample(num_rows=1000) # creating 1000 fake data points
#%%
# Evaluating the synthetic data 
# comparing the synthetic data with the real data (train_data)
from sdv.evaluation.single_table import evaluate_quality,run_diagnostic

# 1000 data points created with gaussain coupla 
quality_report_g1 = evaluate_quality(
    train_data,
    synthetic_data_gaussian_1,
    metadata_por
)
# Quality Report: 93.12%
#%%
print(synthetic_data_gaussian_1.shape)
print(synthetic_data_gaussian_1.head(10))



#%%
# Plotten der Verteilngen echte und synthetische Daten

if 'G3' in train_data.columns and 'G3' in synthetic_data_gaussian_1.columns:
    # Plotten der Verteilung von 'G1'
    plt.figure(figsize=(12, 6))

    # Plot für die echte Daten
    sns.histplot(train_data['G3'], kde=True, color='blue', label='original data', bins=range(0, 21), discrete=True)

    # Plot für die synthetischen Daten
    sns.histplot(synthetic_data_gaussian_1['G3'], kde=True, color='red', label='synthetic data', bins=range(0, 21), discrete=True)

    plt.title('Comparison density of G3', fontsize = 18)
    plt.xlabel('G3', fontsize = 16)
    plt.xticks(range(0, 20), fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylabel('Density', fontsize = 14)
    plt.legend()
    plt.show()
else:
    print("Die Spalte 'G3' ist in einem der DataFrames nicht vorhanden.")













#%%
'''
-----------------------------------------------------------------------------------------
Regression Model
including G1 und G2
------------------------
'''
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(X_train.shape)
print(X_train.columns)

#%%
predictor_scaler = StandardScaler()
target_scaler = StandardScaler()

# Fit the scaler on X_train and transform X_train
X_train_scaled = predictor_scaler.fit_transform(X_train)
# Transform X_test using the same scaler fit on X_train
X_test_scaled = predictor_scaler.transform(X_test)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# Fit the scaler on y_train and transform y_train
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
# Transform y_test using the same scaler fit on y_train
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# Add a constant term (intercept) to the scaled X_train and X_test
X_train_scaled = sm.add_constant(X_train_scaled, prepend=True)
X_test_scaled = sm.add_constant(X_test_scaled, prepend=True)

# Checking the shapes after scaling and adding the constant
print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(y_train_scaled.shape)
print(y_test_scaled.shape)

#%%
#%%
linear_reg_included = LinearRegression()
#%%
linear_reg_included.fit(X_train_scaled, y_train_scaled)
#%%
y_pred_included = linear_reg_included.predict(X_test_scaled)
#%%
# MSE ---
# Mean Squared Error
mse_lin_included = mean_squared_error(y_test_scaled, y_pred_included)
print(f"MSE Linear Regression including G1 and G2: {mse_lin_included}")
# MSE: 0.907
#%%
# MAE ---
# Mean Absolute Error
mae_lin_included = mean_absolute_error(y_test_scaled, y_pred_included)
print("MAE Linear Regression including G1 and G2: ", mae_lin_included)
# MAE: 0.737
#%%
# Adjusted R-squared ---
r2_lin_included = r2_score(y_test_scaled, y_pred_included)
print("R2: ", r2_lin_included)

n = len(y_test_scaled)  # Number of observations
p = 50  # Number of predictors
adjusted_r2_lin_included = 1 - (1 - r2_lin_included) * (n - 1) / (n - p - 1)
print("Adjusted R2 included G1 and G2: ", adjusted_r2_lin_included)
# Adjusted R-squared: 0.8237

#%%
'''
---------------------------------------------------------------------
Linear Regression
    without G1 and G2
---------------------------------------------------------------------
'''

print("Original shapes:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Copy the data to new variables for linear regression
X_train_lin = X_train.copy()
X_test_lin = X_test.copy()
y_train_lin = y_train.copy()
y_test_lin = y_test.copy()

#%%
y_train_lin = pd.DataFrame(y_train_lin)
y_test_lin = pd.DataFrame(y_test_lin)
X_train_lin = pd.DataFrame(X_train_lin)
X_test_lin = pd.DataFrame(X_test_lin)
#%%

# Drop columns 'G1' and 'G2' from the predictors
X_train_lin = X_train_lin.drop(columns=['G1', 'G2'])
X_test_lin = X_test_lin.drop(columns=['G1', 'G2'])
#%%
# We need to add a constant
X_train_lin = sm.add_constant(X_train_lin)
X_test_lin = sm.add_constant(X_test_lin)
#%%
print(X_train_lin.shape)
print(X_test_lin.shape)
#%%
print(X_train_lin.columns)
#%%
# Initialize scalers
predictor_scaler_lin = StandardScaler()
#target_scaler_lin = StandardScaler()

# Scale the predictors
X_train_lin_scaled = predictor_scaler_lin.fit_transform(X_train_lin)
X_test_lin_scaled = predictor_scaler_lin.transform(X_test_lin)

# Scale the target variable
#y_train_lin_scaled = target_scaler_lin.fit_transform(y_train_lin.values.reshape(-1, 1))
#y_test_lin_scaled = target_scaler_lin.transform(y_test_lin.values.reshape(-1, 1))

# Print shapes after dropping columns and scaling
print("\nShapes after dropping 'G1' and 'G2' and scaling:")
print(X_train_lin_scaled.shape)
print(X_test_lin_scaled.shape)
print(y_train_lin.shape)
print(y_test_lin.shape)

#%%
linear_reg = LinearRegression()
#%%
linear_reg.fit(X_train_lin_scaled, y_train_lin)
#%%
y_pred_lin_scaled = linear_reg.predict(X_test_lin_scaled)
#%%
# MSE ---
# Mean Squared Error
mse_lin = mean_squared_error(y_test_lin, y_pred_lin_scaled)
print(f"MSE Linear Regression excluding G1 and G2: {mse_lin}")
# MSE: 8.978
#%%
# MAE ---
# Mean Absolute Error
mae_lin = mean_absolute_error(y_test_lin, y_pred_lin_scaled)
print("MAE Linear Regression excluded G1 and G2: ", mae_lin)
# MAE: 2.250

#%%
from sklearn.metrics import r2_score

r_squared_1 =  r2_score(y_test_lin, y_pred_lin_scaled)

print(r_squared_1)  #0.1880

#%%
residuals_1 = y_test_lin - y_pred_lin_scaled

#%%
#%%
SSR = np.sum(residuals_1 ** 2)

print(f"Sum of Squared Residuals (SSR): {SSR}")

#%%
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lin_scaled, residuals_1)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot (model without G1 and G2)', fontsize=16)
plt.xlabel('Predicted G3', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.ylim(-11, 11)
plt.xticks(np.arange(0, 21, step=2), fontsize=12)
plt.yticks(np.arange(-10, 11, step=2), fontsize=12) 
plt.show()

#%%
'''
----------------------------
Regressing G1 and G2 on G3
----------------------------
'''

X_train1 = X_train.copy()
X_test1 = X_test.copy()
y_train1 = y_train.copy()
y_test1 = y_test.copy()
#%%
X_train1 = sm.add_constant(X_train1)
X_test1 = sm.add_constant(X_test1)
#%%
X_train_1 = X_train1[['const','G1', 'G2']]
X_test_1 = X_test1[['const','G1', 'G2']]
#%%
print(X_train_1.shape)
print(y_train1.shape)

print(X_test_1.shape)
print(y_test1.shape)

#%%
predictor_scaler_lin = StandardScaler()


# Scale the predictors
X_train_1_scaled = predictor_scaler_lin.fit_transform(X_train_1)
X_test_1_scaled = predictor_scaler_lin.transform(X_test_1)

#%%
model_G1G2 = LinearRegression()
model_G1G2.fit(X_train_1_scaled, y_train1)
#%%
y_pred_G1G2 = model_G1G2.predict(X_test_1_scaled)

#%%
residuals = y_test1 - y_pred_G1G2

#%%
# MSE ---
mse = mean_squared_error(y_test1, y_pred_G1G2)
print(f'Mean Squared Error: {mse}') #1.316

#%%
# MAE ---
# Mean Absolute Error
mae_lin = mean_absolute_error(y_test1, y_pred_G1G2)
print("MAE Linear Regression excluded G1 and G2: ", mae_lin)
# MAE: 0.734

#%%
from sklearn.metrics import r2_score

r_squared_G1G2 =  r2_score(y_test1, y_pred_G1G2)

print(r_squared_G1G2)  #0.8809
#%%
SSR = np.sum(residuals ** 2)

print(f"Sum of Squared Residuals (SSR): {SSR}")

#%%
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_G1G2, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot ', fontsize=16)
plt.xlabel('Predicted G3', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.ylim(-11, 11)
plt.xticks(np.arange(0, 21, step=2), fontsize=12)
plt.yticks(np.arange(-10, 11, step=2), fontsize=12) 
plt.show()

#%%


#%%




#%%
# Adjusted R-squared ---
r2_lin = r2_score(y_test_lin_scaled, y_pred_lin_scaled)
print("R2: ", r2_lin)

n = len(y_test_lin)  # Number of observations
p = 50  # Number of predictors
adjusted_r2_lin = 1 - (1 - r2_lin) * (n - 1) / (n - p - 1)
print("Adjusted R2 excluded G1 and G2: ", adjusted_r2_lin)
# Adjusted R-squared: 0.8237


#%%
'''
----------------------------------------------
Linear Model (synthetic)
    trained on synthetic data and original data
     without G1 and G2
----------------------------------------------
'''
y_target = synthetic_data_gaussian_1['G3']
X_features = synthetic_data_gaussian_1.drop(columns=['G3'])

#%%
y_target = pd.DataFrame(y_target)
y_train = pd.DataFrame(y_train)
#%%
# Adding a constant to the synthetic feature matrix
X_features = sm.add_constant(X_features, prepend = True)

#%%
X_train_lin = X_train.copy()
X_train_lin = sm.add_constant(X_train_lin, prepend = True) 
#%%
y_train_lin3 = pd.concat([y_train, y_target], axis=0).reset_index(drop=True)
X_train_lin3 = pd.concat([X_train_lin, X_features], axis=0).reset_index(drop=True)

#%%
X_train_lin3 = X_train_lin3.drop(columns=['G1', 'G2'])
X_test = X_test.drop(columns=['G1', 'G2'])
#%%
X_test = sm.add_constant(X_test)
#%%
print("X_train shape:", X_train_lin3.shape)
print("y_train shape:", y_train_lin3.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

#%%
# Using Standard Scaler
predictor_scaler = StandardScaler()
#target_scaler = StandardScaler()

predictor_scaler = StandardScaler()
#target_scaler = StandardScaler()

# Fit the scaler on X_train and transform X_train
X_train_scaled = predictor_scaler.fit_transform(X_train_lin3)
# Transform X_test using the same scaler fit on X_train
X_test_scaled = predictor_scaler.transform(X_test)

y_train = pd.DataFrame(y_train_lin3)
y_test = pd.DataFrame(y_test)

# Fit the scaler on y_train and transform y_train
#y_train_scaled = target_scaler.fit_transform(y_train_lin3.values.reshape(-1, 1))
# Transform y_test using the same scaler fit on y_train
#y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))


# Checking the shapes after scaling and adding the constant
print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(y_train.shape)
print(y_test.shape)
#%%
linear_reg_syn = LinearRegression()
#%%
linear_reg_syn.fit(X_train_scaled, y_train)
#%%
y_pred_lin_syn = linear_reg_syn.predict(X_test_scaled)
#%%
# MSE ---
# Mean Squared Error
mse_lin_syn = mean_squared_error(y_test, y_pred_lin_syn)
print(f"MSE Linear Regression synthetic data: {mse_lin_syn}")
# MSE:  8.725
#%%
# MAE ---
# Mean Absolute Error
mae_lin_syn = mean_absolute_error(y_test, y_pred_lin_syn)
print("MAE Linear Regression synthteic data: ", mae_lin_syn)
# MAE: 2.166

#%%
# R-squared:
r_squared_syn =  r2_score(y_test, y_pred_lin_syn)

print(r_squared_syn)  #0.2109

#%%
# Adjusted R-squared ---
r2_lin_syn = r2_score(y_test, y_pred_lin_syn)
print("R2: ", r2_lin_syn)

n = len(y_test)  # Number of observations
p = 50  # Number of predictors
adjusted_r2_lin_syn = 1 - (1 - r2_lin_syn) * (n - 1) / (n - p - 1)
print("Adjusted R2 excluded G1 and G2: ", adjusted_r2_lin_syn)
# Adjusted R-squared: 



'''
---------------
Neuronale Netz
einfache Architektur
-------------
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
seed_value = 42

# Set the seed for TensorFlow
tf.random.set_seed(seed_value)

# Set the seed for NumPy
np.random.seed(seed_value)

# Set the seed for Python's random module
random.seed(seed_value)


#learning_rates = [0.005, 0.004, 0.003, 0.0002 ,0.0001, 0.0002]
learning_rates = [0.00001, 0.0001, 0.001, 0.01]

#%%
# Prepare the data
df_processed_por = data_processed_por.copy()
X = df_processed_por.drop(columns=['G3', 'G1', 'G2'])
y = df_processed_por[['G3']]

PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

X = PredictorScaler.fit_transform(X)
y = TargetVarScaler.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#%%
# Initialize lists to store histories and final validation loss
histories = []
final_val_losses = []
stopped_epochs = []

# Loop over each learning rate
for lr in learning_rates:
    print(f"Training model with learning rate = {lr}")
    
    # Define the model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        #Dense(X_train.shape[1], activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # Compile the model with the given learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=32,
                        epochs=100,  # Set a large number of epochs to allow early stopping
                        callbacks=[early_stopping],
                        verbose=1)  # Set verbose to 1 if you want to see the progress
    
    # Store the history, the final validation loss, and the epoch at which training stopped
    histories.append((lr, history))
    final_val_losses.append(history.history['val_loss'][-1])
    stopped_epochs.append(early_stopping.stopped_epoch + 1)  # +1 because epoch count starts at 0

# Determine the best learning rate based on the lowest final validation loss
best_lr_idx = np.argmin(final_val_losses)
best_lr, best_history = histories[best_lr_idx]

print(f"Best learning rate: {best_lr}")
print(f"Final Validation Loss for best learning rate: {final_val_losses[best_lr_idx]}")
print(f"Training stopped after {stopped_epochs[best_lr_idx]} epochs for the best learning rate.")

# Plot the Training and Validation Loss for the best learning rate
plt.figure(figsize=(14, 7))
plt.plot(best_history.history['loss'], label=f'Training Loss (lr={best_lr})')
plt.plot(best_history.history['val_loss'], linestyle='--', label=f'Validation Loss (lr={best_lr})')
plt.title(f'Training and Validation loss (best lr={best_lr})')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot Training and Validation Loss for all learning rates
plt.figure(figsize=(14, 7))

for idx, (lr, history) in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Training Loss (lr={lr})')
    plt.plot(history.history['val_loss'], linestyle='--', label=f'Validation Loss (lr={lr})')

plt.title('Training and Validation Loss for different learning rates')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


#%%
y_pred = model.predict(X_test)

#%%
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE) on Test Set without rescaling: {mse}")
print(f"Mean Absolute Error (MAE) on Test Set without rescaling: {mae}")
#%%
'''
=====================================================================================================================
----------------------------------------------------------------------------------------------------------------------
A more advanced neural network
Model 2
with more hidden layers
trained on the original data
'''

# Define learning rates to test
#learning_rates = [0.00001, 0.0001, 0.001, 0.01]
# Prepare the data

df_processed_por = data_processed_por.copy()
#X = df_processed_por.drop(columns=['G3', 'G1', 'G2'])
X = df_processed_por.drop(columns=['G3'])

y = df_processed_por[['G3']]

PredictorScaler = StandardScaler()
#TargetVarScaler = StandardScaler()

X = PredictorScaler.fit_transform(X)
#y = TargetVarScaler.fit_transform(y)

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#%%

random.seed(seed_value)
#learning_rates = [0.00001, 0.0001, 0.001, 0.01]
learning_rates = [0.0001, 0.0005 ,0.001, 0.005, 0.01]
# Initialize lists to store histories and final validation losses
histories = []
final_val_losses = []
min_val_losses = []
stopped_epochs = []
best_model = None  # To store the model with the best learning rate

# Loop over each learning rate
for lr in learning_rates:
    print(f"Training model with learning rate = {lr}")
    
    # Define the model
    model5 = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        #Dense(70, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # Compile the model with the given learning rate
    model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model5.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=32,
                        epochs=100,  # Set a large number of epochs to allow early stopping
                        callbacks=[early_stopping],
                        verbose=1)  # Set verbose to 1 if you want to see the progress
    
    # Store the history, the final validation loss, and the epoch at which training stopped
    histories.append((lr, history))
    min_val_loss = min(history.history['val_loss'])  # Get the minimum validation loss
    min_val_losses.append(min_val_loss)
    final_val_losses.append(history.history['val_loss'][-1])
    stopped_epochs.append(early_stopping.stopped_epoch + 1)  # +1 because epoch count starts at 0

    # Update the best model
    if best_model is None or final_val_losses[-1] == min(final_val_losses):
        best_model = model5

# Determine the best learning rate based on the lowest final validation loss
best_lr_idx = np.argmin(final_val_losses)
best_lr, best_history = histories[best_lr_idx]

print(f"Best learning rate: {best_lr}")
print(f"Final Validation Loss for best learning rate: {final_val_losses[best_lr_idx]}")
print(f"Minimum Validation Loss for best learning rate: {min_val_losses[best_lr_idx]}")
print(f"Training stopped after {stopped_epochs[best_lr_idx]} epochs for the best learning rate.")

# Plot the Training and Validation Loss for the best learning rate
plt.figure(figsize=(14, 8))
plt.plot(best_history.history['loss'], label=f'Training Loss (lr={best_lr})')
plt.plot(best_history.history['val_loss'], linestyle='--', label=f'Validation Loss (lr={best_lr})')
plt.title(f'Training and Validation Loss (Best lr={best_lr})', fontsize = 22)
plt.ylabel('Mean Squared Error', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.legend(loc='upper right', fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
plt.grid(True)
plt.show()

# Plot Training and Validation Loss for all learning rates
plt.figure(figsize=(14, 8))
for idx, (lr, history) in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Training Loss (lr={lr})')
    plt.plot(history.history['val_loss'], linestyle='--', label=f'Validation Loss (lr={lr})')
plt.title('Training and Validation Loss for Different Learning Rates', fontsize = 22)
plt.ylabel('Mean Squared Error', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.legend(loc='upper right', fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
plt.grid(True)
plt.show()

# Evaluate the model with the best learning rate on the test set
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rsquared = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE) on Test Set without rescaling: {mse}")
print(f"Mean Absolute Error (MAE) on Test Set without rescaling: {mae}")
print(f"R-squared: {rsquared}")


# R-squared
# 8 => -0.265
# 50 => 0.106
# 50, 70 => 0.140
# 70, 70 => 0.219


#%%


'''
=====================================================================================================================
Neural Network with synthetic data
====================================
'''

df_processed_por = data_processed_por.copy()
#%%
X = df_processed_por.drop(columns=['G3'])
y = df_processed_por['G3']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
y_target = synthetic_data_gaussian_1['G3']
y_target = pd.DataFrame(y_target)
y_train = pd.DataFrame(y_train)

X_features = synthetic_data_gaussian_1.drop(columns=['G3'])
#%%
y_train_sin = pd.concat([y_train, y_target], axis=0).reset_index(drop=True)
X_train_sin = pd.concat([X_train, X_features], axis=0).reset_index(drop=True)
#%%

X_train_sin = X_train_sin.drop(columns=['G1', 'G2'])
X_test = X_test.drop(columns=['G1', 'G2'])

#%%
print("X_train shape:", X_train_sin.shape)
print("y_train shape:", y_train_sin.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

#%%
# Using Standard Scaler
predictor_scaler = StandardScaler()
#target_scaler = StandardScaler()

predictor_scaler = StandardScaler()
#target_scaler = StandardScaler()

# Fit the scaler on X_train and transform X_train
X_train_scaled = predictor_scaler.fit_transform(X_train_sin)
# Transform X_test using the same scaler fit on X_train
X_test_scaled = predictor_scaler.transform(X_test)

y_train = pd.DataFrame(y_train_sin)
y_test_1 = pd.DataFrame(y_test)

# Fit the scaler on y_train and transform y_train
#y_train_scaled = target_scaler.fit_transform(y_train_sin.values.reshape(-1, 1))
# Transform y_test using the same scaler fit on y_train
#y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))


# Checking the shapes after scaling and adding the constant
print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(y_train.shape)
print(y_test_1.shape)


#%%
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)


random.seed(seed_value)
#learning_rates = [0.00001, 0.0001, 0.001, 0.01]
learning_rates = [0.0001, 0.0005 ,0.001, 0.005, 0.01]
#learning_rates = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

X_val, X_test, y_val, y_test = train_test_split(X_test_scaled, y_test_1, test_size=0.5, random_state=42)
#%%

# Initialize lists to store histories and final validation losses
histories = []
final_val_losses = []
min_val_losses = []
stopped_epochs = []
best_model = None  # To store the model with the best learning rate

# Loop over each learning rate
for lr in learning_rates:
    print(f"Training model with learning rate = {lr}")
    
    # Define the model
    model7 = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # Compile the model with the given learning rate
    model7.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model7.fit(X_train_scaled, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=32,
                        epochs=100,  # Set a large number of epochs to allow early stopping
                        callbacks=[early_stopping],
                        verbose=1)  # Set verbose to 1 if you want to see the progress
    
    # Store the history, the final validation loss, and the epoch at which training stopped
    histories.append((lr, history))
    min_val_loss = min(history.history['val_loss'])  # Get the minimum validation loss
    min_val_losses.append(min_val_loss)
    final_val_losses.append(history.history['val_loss'][-1])
    stopped_epochs.append(early_stopping.stopped_epoch + 1)  # +1 because epoch count starts at 0

    # Update the best model
    if best_model is None or final_val_losses[-1] == min(final_val_losses):
        best_model = model7

# Determine the best learning rate based on the lowest final validation loss
best_lr_idx = np.argmin(final_val_losses)
best_lr, best_history = histories[best_lr_idx]

print(f"Best learning rate: {best_lr}")
print(f"Final Validation Loss for best learning rate: {final_val_losses[best_lr_idx]}")
print(f"Minimum Validation Loss for best learning rate: {min_val_losses[best_lr_idx]}")
print(f"Training stopped after {stopped_epochs[best_lr_idx]} epochs for the best learning rate.")

# Plot the Training and Validation Loss for the best learning rate
plt.figure(figsize=(14, 8))
plt.plot(best_history.history['loss'], label=f'Training Loss (lr={best_lr})')
plt.plot(best_history.history['val_loss'], linestyle='--', label=f'Validation Loss (lr={best_lr})')
plt.title(f'Training and Validation Loss (Best lr={best_lr})', fontsize = 22)
plt.ylabel('Mean Squared Error', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.legend(loc='upper right', fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
plt.grid(True)
plt.show()

# Plot Training and Validation Loss for all learning rates
plt.figure(figsize=(14, 8))
for idx, (lr, history) in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Training Loss (lr={lr})')
    plt.plot(history.history['val_loss'], linestyle='--', label=f'Validation Loss (lr={lr})')
plt.title('Training and Validation Loss for Different Learning Rates', fontsize = 22)
plt.ylabel('Mean Squared Error', fontsize = 20)
plt.xlabel('Epochs', fontsize = 20)
plt.legend(loc='upper right', fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
plt.grid(True)
plt.show()

# Evaluate the model with the best learning rate on the test set
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rsquared = r2_score(y_test, y_pred), 
print(f"R-squared: {rsquared}")

print(f"Mean Squared Error (MSE) on Test Set without rescaling: {mse}")
print(f"Mean Absolute Error (MAE) on Test Set without rescaling: {mae}")


# R-squared:
# 8 => 0.113
# 50 => 0.137 patience of 5
# 50 => 0.180 patience of 10
# 50, 70 => 0.1246  patience of 10
# 100 => 0.185
# 100 => 0.1856 patience of 10
# 100, 50 => 0.069





#%%
'''
Versuche 
'''











#%%
'''
---------------------------------------------------------------------------------------------------------------------
Vorhersage Sprung von G2 auf G3
'''

#%%
df = data_processed_por.copy()
#%%
df['G2_to_G3'] = df['G3'] - df['G2']
#%%
X = df.drop(columns=['G2_to_G3'])
y = df['G2_to_G3']
#%%
predictor_scaler = StandardScaler()
target_scaler = StandardScaler()
#%%
X_scaled = predictor_scaler.fit_transform(X)
#%%
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
#%%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.3, random_state=42)
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Modell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen auf dem Testdatensatz
y_pred = model.predict(X_test)

# Modellbewertung
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

#%%









































































































# %%
# =====================================================================================================
# Neural Network with Cross Validation ---
# Name: NN_CV
# --------------------------------------------
# We split the data set => into Trainig and Testing
# The test set is still held out for the final evaluation
# A validation set is no longer needed when doing CV
# The training set is split into k smaller sets 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
#%%


#X = data_processed_por.drop(columns = ['G3'])
#y = data_processed_por['G3']
#y = pd.DataFrame(y)


#PredictorScaler=StandardScaler()
#TargetVarScaler=StandardScaler()

#PredictorScalerFit=PredictorScaler.fit(X)
#TargetVarScalerFit=TargetVarScaler.fit(y)

#X = PredictorScalerFit.transform(X)
#y = TargetVarScalerFit.transform(y)
#%%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print("Shape Training Features: ",X_train.shape)
#print("Shape Training Target: ",y_train.shape)
#print("Shape Test Features: ", X_test.shape)
#print("Shape Test Taregt: " ,y_test.shape)
#%%
#kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#%%
#def create_model_CV():
#    model_CV = Sequential()
#    model_CV.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
#    model_CV.add(Dense(200, activation='relu'))
    #model_CV.add(Dense(150, activation='relu'))
#    model_CV.add(Dense(50, activation='relu'))
#    model_CV.add(Dense(1, activation='linear'))
#    model_CV.compile(optimizer=Adam(learning_rate=0.009), loss='mean_squared_error')
#    return model_CV
#%%
#def print_scores(model_CV, X_train, y_train, X_valid, y_valid):
#    train_preds = model_CV.predict(X_train, batch_size=10000)
#    valid_preds = model_CV.predict(X_valid, batch_size=10000)
#    print('Train R2 = ', r2_score(y_train, train_preds),
#          'Train MSE = ', mean_squared_error(y_train, train_preds),
#          ', Valid R2 = ', r2_score(y_valid, valid_preds), 
#          ', Valid MSE = ', mean_squared_error(y_valid, valid_preds))
    
#%%
# Cross Validation Training und Validation
#mse_scores = []
#fold_no = 1
#for train_index, val_index in kfold.split(X_train):
#    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
#    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
#    print("Shape von Training Features in dem Fold: ",X_train_fold.shape)
#    print("Shape von Validation X in dem Fold: " ,X_val_fold.shape)

 #   model_CV = create_model_CV()

    # Early Stopping Callback
  #  early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

   # print(f'Training for fold {fold_no} ...')
    #history = model_CV.fit(X_train_fold, y_train_fold, 
     #                      validation_data=(X_val_fold, y_val_fold),
      #                     epochs=100, batch_size=32, verbose=1,
       #                    callbacks=[early_stopping])

    # Validierung und Auswertung
    #y_val_pred = model_CV.predict(X_val_fold)
    #mse = mean_squared_error(y_val_fold, y_val_pred)
    #mse_scores.append(mse)

    #print(f'Fold {fold_no} Validation Scores:')
    #print_scores(model_CV, X_train_fold, y_train_fold, X_val_fold, y_val_fold)

    # Plotten der Verluste
    #plt.plot(history.history['loss'], label='Train Loss')
    #plt.plot(history.history['val_loss'], label='Val Loss')
    #plt.title(f'Training and Validation Loss for Fold {fold_no}')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()

    #fold_no += 1

#print("Cross Validation MSE Scores: ", mse_scores)
#print("Mean Cross Validation MSE: ", np.mean(mse_scores))

# Modell auf dem gesamten Trainingsdatensatz trainieren
#final_model_CV = create_model_CV()
#final_history = final_model_CV.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])


#%%
# MSE und R² auf dem Testdatensatz bestimmen
#y_test_pred_cv = final_model_CV.predict(X_test)
#test_mse = mean_squared_error(y_test, y_test_pred_cv)
#test_r2 = r2_score(y_test, y_test_pred_cv)

# MSE & R-squared
#print("Test MSE: ", test_mse)
#print("Test R2: ", test_r2)

#print("Final Model Scores on Test Set:")
#print_scores(final_model_CV, X_train, y_train, X_test, y_test)

#%%
# RMSE ---
#rmse_cv = mean_squared_error(y_test, y_test_pred_cv, squared = False)
#print("RMSE: ", rmse_cv)
# RMSE: 0.45993
#%%
# MAE ---
#mae_cv = mean_absolute_error(y_test, y_test_pred_cv)
#print("MAE: ", mae_cv)
# MAE: 0.3143

#%%
# R-squared ---
#r2_cv = r2_score(y_test, y_test_pred_cv)
#print("R-squared: ", r2_cv)
# R-squared:

#%%
# Adjusted R-squared ---
#n = len(y_test)  # Number of observations
#p = 50  # Number of predictors
#adjusted_r2_cv = 1 - (1 - r2_cv) * (n - 1) / (n - p - 1)
#print("Adjusted R2: ", adjusted_r2_cv)
# 0.63087

#%%
