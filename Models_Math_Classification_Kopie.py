#%%
import numpy as np
import pandas as pd
import sdv
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import seaborn as sns
#%%
data_math = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Seminar/Daten/student/student-mat.csv',sep = ';' )
data_por = pd.read_csv('/Users/sophiasiefert/Documents/Vorlesungen /Master/Seminar/Daten/student/student-por.csv', sep = ';')
# %%
# Pass if G3 ≥ 10, else Failure
data_math['Results_G3'] = data_math['G3'].apply(lambda x: 'pass' if x >=10 else 'fail')
data_por['Results_G3'] = data_por['G3'].apply(lambda x: 'pass' if x >=10 else 'fail')
# %%
# Creating Results into binary values
mapping = {'pass': 1, 'fail': 0}
data_math['Results_G3'] = data_math['Results_G3'].map(mapping)

data_por['Results_G3'] = data_por['Results_G3'].map(mapping)
#%%
data_processed_math = data_math.copy()
data_processed_por = data_por.copy()
#%%
data_processed_math = pd.get_dummies(data_math, columns=['school','sex', 'address', 'famsize', 'Pstatus', 'guardian', 'Mjob', 'Fjob', 'reason' ], dtype = int)
#%%
# Binary variablen
data_processed_math['schoolsup'] = data_processed_math['schoolsup'].map({'no': 0, 'yes': 1})
data_processed_math['famsup'] = data_processed_math['famsup'].map({'no': 0, 'yes': 1})
data_processed_math['paid'] = data_processed_math['paid'].map({'no': 0, 'yes': 1})
data_processed_math['activities'] = data_processed_math['activities'].map({'no': 0, 'yes': 1})
data_processed_math['nursery'] = data_processed_math['nursery'].map({'no': 0, 'yes': 1})
data_processed_math['higher'] = data_processed_math['higher'].map({'no': 0, 'yes': 1})
data_processed_math['internet'] = data_processed_math['internet'].map({'no': 0, 'yes': 1})
data_processed_math['romantic'] = data_processed_math['romantic'].map({'no': 0, 'yes': 1})

# %%
# Label Encoding
# for categorical variables with a natural order
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data_processed_math['traveltime'] = label_encoder.fit_transform(data_math['traveltime'])
data_processed_math['Medu'] = label_encoder.fit_transform(data_math['Medu'])
data_processed_math['Fedu'] = label_encoder.fit_transform(data_math['Fedu'])
data_processed_math['traveltime'] = label_encoder.fit_transform(data_math['traveltime'])
data_processed_math['studytime'] = label_encoder.fit_transform(data_math['studytime'])
data_processed_math['failures'] = label_encoder.fit_transform(data_math['failures'])
data_processed_math['famrel'] = label_encoder.fit_transform(data_math['famrel'])
data_processed_math['freetime'] = label_encoder.fit_transform(data_math['freetime'])
data_processed_math['goout'] = label_encoder.fit_transform(data_math['goout'])
data_processed_math['Dalc'] = label_encoder.fit_transform(data_math['Dalc'])
data_processed_math['Walc'] = label_encoder.fit_transform(data_math['Walc'])
data_processed_math['health'] = label_encoder.fit_transform(data_math['health'])
# %%
# One Hot Encoding 
#data_math['sex'] = data_math['sex'].map({'M': 0, 'F': 1})
#data_math['school'] = data_math['school'].map({'GP': 0, 'MS': 1})
#data_math['famsize'] = data_math['famsize'].map({'GT3': 0, 'MS': 1})
# %%
data_processed_math['Age_Greater_Equal_18'] = np.where(data_math['age'] >= 18, 1, 0)
data_processed_math['Age_Less_Than_18'] = np.where(data_math['age'] < 18, 1, 0)
#%%
gewuenschte_spaltenreihenfolge = ['school_GP', 'school_MS',
                                  'sex_F', 'sex_M',
                                  'age',
                                  'address_R','address_U' ,
                                  'famsize_GT3', 'famsize_LE3',
                                  'Pstatus_A', 'Pstatus_T',
                                  'Medu',
                                  'Fedu',
                                  'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 
                                  'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
                                  'reason_course', 'reason_home', 'reason_other', 'reason_reputation',
                                  'guardian_father','guardian_mother', 'guardian_other',
                                  'traveltime',
                                  'studytime', 
                                  'failures',
                                  'schoolsup',
                                  'famsup', 
                                  'paid', 
                                  'activities', 
                                  'nursery', 
                                  'higher',
                                  'internet', 
                                  'romantic', 
                                  'famrel', 
                                  'freetime', 
                                  'goout', 
                                  'Dalc', 
                                  'Walc',
                                  'health', 
                                  'absences', 
                                  'G1', 
                                  'G2', 
                                  'G3', 
                                  'Results_G3' ]
#%%

data_processed_math = data_processed_math.reindex(columns=gewuenschte_spaltenreihenfolge)
#%%
for spalte in data_processed_math.columns:
    verteilung = data_processed_math[spalte].value_counts()
    print("Verteilung der Spalte '{}':".format(spalte))
    print(verteilung)
    print("\n")


#%%
#plt.figure(figsize=(10, 6))
#plt.hist(data_processed_math['Results_G3'], bins=2, edgecolor='black', alpha=0.7, rwidth=0.7)
#plt.xticks([0.25, 0.75], ['fail', 'pass'])
#plt.xlabel('Final grade G3')
#plt.ylabel('Frequency')
#plt.title('Distribution Mathematics final grade G3')
#plt.show()    
#%%
plt.figure(figsize=(10, 6))
plt.hist(data_processed_math['Results_G3'], bins=2, edgecolor='black', alpha=0.7, rwidth=0.7)
plt.xticks([0.25, 0.75], ['fail', 'pass'], fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Final grade G3', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Distribution Mathematics final grade G3', fontsize=18)
plt.show()
#%%
correlation_matrix = data_processed_math.corr()
print(correlation_matrix["Results_G3"].sort_values(ascending=False))

#%%
# Bar Plot Correlation
# ==================================

# Berechnung der gesamten Korrelationsmatrix
correlation_matrix = data_processed_math.corr()

# Sortiere die Korrelationen von 'Results_G3'
target_correlation = correlation_matrix["Results_G3"].drop('Results_G3').sort_values(ascending=False)



# Definiere manuell die "interessantesten" 5 (du kannst hier eigene Features auswählen oder einen Filter setzen)
interesting_features = ['G1', 'G2', 'higher', 'failures', 'goout', 'age']  # Beispiel
interesting_correlations = target_correlation[interesting_features]

# Kombiniere die Top 5 und die interessantesten 5
combined_correlations = pd.concat([interesting_correlations]).drop_duplicates()

# Plotten der Korrelationen
plt.figure(figsize=(10, 6))
sns.barplot(x=combined_correlations.index, y=combined_correlations.values, palette='magma')
plt.xticks(rotation=45)
plt.title('Correlation between Features and Target')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.show()


#%%
# Correlation matrix 
# ==============================
correlation_matrix = data_processed_math.corr()

# Definiere die interessanten Features, inklusive 'Results_G3'
#interesting_features = ['Results_G3', 'G1', 'G2', 'higher', 'failures', 'goout', 'age']
interesting_features = ['G3', 'G1', 'G2', 'higher', 'failures', 'goout', 'age']

# Extrahiere die Korrelationsmatrix nur für die interessanten Features
correlation_subset = correlation_matrix.loc[interesting_features, interesting_features]

# Plotten der Korrelationsmatrix als Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Top 10 Strongest Correlations with G3 (Mathematics dataset)')
plt.show()

#%%
from scipy.stats import pearsonr

results = {}

# Überprüfen, ob die Pearson-Korrelation signifikant ist
for feature in interesting_features:
    corr, p_value = pearsonr(data_processed_math['G3'], data_processed_math[feature])
    results[feature] = {'Pearson Coefficient': corr, 'p-value': p_value}

# Ergebnisse anzeigen
results_df = pd.DataFrame(results).T
print(results_df)

# Interpretation: Signifikante Korrelationen (p-Wert < 0.05)
significant_correlations = results_df[results_df['p-value'] < 0.05]
print("\nSignifikante Korrelationen:\n", significant_correlations)

#%%
'''
-------------------------------------------
Correlation Analysis
    showing the top 10 strongest correlated features
'''

#%%
# import pandas as pd
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
correlations = calculate_correlations(data_processed_math, 'Results_G3')

# Sortiere nach dem absoluten Wert der Korrelationen und wähle die 10 stärksten
sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1][0]), reverse=True)
top_10_correlations = dict(sorted_correlations[:10])

# Zeige die 10 stärksten Korrelationen und p-Werte
print("Die 10 stärksten Korrelationen mit G3:")
for feature, (corr, p_value) in top_10_correlations.items():
    print(f"{feature}: Korrelation = {corr:.3f}, p-Wert = {p_value:.3f}")

# Visualisierung der 10 stärksten Korrelationen
interesting_features = ['Results_G3'] + [feature for feature in top_10_correlations.keys()]

# Extrahiere die Korrelationsmatrix nur für die interessanten Features
correlation_subset = data_processed_math[interesting_features].corr()

# Plotten der Korrelationsmatrix als Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Top 10 Strongest Correlations with G3 (Mathematics dataset)')
plt.show()

#%%
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

# Korrelationen und p-Werte für 'Results_G3'
correlations = calculate_correlations(data_processed_math, 'Results_G3')

# Sortiere nach dem absoluten Wert der Korrelationen und wähle die 10 stärksten
sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1][0]), reverse=True)
top_10_correlations = dict(sorted_correlations[:10])

# Zeige die 10 stärksten Korrelationen und p-Werte
print("Die 10 stärksten Korrelationen mit G3:")
for feature, (corr, p_value) in top_10_correlations.items():
    print(f"{feature}: Korrelation = {corr:.3f}, p-Wert = {p_value:.3f}")

# Visualisierung der 10 stärksten Korrelationen
interesting_features = ['Results_G3'] + [feature for feature in top_10_correlations.keys()]

# Extrahiere die Korrelationsmatrix nur für die interessanten Features
correlation_subset = data_processed_math[interesting_features].corr()

# Plotten der Korrelationsmatrix als Heatmap mit angepassten Schriftgrößen
plt.figure(figsize=(13, 11))

# Heatmap mit angepasster Schriftgröße für die Annotationen
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 12})

# Anpassung der Titel- und Achsenbeschriftungen
plt.title('Top 10 Strongest Correlations with G3 (Mathematics dataset)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()













#%%
# =================================================================================================================================================================
# Splitting the Data into Training, Testing and Validation
# ================================================================================================================================================================
from sklearn.model_selection import train_test_split
df_processed_math = data_processed_math.drop(columns = ['G3'])

#%%
X = df_processed_math.drop(columns=['Results_G3'])
y = df_processed_math['Results_G3']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%

train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
print(train_data.shape)

test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
print(test_data.shape)
#%%
# =========================================================================================================================================================
# Um data leakage zu verhindern wird nur von den Trainingsdaten synthetische Daten erstellt
# Synthetische Daten erstellen

from sdv.metadata import SingleTableMetadata
#%%
from sdv.single_table import GaussianCopulaSynthesizer
#%%
metadata_math = SingleTableMetadata()
#%%
metadata_math.detect_from_dataframe(train_data)
#%%
synthesizer_gaussian = GaussianCopulaSynthesizer(
    metadata_math)

synthesizer_gaussian.fit(train_data)
#%%
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
    metadata_math
)
# Quality Report:94.91%
#%%
print(synthetic_data_gaussian_1.shape)
print(synthetic_data_gaussian_1.head(10))
#%%
diagnostic = run_diagnostic(train_data, synthetic_data_gaussian_1, metadata_math)
#%%
# Plotten der Verteilngen echte und synthetische Daten

if 'Results_G3' in train_data.columns and 'Results_G3' in synthetic_data_gaussian_1.columns:
    # Plotten der Verteilung von 'G1'
    plt.figure(figsize=(12, 6))

    # Plot für die echte Daten
    sns.histplot(train_data['Results_G3'], kde=True, color='blue', label='original data', stat='density', bins=2)

    # Plot für die synthetischen Daten
    sns.histplot(synthetic_data_gaussian_1['Results_G3'], kde=True, color='red', label='synthetic data', stat='density', bins=2)

    plt.title('Comparison density of G3')
    plt.xlabel('G3')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
else:
    print("Die Spalte 'G1' ist in einem der DataFrames nicht vorhanden.")
#%%

#%%
# Evaluating the synthetic data 
# comparing the synthetic data with the real data (data_math)
from sdv.evaluation.single_table import evaluate_quality

# 1000 data points created with gaussain coupla 
quality_report_g1 = evaluate_quality(
    data_math,
    synthetic_data_gaussian_1,
    metadata_math
)
# 90.83%


#%%
# Evaluating the synthetic data 
# comparing the synthetic data with the real data (data_math_processed)
from sdv.evaluation.single_table import evaluate_quality

# Gaussian Copula ----
# 1000 data points created with gaussain coupla 
quality_report_g1_p = evaluate_quality(
    data_processed_math,
    synthetic_data_gaussian_1_p,
    metadata_2
)
# 93.94%

#%%
# ==================================================================================================
# Deciision Tree ----
#------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier, plot_tree

#%%
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

'''
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------
 Decision Tree                                                  (1)
 including G1 and G2
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
tree = DecisionTreeClassifier(random_state  = 42)
tree = tree.fit(X_train, y_train)

predict_tree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)

#%%
# Accuracy 
accuracy = accuracy_score(y_test, predict_tree)
print('Accuracy of the Decision Tree with G1 and G2',accuracy)
# Accuracy of 0.848
#%%
# AUC
auc_score = roc_auc_score(y_test, y_prob_tree[:,1])
print("AUC Decision Tree with G1 and G2: ", auc_score)
# AUC: 0.840

#%%
# Log-Loss
loss = log_loss(y_test, y_prob_tree)
print('Log Loss', loss)
# Log Loss: 5.451

#%%
# Feature importance:
# All Features
feature_importances = tree.feature_importances_

# DataFrame zur Visualisierung erstellen
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sortieren nach Bedeutung
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plotten der Feature Importances
plt.figure(figsize=(18, 12))
sns.barplot(x=features_df['Importance'], y=features_df['Feature'], palette='viridis')
plt.title('Feature Importance Decision Tree (including G1 and G2)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have already trained your model and have feature importances
# feature_importances = tree.feature_importances_

# Create a DataFrame for visualization
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Get the top 10 features
top_features_df = features_df.head(10)

# Plotting the Feature Importances
plt.figure(figsize=(15, 12))
sns.barplot(x=top_features_df['Importance'], y=top_features_df['Feature'], palette='viridis')
plt.title('Top 10 Feature Importance  Decision Tree (including G1 and G2)', fontsize=22)
plt.xlabel('Importance', fontsize=22)
plt.ylabel('Feature', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#%%
# Confusion Matrix
cm_1 = confusion_matrix(y_test, predict_tree)
print("Confusion Matrix Decision Tree (including G1, G2)")
print(cm_1)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_1)
disp.plot()
plt.title('Confusion Matrix - Decision Tree (including G1, G2)')
plt.show()


#%%
import matplotlib.pyplot as plt

'''
__________________________________________________________________________________________________________________
Decision Tree 1
ROC
including G1 and G2
========================================= = = = = =================================================================
'''
y_scores_decsiontree1 = tree.predict_proba(X_test)[:, 1]
grid_fpr_decison_tree1, grid_tpr_decision_tree1, grid_thresholds_decision_tree1 = roc_curve(y_true = y_test, y_score=y_scores_decsiontree1)


plt.figure(figsize=(8, 6))
plt.plot(grid_fpr_decison_tree1, grid_tpr_decision_tree1, label='Decision Tree ROC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve For Decision Tree (including G1 and G2)')
plt.show()

#%%
'''
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Decision Tree                                                       (2)
    without G1 and G2
    prediction the result of G3 without G1 and G2
    as the training data we will use the split which we have choosen for creating the synthetic data
________________________________________________________________________________________________________________________________________
'''

X_train_tree = X_train.copy()
X_test_tree = X_test.copy()
y_train_tree = y_train.copy()
y_test_tree = y_test.copy()


#%%
# Drop G1 and G2
X_train_tree = X_train_tree.drop(columns=['G1', 'G2'])
X_test_tree = X_test_tree.drop(columns=['G1', 'G2'])

#%%
from sklearn.tree import DecisionTreeClassifier, plot_tree

clf_tree2 = DecisionTreeClassifier(random_state  = 42)
clf_tree2 = clf_tree2.fit(X_train_tree, y_train_tree)
#%%
# Prediction
predict_clf_tree2 = clf_tree2.predict(X_test_tree)
#%%
# Probability Prediction
y_prob_tree2 = clf_tree2.predict_proba(X_test_tree)
#%%
plot_tree(clf_tree2)
#%%
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, log_loss
#%%
# Accuracy 
accuracy = accuracy_score(y_test_tree, predict_clf_tree2)
print('Accuracy of the Decision Tree without G1 and G2',accuracy)
# Accuracy of 0.68067
#%%
# AUC
auc_score = roc_auc_score(y_test_tree,y_prob_tree2[:,1] )
print("AUC Decision Tree without G1 and G2: ", auc_score)
# AUC: 0.6392

#%%
# Log-Loss
loss = log_loss(y_test_tree, y_prob_tree2)
print('Log Loss', loss)
# Log Loss: 11.509
#%%
# ROC Curve - Decision Tree 2
y_scores_decsiontree2 = clf_tree2.predict_proba(X_test_tree)[:, 1]
grid_fpr_decison_tree2, grid_tpr_decision_tree2, grid_thresholds_decision_tree2 = roc_curve(y_true = y_test_tree, y_score=y_scores_decsiontree2)

plt.figure(figsize=(8, 6))

plt.plot(grid_fpr_decison_tree2, grid_tpr_decision_tree2, label='Decision Tree ROC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve For Decision Tree (excluding G1 and G2)')
plt.show()

#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Confusion Matrix
cm = confusion_matrix(y_test_tree ,predict_clf_tree2)
print("Confusion Matrix Decision Tree", cm)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title('Confusion Matrix - Decision Tree ')
plt.show()
#disp.plot()
#%%
# Feature Importance
feature_importances = clf_tree2.feature_importances_

# DataFrame zur Visualisierung erstellen
features_df = pd.DataFrame({
    'Feature': X_train_tree.columns,
    'Importance': feature_importances
})

# Sortieren nach Bedeutung
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plotten der Feature Importances
plt.figure(figsize=(18, 12))
sns.barplot(x=features_df['Importance'], y=features_df['Feature'], palette='viridis')
plt.title('Feature Importance from Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#%%
# Feature Importance
# Plotting the top 10 features
# DataFrame zur Visualisierung erstellen
features_df = pd.DataFrame({
    'Feature': X_train_tree.columns,
    'Importance': feature_importances
})

# Sortieren nach Bedeutung
features_df = features_df.sort_values(by='Importance', ascending=False).head(10)

# Plotten der Feature Importances
plt.figure(figsize=(15, 12))
sns.barplot(x=features_df['Importance'], y=features_df['Feature'], palette='viridis')

# Titel und Achsenbeschriftungen
plt.title('Top 10 Feature Importance Decision Tree', fontsize=22)
plt.xlabel('Importance', fontsize=22)
plt.ylabel('Feature', fontsize=22)

# Anpassen der Schriftgröße der Achsenbeschriftungen
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Plot anzeigen
plt.show()

#%%
'''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Decision Tree                                                       (3)
        with some additional synthetic data
----------------------------------------------------------------------------------------------------------------------------------------
'''

y_target = synthetic_data_gaussian_1['Results_G3']
X_features = synthetic_data_gaussian_1.drop(columns=['Results_G3'])


y_train_tree3 = pd.concat([y_train, y_target], axis=0).reset_index(drop=True)
X_train_tree3 = pd.concat([X_train, X_features], axis=0).reset_index(drop=True)
#%%
X_train_tree3 = X_train_tree3.drop(columns=['G1', 'G2'])
X_test = X_test.drop(columns=['G1', 'G2'])
#%%
print("X_train shape:", X_train_tree3.shape)
print("y_train shape:", y_train_tree3.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)
#%%
model_tree = DecisionTreeClassifier(random_state  = 42)
model_tree = model_tree.fit(X_train_tree3, y_train_tree3)

#%%
predict_model_tree = model_tree.predict(X_test)
y_model_tree = model_tree.predict_proba(X_test)

#%%
accuracy = accuracy_score(y_test, predict_model_tree)
print('Accuracy of the Decision Tree synthetic data',accuracy)
# Accuracy of 0.5966

#%%
# AUC
auc_score = roc_auc_score(y_test, y_model_tree[:,1])
print("AUC Decision Tree synthetic data: ", auc_score)
# AUC: 0.5506

#%%
# Log-Loss
loss = log_loss(y_test, y_model_tree)
print('Log Loss, Decision Tree synthetic data', loss)
# Log Loss: 14.538

#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Confusion Matrix
cm = confusion_matrix(y_test ,predict_model_tree)
print("Confusion Matrix Decision Tree syntheic data", cm)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title('Confusion Matrix - Decision Tree 3 (synthetic data)')
plt.show()
#%%

# ROC Curve - Decision Tree 3
        # Trained on synthetic data
y_scores_decsiontree3 = model_tree.predict_proba(X_test)[:, 1]
grid_fpr_decison_tree3, grid_tpr_decision_tree3, grid_thresholds_decision_tree3 = roc_curve(y_true = y_test, y_score=y_scores_decsiontree3)

plt.figure(figsize=(8, 6))

plt.plot(grid_fpr_decison_tree3, grid_tpr_decision_tree3, label='Decision Tree ROC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve For Decision Tree (synthetic data)')
plt.show()

#%%
# #####################################################################################################################################

# Plotting all Decision Trees together
plt.figure(figsize=(8, 6))
plt.plot(grid_fpr_decison_tree1, grid_tpr_decision_tree1, label='Decision Tree 1 (including G1, G2)')
plt.plot(grid_fpr_decison_tree2, grid_tpr_decision_tree2, label='Decision Tree 2 (excluding G1, G2)')
plt.plot(grid_fpr_decison_tree3, grid_tpr_decision_tree3, label='Decision Tree 3 (synthetic Data)')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC Curve for different Decision Trees')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#%%
'''

----------------------------------------------------------------------------------------------------------------------------------------
Random Forest Classifier with G1 and G2
---------------------------------------------------------------------------------------------------------------------------------------
'''

rf1_all = RandomForestClassifier(random_state=42)
rf1_all = rf1_all.fit(X_train, y_train)

#%%
prediction_all = rf1_all.predict(X_test) 
#%%
rf1_all_y_hat = rf1_all.predict_proba(X_test)
#%%
# Accuracy ---
accuracy_all = accuracy_score(y_test, prediction_all)
print("Accuracy Random Forest Classifier (no tuning):", accuracy_all)
# 0.915

#%%
#print("The log loss of the model with Grid Search is: " + str(log_loss(y_test_rf1, rf1_y_hat)))
loss = log_loss(y_test, rf1_all_y_hat)
print("Log Loss of the model without Random Search", loss)
print("The ROC AUC score of the model without random Search: " +str(metrics.roc_auc_score(y_test, rf1_all_y_hat[:,1])))
# AUC: 0.979
#%%
naiive_fpr_all, naiive_tpr_all, naiive_thresholds_all = roc_curve(y_true=y_test, y_score=rf1_all_y_hat[:,1])
plt.figure(figsize=(8, 6))
plt.plot(naiive_fpr_all, naiive_tpr_all, label='RFM with G1,G2 ROC AUC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve Random Forest Model 1')
plt.show()

#%%
cm_all = confusion_matrix(y_test, prediction_all)
print("Confusion Matrix Decision Tree (including G1, G2)")
print(cm_all)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_all)
disp.plot()
plt.title('Confusion Matrix - Random Forest (including G1, G2)')
plt.show()









#%%
# ====================================================================================================
# Random Forest Classifier
# ---------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

'''
================================================================================================================================================================
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Model 1
    without G1 and G2
    Random Forest Model 
    no hyperparameter tuning
----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
from sklearn import metrics
X_train_rf1 = X_train.copy()
X_test_rf1 = X_test.copy()
y_train_rf1 = y_train.copy()
y_test_rf1 = y_test.copy()

#%%

X_train_rf1 = X_train_rf1.drop(columns=['G1', 'G2'])
X_test_rf1 = X_test_rf1.drop(columns=['G1', 'G2'])

#%%
model_rf1 = RandomForestClassifier(random_state=42)
model_rf1 = model_rf1.fit(X_train_rf1, y_train_rf1)

#%%
predictions_rf1 = model_rf1.predict(X_test_rf1)
#%%
rf1_y_hat = model_rf1.predict_proba(X_test_rf1)
y_pred_proba1 = model_rf1.predict_proba(X_test_rf1)[:,1]

#%%
# Feature Importance ----
feature_importances = model_rf1.feature_importances_

# DataFrame zur Visualisierung erstellen
features_df = pd.DataFrame({
    'Feature': X_train_rf1.columns,
    'Importance': feature_importances
})

# Sortieren nach Bedeutung
features_df = features_df.sort_values(by='Importance', ascending=False).head(10)

# Plotten der Feature Importances
plt.figure(figsize=(15, 12))
sns.barplot(x=features_df['Importance'], y=features_df['Feature'], palette='viridis')
plt.title('Top 10 Feature Importance Randrom Forest ', fontsize = 22)
plt.xlabel('Importance', fontsize = 22)
plt.ylabel('Feature', fontsize = 22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#%%

from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

# Compute permutation importance
result = permutation_importance(model_rf1, X_test_rf1, y_test_rf1, n_repeats=10, random_state=42)

perm_importances = result.importances_mean
perm_std = result.importances_std
sorted_idx = perm_importances.argsort()
feature_names = X_test_rf1.columns

# Create a DataFrame for importances
importance_df = pd.DataFrame({'Importance': perm_importances, 'Std': perm_std}, 
                              index=feature_names[sorted_idx]).sort_values('Importance', ascending=True)

# Plotting
plt.figure(figsize=(17, 7))
plt.bar(importance_df.index, importance_df['Importance'], yerr=importance_df['Std'], 
        color='skyblue', edgecolor='black')
plt.title('Permutation Feature Importances', fontsize=16)
plt.ylabel('Importance', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xticks(fontsize=12, rotation=90)  # Rotate feature names to be vertical
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Assuming model_rf1 is already trained
# Calculate baseline accuracy
baseline_score = accuracy_score(y_test_rf1, model_rf1.predict(X_test_rf1))

# Compute permutation importance using accuracy
result = permutation_importance(model_rf1, X_test_rf1, y_test_rf1, 
                                n_repeats=10, random_state=42, 
                                scoring='accuracy')

perm_importances = result.importances_mean
perm_std = result.importances_std
sorted_idx = perm_importances.argsort()
feature_names = X_test_rf1.columns

# Create a DataFrame for importances
importance_df = pd.DataFrame({'Importance': perm_importances, 'Std': perm_std}, 
                              index=feature_names[sorted_idx]).sort_values('Importance', ascending=True)

# Plotting
plt.figure(figsize=(17, 8)) #(x,y)
plt.bar(importance_df.index, importance_df['Importance'], yerr=importance_df['Std'], 
        color='skyblue', edgecolor='black')
plt.title('Permutation Feature Importances (Based on Accuracy)', fontsize=20)
plt.ylabel('Importance', fontsize=18)
plt.xlabel('Features', fontsize=18)
plt.xticks(fontsize=16, rotation=90)  # Rotate feature names to be vertical
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()








#%%
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence

# Partial Dependence Plot -----

# Get feature names from the DataFrame
feature_names = X_train_rf1.columns.tolist()
print("Available features:", feature_names)


feature_name = "absences"  #name of the feature you want

# Get the index of the feature from its name
feature_index = X_train_rf1.columns.get_loc(feature_name)

# Compute Partial Dependence for the specified feature
pd_results = partial_dependence(
    model_rf1, 
    X_test_rf1, 
    features=feature_index, 
    kind="average", 
    grid_resolution=5)


display = PartialDependenceDisplay(
    [pd_results], features=[(feature_index,)], feature_names=feature_names, target_idx=0, deciles=deciles
)

display.plot()

#plt.suptitle(f'Partial dependence plot for {feature_name}', fontsize=22)
plt.suptitle(f'Partial dependence plot for {feature_name}')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
plt.xlabel('absences')
plt.ylabel('Partial dependence')

plt.show()


#%%
# Accuracy ---
accuracy = accuracy_score(y_test_rf1, predictions_rf1)
print("Accuracy Random Forest Classifier (no tuning):", accuracy)
# 0.647


#%%
#print("The log loss of the model with Grid Search is: " + str(log_loss(y_test_rf1, rf1_y_hat)))
loss = log_loss(y_test_rf1, rf1_y_hat)
print("Log Loss of the model without Random Search", loss)
print("The ROC AUC score of the model without random Search: " +str(metrics.roc_auc_score(y_test_rf1, y_pred_proba1)))
# Log Loss => 0.660
# AUC => 0.635
#%%

#%%
# Random Forest Model 1 (no tuning)
# ROC
naiive_fpr, naiive_tpr, naiive_thresholds = roc_curve(y_true=y_test_rf1, y_score=rf1_y_hat[:,1])
plt.figure(figsize=(8, 6))
plt.plot(naiive_fpr, naiive_tpr, label='Naive ROC AUC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve Random Forest Model 1')
plt.show()

#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Confusion Matrix

cm = confusion_matrix(y_test_rf1 ,predictions_rf1)
print("Confusion Matrix Random Forest 1", cm)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title('Confusion Matrix - Random Forest 1')
plt.show()

#%%

'''
------------------------------------------------------------------------
Model 2
    Random Forest Classifier
    with Random Search CV
------------------------------------------------------------------------
'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

#===============================================================================================================
'''
Random Forest Model 2 ----
Tuned model with Random Search CV
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

max_features = [ 'sqrt', 'log2', None]

random_search_params = {
    'n_estimators': [int(x) for x in range(50, 701, 50)],  # Explore a broad range
    'max_features': max_features,
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Perform Randomized Search
random_search2 = RandomizedSearchCV(estimator=rf, 
                                   param_distributions=random_search_params, 
                                   n_iter=200,  # Number of different combinations to try
                                   cv=5, 
                                   verbose=2, 
                                   random_state=42, 
                                   n_jobs=-1, 
                                   scoring='accuracy')

# Fit the model
random_search2.fit(X_train_rf1, y_train_rf1)

# Best parameters from Randomized Search
print(f'Best parameters from Randomized Search: {random_search2.best_params_}')

#%%
y_pred2 = random_search2.predict(X_test_rf1)
y_pred_proba2 = random_search2.predict_proba(X_test_rf1)[:,1]
#%%
# Log Loss berechnen
logloss2 = log_loss(y_test_rf1, y_pred_proba2)
print(f'Log Loss tunded RFM: {logloss2}')
# 0.597

# AUC berechnen
#auc2 = roc_auc_score(y_test_rf1, y_pred_proba2[:,1])
#print(f'AUC Tuned RFM: {auc}')
# 0.610
#%%
# Accuracy
accuracy2 = accuracy_score(y_test_rf1, y_pred2)
print('Accuracy score Tuned RFM ', accuracy2)
# 0.714

#%%

'''
----------------------------------
Permutation Importance ---
Tuned RFM 2
---------------------------------
'''
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

# Compute permutation importance
result = permutation_importance(random_search2.best_estimator_, X_test_rf1, y_test_rf1, n_repeats=10, random_state=42)

perm_importances = result.importances_mean
perm_std = result.importances_std
sorted_idx = perm_importances.argsort()
feature_names = X_test_rf1.columns

# Create a DataFrame for importances
importance_df = pd.DataFrame({'Importance': perm_importances, 'Std': perm_std}, 
                              index=feature_names[sorted_idx]).sort_values('Importance', ascending=True)

# Plotting
plt.figure(figsize=(17, 7))
plt.bar(importance_df.index, importance_df['Importance'], yerr=importance_df['Std'], 
        color='skyblue', edgecolor='black')
plt.title('Permutation Feature Importances', fontsize=16)
plt.ylabel('Importance', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xticks(fontsize=12, rotation=90)  # Rotate feature names to be vertical
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%%
'''
-----------------------------
Permutation Importance
Accuracy
Tuned Model 2
'''
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Assuming model_rf1 is already trained
# Calculate baseline accuracy
baseline_score = accuracy_score(y_test_rf1, random_search2.best_estimator_.predict(X_test_rf1))

# Compute permutation importance using accuracy
result = permutation_importance(model_rf1, X_test_rf1, y_test_rf1, 
                                n_repeats=10, random_state=42, 
                                scoring='accuracy')

perm_importances = result.importances_mean
perm_std = result.importances_std
sorted_idx = perm_importances.argsort()
feature_names = X_test_rf1.columns

# Create a DataFrame for importances
importance_df = pd.DataFrame({'Importance': perm_importances, 'Std': perm_std}, 
                              index=feature_names[sorted_idx]).sort_values('Importance', ascending=True)

# Plotting
plt.figure(figsize=(17, 8)) #(x,y)
plt.bar(importance_df.index, importance_df['Importance'], yerr=importance_df['Std'], 
        color='skyblue', edgecolor='black')
plt.title('Permutation Feature Importances (Based on Accuracy)', fontsize=20)
plt.ylabel('Importance', fontsize=18)
plt.xlabel('Features', fontsize=18)
plt.xticks(fontsize=16, rotation=90)  # Rotate feature names to be vertical
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#%%

premu_rfr_train = permutation_importance(random_search2.best_estimator_, 
X_train_rf1, 
y_train_rf1, 
scoring = 'accuracy')

premu_rfr_test = permutation_importance(random_search2.best_estimator_, X_test_rf1, y_test_rf1, scoring = 'accuracy' )

#results = [premu_rfr_train]
results = [premu_rfr_test]

names = ['Random Forest']


graph_data = {}
for result, name in zip(results, names):
    graph_data[name] = result['importances_mean']

n_cols = len(X_train_rf1)
# make final dataframe
graph_data = pd.DataFrame.from_dict(graph_data, orient='index', columns=X_train_rf1.columns)
graph_data.reset_index(inplace=True, drop=False)
graph_data.rename(columns={'index': 'model_name'}, inplace=True)
graph_data = graph_data.melt(id_vars='model_name')

# create visual
plt.figure(figsize=[23,6])
plt.axhline(0, c='black')
[plt.axvline(i + 0.5, linestyle='--', c='black') for i in range(0, n_cols)]
sns.barplot(x=graph_data['variable'], y=graph_data['value'])
plt.title("Permutation Feature Importance Tuned Random Forest (Test)", fontsize = 20)
plt.xlabel("Features", fontsize = 18)
plt.ylabel("Change in Accuracy", fontsize = 18)
plt.xticks(rotation=90, fontsize = 16)
plt.yticks(fontsize = 16)

plt.show()
#%%
#%%
'''
-----------------------
Partial Dependence Plot
----------------------
'''


from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence

# Partial Dependence Plot -----

# Get feature names from the DataFrame
feature_names = X_train_rf1.columns.tolist()
print("Available features:", feature_names)


feature_name = "failures"  #name of the feature you want

# Get the index of the feature from its name
feature_index = X_train_rf1.columns.get_loc(feature_name)

# Compute Partial Dependence for the specified feature
pd_results = partial_dependence(
    random_search2.best_estimator_, 
    X_test_rf1, 
    features=feature_index, 
    kind="average", 
    grid_resolution=5)


display = PartialDependenceDisplay(
    [pd_results], features=[(feature_index,)], feature_names=feature_names, target_idx=0, deciles=deciles
)

display.plot()

#plt.suptitle(f'Partial dependence plot for {feature_name}', fontsize=22)
plt.title(f'Partial dependence plot for {feature_name}')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
plt.xlabel('failures')
plt.ylabel('Partial dependence')

plt.show()


#%%
# Random Forest Model 2
# ROC plot

naiive_fpr_tuned, naiive_tpr_tuned, naiive_thresholds_tuned = roc_curve(y_true=y_test_rf1, y_score = y_pred_proba2)

tuned_auc = roc_auc_score(y_test_rf1, y_pred_proba2)

plt.figure(figsize=(8, 6))
plt.plot(naiive_fpr_tuned, naiive_tpr_tuned, label=f'Tuned Random Forest ROC AUC (AUC = {tuned_auc:.3f})')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve Tuned Random Forest Model 2')
plt.show()

print(f"Calculated AUC: {tuned_auc:.3f}")




grid_search.best_score_
#%%
# --------------------------------------------
# Plotting the results of the RandomSearch CV ------

random_results = pd.DataFrame(random_search2.cv_results_)

# Zeige die wichtigsten Spalten an
random_results = random_results[['param_n_estimators', 'param_max_features', 'mean_test_score']]

# Sortiere nach der Anzahl der Bäume
random_results = random_results.sort_values('param_n_estimators')
print(random_results)

#%%
random_results['param_max_features'] = random_results['param_max_features'].fillna('None')

plt.figure(figsize=(12, 8))

for max_feature in random_results['param_max_features'].unique():
    subset = random_results[random_results['param_max_features'] == max_feature]
    print(f"Plotting max_features={max_feature}:")
    print(subset)
    
    plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o', label=f'max_features={max_feature}')

plt.xlabel('Number of Estimators (n_estimators)', fontsize = 18)
plt.ylabel('Mean Test Accuracy', fontsize = 18)
plt.title('n_estimators vs. Accuracy for different max_features', fontsize = 20)

plt.ylim([0.600, 0.800])
plt.legend(title='max_features', fontsize = 18, title_fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

plt.show()


#%%
'''
Feature Importancy by permutation
'''

print(f"model score on training data: {random_search2.best_estimator_.score(X_train_rf1, y_train_rf1)}")
print(f"model score on testing data: {random_search2.best_estimator_.score(X_test_rf1, y_test_rf1)}")

#%%
def get_score_after_permutation(model, X, y, curr_feat):
    """return the score of model when curr_feat is permuted"""

    X_permuted = X.copy()
    col_idx = list(X.columns).index(curr_feat)
    # permute one column
    X_permuted.iloc[:, col_idx] = np.random.permutation(
        X_permuted[curr_feat].values
    )

    permuted_score = model.score(X_permuted, y)
    return permuted_score


def get_feature_importance(model, X, y, curr_feat):
    """compare the score when curr_feat is permuted"""

    baseline_score_train = model.score(X, y)
    permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)

    # feature importance is the difference between the two scores
    feature_importance = baseline_score_train - permuted_score_train
    return feature_importance


curr_feat = "absences"

feature_importance = get_feature_importance(random_search2.best_estimator_ , X_train_rf1, y_train_rf1, curr_feat)
print(
    f'feature importance of "{curr_feat}" on train set is '
    f"{feature_importance:.3}"
)

#%%
n_repeats = 10

list_feature_importance = []
for n_round in range(n_repeats):
    list_feature_importance.append(
        get_feature_importance(random_search2.best_estimator_ , X_train_rf1, y_train_rf1, curr_feat)
    )

print(
    f'feature importance of "{curr_feat}" on train set is '
    f"{np.mean(list_feature_importance):.3} "
    f"± {np.std(list_feature_importance):.3}"
)

#%%

def permutation_importance(model, X, y, n_repeats=10):
    """Calculate importance score for each feature."""

    importances = []
    for curr_feat in X.columns:
        list_feature_importance = []
        for n_round in range(n_repeats):
            list_feature_importance.append(
                get_feature_importance(model, X, y, curr_feat)
            )

        importances.append(list_feature_importance)

    return {
        "importances_mean": np.mean(importances, axis=1),
        "importances_std": np.std(importances, axis=1),
        "importances": importances,
    }


#%%
def plot_feature_importances(perm_importance_result, feat_name):
    """bar plot the feature importance"""

    fig, ax = plt.subplots()

    indices = perm_importance_result["importances_mean"].argsort()
    plt.barh(
        range(len(indices)),
        perm_importance_result["importances_mean"][indices],
        xerr=perm_importance_result["importances_std"][indices],
    )

    ax.set_yticks(range(len(indices)))
    _ = ax.set_yticklabels(feat_name[indices])


#%%

perm_importance_result_train = permutation_importance(
    random_search2.best_estimator_, X_train_rf1, y_train_rf1, n_repeats=10
)

plot_feature_importances(perm_importance_result_train, X_train.columns)

#%%
'''
=================================
Random Forest Model 3
    with synthetic data and tuned
=================================
'''
y_target = synthetic_data_gaussian_1['Results_G3']
X_features = synthetic_data_gaussian_1.drop(columns=['Results_G3'])


y_train_rf3 = pd.concat([y_train, y_target], axis=0).reset_index(drop=True)
X_train_rf3 = pd.concat([X_train, X_features], axis=0).reset_index(drop=True)

#%%
X_train_rf3 = X_train_rf3.drop(columns=['G1', 'G2'])
X_test = X_test.drop(columns=['G1', 'G2'])
#%%
print("X_train shape:", X_train_rf3.shape)
print("y_train shape:", y_train_rf3.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

#%%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

#%%


max_features = [ 'sqrt', 'log2', None]

rf_hyperparam_grid = {
    'n_estimators': [int(x) for x in range(50, 701, 50)],  # Explore a broad range
    'max_features': max_features,
}

#%%


# Perform Randomized Search
random_search = RandomizedSearchCV(estimator=rf, 
                                   param_distributions=rf_hyperparam_grid, 
                                   n_iter=200,  # Number of different combinations to try
                                   cv=5, 
                                   verbose=2, 
                                   random_state=42, 
                                   n_jobs=-1, 
                                   scoring='accuracy')

#%%
%%time
random_search.fit(X_train_rf3, y_train_rf3)

#%%
random_search.best_score_
# 0.699
#%%
rf_best = random_search.best_estimator_
rf_best

# max_features = log2
# n_estimators = 550


#%%
y_pred_proba3 = rf_best.predict_proba(X_test)[:,1]
y_pred3 = rf_best.predict(X_test)
#%%
# Log Loss berechnen
logloss = log_loss(y_test, y_pred_proba3)
print(f'Log Loss RFM synthetic data: {logloss}')
# 0.659

# AUC berechnen
auc = roc_auc_score(y_test, y_pred_proba3)
print(f'AUC RFM synthetic data: {auc}')
# 0.610

#%%
# Accuracy
accuracy3 = accuracy_score(y_test, y_pred3)
print('Accuracy score RFM synthetic data', accuracy3)
# 0.772
#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Confusion Matrix
cm = confusion_matrix(y_test ,y_pred3)
print("Confusion Matrix Random Forest 3 ", cm)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title('Confusion Matrix - Random Forest 3 (synthetic data)')
plt.show()

#%%
# --------------------------------------------
# Plotting the results of the RandomSearch CV ------

random_results = pd.DataFrame(random_search.cv_results_)

# Zeige die wichtigsten Spalten an
random_results = random_results[['param_n_estimators', 'param_max_features', 'mean_test_score']]

# Sortiere nach der Anzahl der Bäume
random_results = random_results.sort_values('param_n_estimators')
print(random_results)

random_results['param_max_features'] = random_results['param_max_features'].fillna('None')

plt.figure(figsize=(12, 8))

for max_feature in random_results['param_max_features'].unique():
    subset = random_results[random_results['param_max_features'] == max_feature]
    print(f"Plotting max_features={max_feature}:")
    print(subset)
    
    plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o', label=f'max_features={max_feature}')

plt.xlabel('Number of Estimators (n_estimators)', fontsize = 18)
plt.ylabel('Mean Test Accuracy', fontsize = 18)
plt.title('n_estimators vs. Accuracy for different max_features', fontsize = 20)

plt.ylim([0.600, 0.800])
plt.legend(title='max_features', fontsize = 18, title_fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

plt.show()


#%%
# ROC-Curve for the RFM with synthetic data

naiive_fpr_syn, naiive_tpr_syn, naiive_thresholds_syn = roc_curve(y_true=y_test, y_score = y_pred_proba3)

syn_auc = roc_auc_score(y_test, y_pred_proba3)

plt.figure(figsize=(8, 6))
plt.plot(naiive_fpr_syn, naiive_tpr_syn, label='Tuned Random Forest ROC (synthetic data)')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve Tuned Random Forest Model 3 (sythetic data)')
plt.show()


print(f"Calculated AUC: {syn_auc:.3f}")






#%%
# ===================================================================================================
# Random Forest Classifier ----
# with principal component analysis
# ------------------------------

#y = data_processed_math['Results_G3']
#x = data_processed_math.drop(columns =[ 'Results_G3'])
#x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
# stratify = y => to ensure that both the train and test sets have the same proportion of 0s and
# 1s as the original dataset
#%%
#print(y.shape) #(395,)
#print(x.shape) #(395, 50)
#print(x_train3.shape) #(276, 50)
#print(x_test3.shape) #(119, 50)
#print(y_train3.shape) #(276,)
#print(y_test3.shape) # (119,)
#%%
# Scaling the data
# control for the fact, that different variabnles are measured on different scales
#from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()
#X_train_scaled = ss.fit_transform(x_train3)
#X_test_scaled = ss.transform(x_test3)
#y_train = np.array(y_train3)
#%%
#print(X_train_scaled.shape)
#print(X_test_scaled.shape)
#print(y_train.shape)

#%%
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import recall_score
#rfc_pca = RandomForestClassifier(random_state = 42)
#rfc_pca.fit(X_train_scaled, y_train)

#%%
# Feature importance ----
#import seaborn as sns
#feats = {}
#for feature, importance in zip(data_processed_math.columns, rfc_pca.feature_importances_):
#    feats[feature] = importance
#importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
#importances = importances.sort_values(by='Gini-Importance', ascending=False)
#importances = importances.reset_index()
#importances = importances.rename(columns={'index': 'Features'})
#sns.set(font_scale = 5)
#sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
#fig, ax = plt.subplots()
#fig.set_size_inches(30,15)
#sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
#plt.xlabel('Importance', fontsize=25, weight = 'bold')
#plt.ylabel('Features', fontsize=25, weight = 'bold')
#plt.title('Feature Importance', fontsize=25, weight = 'bold')
#display(plt.show())
#display(importances)


#%%
#%%
#%%
# ====================================================================================================
# Random Forest
# Dealing with the imbalanced data set
# Random Forest algorithm based on sampling with replacement
# Extract multiple example subsets randomly with replacement from the majority class and the 
# example number of extracted example subsets is the same with minority class example dataset
# The multiple new training datasets were constructed by combining the each extracted majority example
# subset and minority class dataset respectivley, and multiple random forest classifiers were trained 
# on these training datasets
# ==> The paper shows that the improved random forest algorithm outperfomed original random forest and
#       other methods and coulde deal with imbalanced data 



#%%
from sklearn.utils import resample
# %%
y = data_processed_math['Results_G3']
x = data_processed_math.drop(columns =[ 'Results_G3'])
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#%%
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
#%%
minority_class = x_train[y_train == 0].reset_index(drop=True)
majority_class = x_train[y_train == 1].reset_index(drop=True)
y_minority = y_train[y_train == 0].reset_index(drop=True)
y_majority = y_train[y_train == 1].reset_index(drop=True)

#%%
print(minority_class.shape) 
# (84, 50)  => is the feature set which belongs to the minority class so, all students which failed

print(majority_class.shape)
# (192, 50) => is the feature set which belongs to the majority class so, all students which passed

print(y_minority.shape)
# (84,) => target => all 0 => failed

print(y_majority.shape)
# (192, ) => target => all 1 => passed







# =================================================================================================
# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Hyperparameter für die Random Forest Modelle
#best_params = {
#    'n_estimators':500,
#    'min_samples_split': 50,
#    'min_samples_leaf': 5,
#    'max_features': None,
#    'max_depth': 10,
#    'criterion': 'gini'
#}

best_params = {
    'n_estimators':200,
    'max_features': None,
}

# Data setup
y = data_processed_math['Results_G3']
X = data_processed_math.drop(columns=['Results_G3', 'G1', 'G2', 'G3'])

# Define the number of folds for cross-validation
n_folds = 5
n_classifiers = 10

# Initialize lists to store evaluation metrics
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []
specificity_scores = []
auc_scores = []
log_loss_scores = []
all_fprs = []
all_tprs = []
all_thresholds = []
classifier_thresholds = []
fold_thresholds = [] 

fold_fprs = []  # FPRs for current classifier
fold_tprs = []

mean_fpr = np.linspace(0, 1, 100)

# StratifiedKFold für Cross-Validation
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Trainiere 10 Modelle mit den besten Hyperparametern
for clf_idx in range(n_classifiers):
    print(f'Training classifier {clf_idx + 1}/{n_classifiers}')
    
    fold_precision_scores = []
    fold_recall_scores = []
    fold_f1_scores = []
    fold_accuracy_scores = []
    fold_specificity_scores = []
    fold_auc_scores = []
    fold_log_loss_scores = []
    
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'Fold {fold_idx + 1}/{n_folds} for classifier {clf_idx + 1}')
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Trainiere den Random Forest mit den besten Hyperparametern
        clf = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_features=best_params['max_features'],
            #max_depth=best_params['max_depth'],
            #min_samples_split=best_params['min_samples_split'],
            #min_samples_leaf=best_params['min_samples_leaf'],
            #criterion=best_params['criterion'],
            random_state=clf_idx
        )
        clf.fit(X_train, y_train)
        
        # Vorhersage auf dem Testset
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]


        
        # Berechne Metriken
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_test, y_prob)
        log_loss_val = log_loss(y_test, y_prob)
        
        fold_precision_scores.append(precision)
        fold_recall_scores.append(recall)
        fold_f1_scores.append(f1)
        fold_accuracy_scores.append(accuracy)
        fold_specificity_scores.append(specificity)
        fold_auc_scores.append(auc)
        fold_log_loss_scores.append(log_loss_val)

        fold_thresholds.append(thresholds)

        fold_fprs.append(fpr)
        fold_tprs.append(tpr)

    classifier_thresholds.append(fold_thresholds)
    # Durchschnittliche Metriken über alle Folds
    precision_scores.append(np.mean(fold_precision_scores))
    recall_scores.append(np.mean(fold_recall_scores))
    f1_scores.append(np.mean(fold_f1_scores))
    accuracy_scores.append(np.mean(fold_accuracy_scores))
    specificity_scores.append(np.mean(fold_specificity_scores))
    auc_scores.append(np.mean(fold_auc_scores))
    log_loss_scores.append(np.mean(fold_log_loss_scores))
    
    avg_fpr = np.mean(fold_fprs, axis=0)
    avg_tpr = np.mean(fold_tprs, axis=0)
    
    # Store average FPR and TPR
    all_fprs.append(avg_fpr)
    all_tprs.append(avg_tpr)


# Durchschnittliche Metriken über alle Classifier
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_accuracy = np.mean(accuracy_scores)
avg_specificity = np.mean(specificity_scores)
avg_auc = np.mean(auc_scores)
avg_log_loss = np.mean(log_loss_scores)

# Ausgabe der durchschnittlichen Metriken
print(f'Average Precision across {n_classifiers} classifiers: {avg_precision:.4f}')
print(f'Average Recall across {n_classifiers} classifiers: {avg_recall:.4f}')
#print(f'Average F1 Score across {n_classifiers} classifiers: {avg_f1:.4f}')
print(f'Average Accuracy across {n_classifiers} classifiers: {avg_accuracy:.4f}')
print(f'Average Specificity across {n_classifiers} classifiers: {avg_specificity:.4f}')
print(f'Average AUC across {n_classifiers} classifiers: {avg_auc:.4f}')
print(f'Average Log Loss across {n_classifiers} classifiers: {avg_log_loss:.4f}')

mean_thresholds = np.mean(classifier_thresholds, axis=0)

#%%
mean_thresholds

#%%
plt.figure(figsize=(8, 6))
plt.plot(all_fprs[-1], all_tprs[-1], color='b', label=f'Mean ROC Curve (AUC = {avg_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve across all classifiers and folds')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Data setup
y = data_processed_math['Results_G3']
X = data_processed_math.drop(columns=['Results_G3', 'G1', 'G2', 'G3'])

# Hyperparameters
best_params = {
    'n_estimators': 200,
    'max_features': None,
}

# Define number of folds and classifiers
n_folds = 5
n_classifiers = 10

# Evaluation metrics
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []
specificity_scores = []
auc_scores = []
log_loss_scores = []

# ROC curve data for averaging
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(mean_fpr)

# Stratified K-Folds cross-validator
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Train multiple classifiers and store results
for clf_idx in range(n_classifiers):
    print(f'Training classifier {clf_idx + 1}/{n_classifiers}')
    
    fold_precision_scores = []
    fold_recall_scores = []
    fold_f1_scores = []
    fold_accuracy_scores = []
    fold_specificity_scores = []
    fold_auc_scores = []
    fold_log_loss_scores = []

    # Store fold-wise ROC curve data
    fold_fprs = []
    fold_tprs = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f'Fold {fold_idx + 1}/{n_folds} for classifier {clf_idx + 1}')
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_features=best_params['max_features'],
            random_state=clf_idx
        )
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)


        all_true_labels.extend(y_test)
        all_predicted_labels.extend(y_pred)

        # Interpolate TPR values to a common FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)
        fold_tprs.append(interp_tpr)  # Store TPR for this fold

        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        log_loss_val = log_loss(y_test, y_prob)

        # Store metrics
        fold_precision_scores.append(precision)
        fold_recall_scores.append(recall)
        fold_f1_scores.append(f1)
        fold_accuracy_scores.append(accuracy)
        fold_specificity_scores.append(specificity)
        fold_auc_scores.append(auc)
        fold_log_loss_scores.append(log_loss_val)

    # Average the TPRs across folds for this classifier
    mean_tpr += np.mean(fold_tprs, axis=0)

    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

    # Display the confusion matrix
    print("Aggregated Confusion Matrix across all classifiers and folds:")

    # Average the metrics over folds
    precision_scores.append(np.mean(fold_precision_scores))
    recall_scores.append(np.mean(fold_recall_scores))
    f1_scores.append(np.mean(fold_f1_scores))
    accuracy_scores.append(np.mean(fold_accuracy_scores))
    specificity_scores.append(np.mean(fold_specificity_scores))
    auc_scores.append(np.mean(fold_auc_scores))
    log_loss_scores.append(np.mean(fold_log_loss_scores))

# Average the TPR over all classifiers
mean_tpr /= n_classifiers
mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)

# Calculate mean AUC across classifiers
avg_auc = np.mean(auc_scores)

# Plot Mean ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC Curve (AUC = {avg_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Mean ROC Curve across all classifiers and folds', fontsize=16)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Print average metrics
print(f'Average Precision: {np.mean(precision_scores):.4f}')
print(f'Average Recall: {np.mean(recall_scores):.4f}')
print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
print(f'Average Accuracy: {np.mean(accuracy_scores):.4f}')
print(f'Average Specificity: {np.mean(specificity_scores):.4f}')
print(f'Average AUC: {avg_auc:.4f}')
print(f'Average Log Loss: {np.mean(log_loss_scores):.4f}')








#%%
# ALL ROC from all RFM together
plt.figure(figsize=(13, 9))
plt.plot(naiive_fpr, naiive_tpr, label='RFM without tuning ')
plt.plot(naiive_fpr_tuned, naiive_tpr_tuned, label='Tuned RFM ')
plt.plot(naiive_fpr_syn, naiive_tpr_syn, label='RFM with synthetic data ')
plt.plot(naiive_fpr_all, naiive_tpr_all, label='RFM including G1,G2 ')
#plt.plot(mean_fpr, mean_tpr, color='black', label=f'Improved RFM')
#plt.plot(all_fprs[-1], all_tprs[-1], color='black', label=f'Improved RFM')
#plt.plot(all_fprs[-1], all_tprs[-1], color='black', label=f'Improved RFM')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC AUC Curve different Random Forest Models', fontsize = 20)
plt.legend(loc='lower right', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()


#%%
# Plotting all Decision Trees together
plt.figure(figsize=(13, 9))
plt.plot(grid_fpr_decison_tree1, grid_tpr_decision_tree1, label='Decision Tree 1 (including G1, G2)')
plt.plot(grid_fpr_decison_tree2, grid_tpr_decision_tree2, label='Decision Tree 2 (excluding G1, G2)')
plt.plot(grid_fpr_decison_tree3, grid_tpr_decision_tree3, label='Decision Tree 3 (synthetic data)')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC AUC different Decision Trees', fontsize = 20)
plt.legend(loc='lower right', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()







#%%

# Decision Tree
grid_fpr_decison_tree, grid_tpr_decision_tree, grid_thresholds_decision_tree = roc_curve(y_true = y_test_real, y_score=y_scores_decsiontree)

import matplotlib.pyplot as plt
plt.plot(grid_fpr_decison_tree, grid_tpr_decision_tree, label='Decision Tree ROC AUC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve For Random Forest Model trained on synthetic data')
plt.show()
#%%
# Naiive Model
naiive_fpr, naiive_tpr, naiive_thresholds = roc_curve(y_true=y_test, y_score=naiive_y_hat[:,1])

plt.plot(naiive_fpr, naiive_tpr, label='Naive ROC AUC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve For Naiive Random Forest Model')
plt.show()

#%%

# Grid Search Model
grid_fpr_tunedRFM, grid_tpr_tunedRFM, grid_thresholds_tunedRFM = roc_curve(y_true=y_test, y_score=grid_y_hat[:,1])

plt.plot(naiive_fpr, naiive_tpr, label='Naive ROC AUC')
plt.plot(grid_fpr_tunedRFM, grid_tpr_tunedRFM, label='Grid Search ROC AUC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve For Random Forest Model with Grid vs Naiive Model')
plt.title('ROC AUC Curve For Random Forest Model with Grid Search')
plt.show()

#%%

# ROC -----
# Grid Search Model 2
grid_fpr_2, grid_tpr_2, grid_thresholds_2 = roc_curve(y_true = y_test_syn, y_score=grid_y_hat_synch[:,1])

import matplotlib.pyplot as plt
plt.plot(grid_fpr_2, grid_tpr_2, label='Grid Search ROC AUC')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC AUC Curve For Random Forest Model trained on synthetic data')
plt.show()
# %%
plt.figure(figsize=(8, 6))
plt.plot(grid_fpr_decison_tree, grid_tpr_decision_tree, label='Decision Tree ROC AUC')
plt.plot(naiive_fpr, naiive_tpr, label='Naive RFM ROC AUC')
plt.plot(grid_fpr_tunedRFM, grid_tpr_tunedRFM, label='Tuned RFM Random Search ROC AUC')
plt.plot(grid_fpr_2, grid_tpr_2, label='RFM with synthetic data ROC AUC')
plt.plot(all_fprs[-1], all_tprs[-1], color='black', label=f'Mean ROC Curve across all classifiers and folds')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC From Different Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
# %%
