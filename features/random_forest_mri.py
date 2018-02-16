
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')

import numpy as np
import sklearn
import scipy.io as sci
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

mat_features = sci.loadmat('data/PGSE_features.mat')
mat_noisy_features = sci.loadmat('data/PGSE_features_SNR40.mat')
mat_params = sci.loadmat('data/PGSE_params.mat')

features = mat_features['PGSE_features']
noisy_features = mat_noisy_features['PGSE_features_SNR40']
params = mat_params['PGSE_params']

np_features = np.array(features)
np_noisy_features = np.array(noisy_features)
np_params = np.array(params)

np_params = np.delete(np_params, [1,2,5], 1) # To remove the undesired parameters
for row in np_params:
    row[2] = row[2] * 10**9 # To standardise the units of the data - 10^9 seems to work best

labels = ['D1', 'D2', 'D3', 'D4', 'D5', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
no_of_shells = 16


# In[2]:


X_train, X_test, y_train, y_test = train_test_split(np_features, np_params, test_size=0.2, random_state=0)


# In[7]:


plt.figure(1, figsize=(16, 5))

plt.subplot(131)
plt.hist(y_train[:, 0])
plt.hist(y_test[:, 0])
plt.title("Volume Fraction")

plt.subplot(132)
plt.hist(y_train[:, 1])
plt.hist(y_test[:, 1])
plt.title("Intracellular Exchange Time")

plt.subplot(133)
plt.hist(y_train[:, 2])
plt.hist(y_test[:, 2])
plt.title("Diffusivity")

plt.savefig('results/split_distribution.png')
plt.show()


# The graphs above show that the train_test_split function in the scikit-learn module produce a split in the data that keeps the distribution of the parameters consistent.

# In[8]:


rfr = RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', random_state=0).fit(X_train, y_train)
y_predicted = rfr.predict(X_test)


# In[9]:


parameters = ["Volume Fraction", "Intracellular Exchange Time", "Diffusivity"]
separate_scores = []

for p in range(3):
    score = pearsonr(y_test[:, p], y_predicted[:, p])[0].astype(str)
    separate_scores.append(score)
    print(parameters[p] + ': ' + score)


# In[11]:


plt.figure(2, figsize=(16, 5))

plt.subplot(131)
plt.scatter(y_test[:, 0], y_predicted[:, 0])
plt.plot(plt.xlim(), plt.ylim(), '--', c='0.5')
plt.ylabel('Predictions')
plt.xlabel('True values')
plt.text(0.4, 0.7, r'$R^2=$'+separate_scores[0][0:5], fontsize=15)
plt.title("Volume Fraction")
plt.grid(True)

plt.subplot(132)
plt.scatter(y_test[:, 1], y_predicted[:, 1])
plt.plot(plt.xlim(), plt.ylim(), '--', c='0.5')
plt.ylabel('Predictions')
plt.xlabel('True values')
plt.text(0, 0.66, r'$R^2=$'+separate_scores[1][0:5], fontsize=15)
plt.title("Intracellular Exchange Time")
plt.grid(True)

plt.subplot(133)
plt.scatter(y_test[:, 2], y_predicted[:, 2])
plt.plot(plt.xlim(), plt.ylim(), '--', c='0.5')
plt.ylabel('Predictions')
plt.xlabel('True values')
plt.text(0.75, 2.1, r'$R^2=$'+separate_scores[2][0:5], fontsize=15)
plt.title("Diffusivity")
plt.grid(True)

plt.savefig('results/noisefree_precision.png')
plt.show()


# The graphs above show the precision of the predictions made by the Random Forest Regressor according to the true value of the parameters. The model predicts Diffusivity most accurately. Intracellular Exchange Time is predicted well at lower values but begins being underestimated as values increase. 

# This graph shows how much each feature in the dataset contributes to the predictions that the model makes. It would be useful to run multiple splits of the data and train as many models to see if the split of data has any significant impact on the importance of the features.

# In[12]:


all_feature_imp = rfr.feature_importances_

final_feature_imp = []

for i in range(15):
    temp = 0

    for j in range(no_of_shells):
        temp += all_feature_imp[i + 15 * j]

    final_feature_imp.append(temp)

order = sorted(range(len(final_feature_imp)), reverse=True, key=final_feature_imp.__getitem__) # sorted order of importances by index

sorted_final_feature_imp = sorted(final_feature_imp, reverse=True)

ordered_labels = [None] * len(labels)

for k in range(len(labels)): # sorts the labels in order of the importances
    ordered_labels[k] = labels[order[k]]


# In[13]:


plt.figure(3, figsize=(16, 5))
plt.bar(range(len(sorted_final_feature_imp)), sorted_final_feature_imp, tick_label=ordered_labels)
plt.xlabel("Features")
plt.ylabel("Contribution")
plt.title("Feature Importances")
plt.savefig('results/noisefree_importances.png')
plt.show()


# In[17]:


reversed_order = order[::-1]
scores = []

for i in range(-1, len(reversed_order)):
    feature = []
    
    if i != -1:
        for j in range(no_of_shells):
            feature.append(reversed_order[i] + 15*j)
    
    X_temp_train = np.delete(X_train, feature, 1)
    X_temp_test = np.delete(X_test, feature, 1)
    rfr = RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', random_state=0).fit(X_temp_train, y_train)
    predictions = rfr.predict((X_temp_test))
    current_scores = []
    
    for r in range(3):
        current_scores.append(pearsonr(y_test[:, r], predictions[:, r])[0])
    
    scores.append(current_scores)


# In[19]:


reversed_labels = ordered_labels[::-1]
np_scores = np.array(scores)
plt.figure(4, figsize=(15, 25))
 
plt.subplot(311)
plt.plot(np_scores[1:, 0])
plt.plot((0,14), (np_scores[0, 0], np_scores[0, 0]), '--', c='0.3')
plt.xticks(range(15), reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Volume Fraction")

plt.subplot(312)
plt.plot(np_scores[1:, 1])
plt.plot((0,14), (np_scores[0, 1], np_scores[0, 1]), '--', c='0.3')
plt.xticks(range(15), reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Intracellular Exchange Time")

plt.subplot(313)
plt.plot(np_scores[1:, 2])
plt.plot((0,14), (np_scores[0, 2], np_scores[0, 2]), '--', c='0.3')
plt.xticks(range(15), reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Diffusivity")

plt.savefig('results/noisefree_individual_feature_removal.png')
plt.show()


# Individual feature removal trains 15 models with 14 features where each feature in the original dataset is removed once. This shows the effect that each individual feature has on the score of the model.
# It could be useful to observe which features, when removed, result in an increase in score for all three parameters - like S3. These could be targets for permanent removal although their relationship with other features may have to be considered. If there are other features with similar information there is redundancy and so can handle feature removal, however if all similar features are removed, you are losing information.

# In[20]:


original_0 = scores[0][0]
original_1 = scores[0][1]
original_2 = scores[0][2]
score_difference = []

for i in range(1, len(scores)):
    temp_difference = []
    temp_difference.append(original_0 - scores[i][0])
    temp_difference.append(original_1 - scores[i][1])
    temp_difference.append(original_2 - scores[i][2])
    score_difference.append(temp_difference);


# In[21]:


np_difference = np.array(score_difference)

plt.figure(5, figsize=(16, 5))

plt.plot(np_difference[:, 0], label='Volume Fraction')
plt.plot(np_difference[:, 1], label='Intracellular Exchange Time')
plt.plot(np_difference[:, 2], label='Diffusivity')
plt.legend()

plt.xticks(range(15), reversed_labels)
plt.xlabel("Features")
plt.ylabel("Decrease in score")

plt.savefig('results/noisefree_individual_score_change.png')
plt.show()


# This graph shows how much the score of the model changes when each individual feature is removed. If the y axis shows a negative change that means that when that feature was removed, the score of the model actually increased. Need to find a way to represent this so you can better see changes in Volume Fraction (?).

# In[13]:


con_scores = []
removed_features = []

for i in range(-1, len(reversed_order)-1):
    if i != -1:
        for j in range(no_of_shells):
            removed_features.append(reversed_order[i] + 15*j)
        
    X_temp_train = np.delete(X_train, removed_features, 1)
    X_temp_test = np.delete(X_test, removed_features, 1)
    rfr = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=0).fit(X_temp_train, y_train)
    predictions = rfr.predict((X_temp_test))
    current_scores = []
    
    for r in range(3):
        current_scores.append(pearsonr(y_test[:, r], predictions[:, r])[0])
    
    con_scores.append(current_scores)


# In[33]:


np_con_scores = np.array(con_scores)

plt.figure(6, figsize=(15, 25))

plt.subplot(311)
plt.plot(np_con_scores[:, 0].tolist() + [0])
plt.xticks(range(16), ['Original'] + reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Volume Fraction")

plt.subplot(312)
plt.plot(np_con_scores[:, 1].tolist() + [0])
plt.xticks(range(16), ['Original'] + reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Intracellular Exchange Time")

plt.subplot(313)
plt.plot(np_con_scores[:, 2].tolist() + [0])
plt.xticks(range(16), ['Original'] + reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Diffusivity")

plt.savefig('results/cumulative_feature_removal.png')
plt.show()


# Sequential feature removal trains 14 models each with a decreasing amount of features by removing one more feature per model as per the reverse order of the feature importance graphed earlier.

# In[15]:


con_scores = []
removed_features = []

for i in range(-1, len(order)-1):
    if i != -1:
        for j in range(no_of_shells):
            removed_features.append((14-i) + 15*j)
        
    X_temp_train = np.delete(X_train, removed_features, 1)
    X_temp_test = np.delete(X_test, removed_features, 1)
    rfr = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=0).fit(X_temp_train, y_train)
    predictions = rfr.predict((X_temp_test))
    current_scores = []
    
    for r in range(3):
        current_scores.append(pearsonr(y_test[:, r], predictions[:, r])[0])
    
    con_scores.append(current_scores)


# In[34]:


np_con_scores = np.array(con_scores)
reversed_labels = labels[::-1]

plt.figure(6, figsize=(15, 25))

plt.subplot(311)
plt.plot(np_con_scores[:, 0].tolist() + [0])
plt.xticks(range(16), ['Original'] + reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Volume Fraction")

plt.subplot(312)
plt.plot(np_con_scores[:, 1].tolist() + [0])
plt.xticks(range(16), ['Original'] + reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Intracellular Exchange Time")

plt.subplot(313)
plt.plot(np_con_scores[:, 2].tolist() + [0])
plt.xticks(range(16), ['Original'] + reversed_labels)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Diffusivity")

plt.savefig('results/cumulative_feature_removal_inorder.png')
plt.show()

