"""This code helps train two variations of a SAN model on the raw traces of the ASCAD dataset.
The reults are saved as '.npy' files and are plotted using the matplotlib library"""

#Importing all neccessary libraries
import san
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif  # chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import tqdm

################################################
############## EXTRACTING DATA #################

#utils.dataExt()                        #Uncomment this line to generate traces and labels in numpy format

# Loading the '.npy' files that were generated by the dataExt() function in utils 
raw_traces=np.load('traces.npy')        #Loading raw traces
Profiling_labels=np.load('labels.npy')  #Loading labels

train_traces,train_labels=utils.dataBalance(raw_traces,Profiling_labels) #Generating a balanced dataset

#Setting x and y for training
x = train_traces
y = train_labels

################################################

#Setting all parameters as per the variabless specified in the utils file
clf = san.SAN(num_epochs=utils.EPOCHS, num_heads=utils.ATTENTION_HEADS, batch_size=utils.BATCH_SIZE, dropout=utils.DROPOUT_RATE, hidden_layer_size=utils.HIDDEN_LAYER_SIZE, stopping_crit = utils.CRITICAL_STOPPING, learning_rate = utils.LEARNING_RATE,baseline=utils.BASELINE)
kf = KFold(n_splits=utils.CROSS_VALIDATION_FOLDS)

accuracy_results = []

#Looping through the various folds for K-fold cross validation
for train_index, test_index in kf.split(x):
    train_x = x[train_index]
    test_x = x[test_index]
    train_y = y[train_index]
    test_y = y[test_index]
    x_sp = sparse.csr_matrix(train_x)
    xt_sp = sparse.csr_matrix(test_x)

    clf.fit(x_sp, train_y)               #Fitting the model
    predictions = clf.predict(xt_sp)
    score = accuracy_score(predictions, test_y)
    accuracy_results.append(score)
    
print("Accuracy (ASCAD Dataset) {} ({})".format(np.mean(accuracy_results), np.std(accuracy_results)))  #Printing the accuracy of the 

#Calculating the mean attention from the final weight matrix of the attention layer
global_attention_weights = clf.get_mean_attention_weights()
np.save('global_exp.npy',global_attention_weights)
utils.drawGraph(global_attention_weights)
###################################################################################

#Calculating the instance based attention 
local_attention_matrix = clf.get_instance_attention(x)
np.save('local_exp.npy',local_attention_matrix)
###################################################################################

