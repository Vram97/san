"""This file is the utilities file that holds many of the functions that were used to aid this project.
This also contains the tunable parameters for the models that were trained"""

#Importing all essential libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

##########################################################################
################ TUNABLE HYPER PARAMETERS ################################

EPOCHS=8                  #The number of times the model is trained on the same data. It is set to 40 for the MLP model owing to the increase in learnable parameters.
BATCH_SIZE=20             #The number of traces within a single batch
CRITICAL_STOPPING=10      #The number of iterations upto which an increase in the loss will be tolerated
LEARNING_RATE=0.02        #Speed of learning
CROSS_VALIDATION_FOLDS=10 #This determines how much of the data goes for training and testing. Generally done when there is a dearth in data.

##########################################################################
################ TUNABLE MODEL PARAMETRS #################################

ATTENTION_HEADS=5         # The value of 5 was chosen to keep the computational complexity low. However, it can be increased in order for the model to get a better idea of the features.
DROPOUT_RATE=0.2          # Combats over-fitting.
HIDDEN_LAYER_SIZE=2       # This determines the number of neurons in the layer after the attention layer. Keeping it below 5, delivers good results while keeping the computational complexity low.
BASELINE= False           #This parameter will determine whether to run the baseline SAN model or the MLP one.

##########################################################################
################ DATA EXTRACTION PARAMETERS ##############################

TRACES_LOWER=0
TRACES_UPPER=15000
FEATURES_LOWER=44000
FEATURES_UPPER=47000
FINAL_FEATURE_EXTRACT=700   # This was set at 700 since [45400,46100] is the popularly accepted window for training models.
PATH_TO_ASCAD= 'ASCAD.h5'   #This specifies the path to the ASCAD dataset on the user's system
PATH_TO_RAW='ATMega8515_raw_traces.h5' #This specifies the path to the raw traces

##########################################################################
############### FIXED PARAMETERS #########################################

#These values were obtained from the literature on DL based SCA
FEATURE_WINDOW_LOWER=45400
FEATURE_WINDOW_UPPER=46100
RAW_FEATURES=100000

##############################################################################
##################### DATA EXTRACTION ########################################

# This function helps extract all relevant information from the ASCAD dataset. It finally saves the [TRACES_LOWER,TRACES_UPPER] feature as '.npy' files for ease of extraction. 
def dataExt(path=PATH_TO_ASCAD,path2=PATH_TO_RAW):

    with h5py.File(path, 'r') as hdf:
    
    ############################################################################
        #Extracting the profiling traces
        data_profiling = hdf.get('Profiling_traces')
        Profiling_traces = np.array(data_profiling.get('traces'))
        
        #Extracting the attack traces
        data_attack = hdf.get('Attack_traces')
        Attack_traces = np.array(data_attack.get('traces'))
        
        # Items
        Attack_Traces_items = list(data_attack.items())
        Profiling_Traces_items = list(data_profiling.items())

        # This gives the labels for profiling and attack
        Profiling_labels = np.array(data_profiling.get('labels')).reshape((-1,1))[TRACES_LOWER:TRACES_UPPER]
        Attack_labels = np.array(data_attack.get('labels'))

        # Metadata
        Profiling_metadata = np.array(data_profiling.get('metadata'))
        Attack_metadata = np.array(data_attack.get('metadata'))

    ###############################################################################

    #Data extraction for raw traces
    with h5py.File(path2, 'r') as hdf:
        ls_raw = list(hdf.keys())
        raw_traces=hdf['traces'][TRACES_LOWER:TRACES_UPPER]   #Extracting raw traces
    
    #Saving raw traces and labels
    np.save('traces.npy',raw_traces)
    np.save('labels.npy',Profiling_labels)


#This function helps balance the dataset i.e it equalizes the number of traces per key value.
def dataBalance(traces,labels):

  unq_keys,counts=np.unique(labels,return_counts=True)  #Calculates all the unique labels and their frquencies in the dataset
  length=unq_keys.shape[0]                              #The number of unique labels in the dataset
  min_counts=min(counts)                                #Calculates the most infrequent key

  data_full=np.concatenate((traces,labels),axis=1)      #Concatenating traces and labels

  #Restructuring the dataset with only as many traces as 'min_counts' per key
  indices=np.array([])                                                          #Array that stores all indices of the traces that qualify in the balanced dataset
  
  #Looping through all traces in the dataset
  for i in range(length):
      temp=np.array(np.where(data_full[:,RAW_FEATURES]==unq_keys[i])[0][0:min_counts])  #Finding indices
      indices=np.append(indices,temp)
  indices=indices.reshape((-1,1)).astype(int)

  sorted_data=data_full[indices].reshape((-1,RAW_FEATURES+1))                           #Sorted dataset

  train_traces=sorted_data[:,FEATURES_LOWER:FEATURES_UPPER]                             #All traces
  train_labels=sorted_data[:,RAW_FEATURES]                                              #All labels

  return train_traces,train_labels

##########################################################################
################# PLOTTING FUNCTIONS #####################################
"""These functions help visualize the spread of the importance of various features within the traces, after training"""

#This function first calculates the top 'FINAL_FEATURE_EXTRACT' features and then splits those features into the ones lying between the [45400,46100] range and outside it.
def mostImportant(arr,features):
  temp=arr.argsort()[-features:][::-1]  #Calculating top features
  points=np.array([],dtype=int)         #Array that holds all points outside the specified window
  window_points=np.array([],dtype=int)  #Array that holds all points inside the specified window

  #Looping through all top features
  for i in temp:
    if(i>=abs(FEATURE_WINDOW_LOWER-FEATURES_LOWER)-1 and i<abs(FEATURE_WINDOW_UPPER-FEATURES_LOWER)):
      window_points=np.append(window_points,i)
    else:
      points=np.append(points,i)
  return points,window_points


# This function draws the scatter plot of the importance of all the features and marks them as red or blue depending on where they belong(Inside or Outside [45400,46100] window)
def drawGraph(att_mat,features=FINAL_FEATURE_EXTRACT):
  x,x_window=mostImportant(att_mat,features)                                      #Call to most important function
  print(f"Only {x_window.shape[0]} points lie within the [45400,46100] window")   #This print statement tells you how many of the features lie in the [45400,46100] window
  
  #Calculating y values of all graphs
  y=att_mat[x]
  y_window=att_mat[x_window]

  #Plotting the features
  plt.figure(figsize=(16,10))
  plt.xlabel('Features')
  plt.ylabel('Importance')
  plt.scatter(x+FEATURES_LOWER,y)                              #Plotting features outside the window
  plt.scatter(x_window+FEATURES_LOWER,y_window,color='red')    #Plotting features inside window
  plt.legend(['Points outside window','Points within window'])
  plt.show()

#############################################################################################################################################