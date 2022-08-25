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

##########################################################################
############### RESULTS ##################################################

#These values were obtained from all the experiments carried out during the DR and served as inputs to the plotting functions
#Baseline model with 2 hidden layer size
BASELINE_2_ACCEPTED_WINDOW_POINTS=161
BASELINE_2_MAX_WINDOW_POINTS=182
BASELINE_2_MAX_WINDOW=[43999,44699]
BASELINE_2_BEST_WINDOW=[44931,45631]

#Baseline model with 1 hidden layer size
BASELINE_1_ACCEPTED_WINDOW_POINTS=167
BASELINE_1_MAX_WINDOW_POINTS=177
BASELINE_1_MAX_WINDOW=[45972,46672]
BASELINE_1_BEST_WINDOW=[45858,46558]

#MLP model with 2 hidden layer size
MLP_2_ACCEPTED_WINDOW_POINTS=174
MLP_2_MAX_WINDOW_POINTS=179
MLP_2_MAX_WINDOW=[45471,46171]
MLP_2_BEST_WINDOW=[44969,45669]

#MLP model with 1 hidden layer size
MLP_1_ACCEPTED_WINDOW_POINTS=155
MLP_1_MAX_WINDOW_POINTS=174
MLP_1_MAX_WINDOW=[44355,45055]
MLP_1_BEST_WINDOW=[45048,45748]

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


#This function does exactly what the above function does but has a customizable window input. This function was used to make subplots.
def mostImportant_window(arr,features,window):
  temp=arr.argsort()[-features:][::-1]          #Calculating top features
  points=np.array([],dtype=int)                 #Array that holds all points outside the specified window
  window_points=np.array([],dtype=int)          #Array that holds all points inside the specified window

  #Looping through all top features
  for i in temp:
    if(i>=window[0] and i<window[1]):
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

#This function draws the sub-plots for all the models. It takes the values from the results as inputs
def drawSubplots(att_mat,features=FINAL_FEATURE_EXTRACT):
  
  x2,x2_window=mostImportant_window(att_mat,features,[FEATURE_WINDOW_LOWER-FEATURES_LOWER+1,FEATURE_WINDOW_UPPER-FEATURES_LOWER+1])      #Calculates the feature indices for the [45400,46100] window
  x3,x3_window=mostImportant_window(att_mat,features,[MLP_2_MAX_WINDOW[0]-FEATURES_LOWER+1,MLP_2_MAX_WINDOW[1]-FEATURES_LOWER+1])        #Calculates the feature indices for the window with the maximum features
  x4,x4_window=mostImportant_window(att_mat,features,[MLP_2_BEST_WINDOW[0]-FEATURES_LOWER+1,MLP_2_BEST_WINDOW[1]-FEATURES_LOWER+1])                                                                    #Calculates the feature indices for the window with the best cumulative importance
 
  #Calculating x and y values for all features within the [FEATURES_LOWER,FEATURES_UPPER] window
  x=np.argsort(att_mat)[-1:-700:-1]
  y=att_mat[x]

  #Calculating importance values for the corresponding feature indices
  y2=att_mat[x2]
  y2_window=att_mat[x2_window]
  y3=att_mat[x3]
  y3_window=att_mat[x3_window]
  y4=att_mat[x4]
  y4_window=att_mat[x4_window]

  #Plotting sub-plots
  plt.figure(figsize=(16,10))
  plt.xlabel('Features')
  plt.ylabel('Importance')

  #Overall sub-plot
  plt.subplot(2,2,1)
  plt.scatter(x+44000,y)
  plt.xlabel('Features')
  plt.ylabel('Importance')

  #Plotting sub-plot for accepted window
  plt.subplot(2,2,2)
  plt.scatter(x2+FEATURES_LOWER,y2)
  plt.scatter(x2_window+FEATURES_LOWER,y2_window,color='red')
  plt.xlabel('Features')
  plt.ylabel('Importance')
  plt.legend(['Features outside window','Features within [45400,46100]'])

  #Plotting sub-plot for window with maximum features
  plt.subplot(2,2,3)
  plt.scatter(x3+FEATURES_LOWER,y3)
  plt.scatter(x3_window+FEATURES_LOWER,y3_window,color='green')
  plt.xlabel('Features')
  plt.ylabel('Importance')
  plt.legend(['Features outside window','Features within max points window'])

  #Plotting sub-plots for window with highest importance
  plt.subplot(2,2,4)
  plt.scatter(x4+FEATURES_LOWER,y4)
  plt.scatter(x4_window+FEATURES_LOWER,y4_window,color='orange')
  plt.xlabel('Features')
  plt.ylabel('Importance')
  plt.legend(['Features outside window','Features within best window'])
  plt.show()

##############################################################################################
###################### WINDOW RE-CALCULATION METRICS ##########################################

#This function calculates the best window within the ['FEATURES_LOWER','FEATURES_UPPER'] window. The metric is the cumulative sum of all importances within the window.
def findbestWindow(att_mat):
    temp=att_mat.argsort()[-FINAL_FEATURE_EXTRACT:][::-1]  #Calculating top features
    temp_index_sort=np.sort(temp)                          #Array of all sorted indices from the top features
    temp_vals_sort=att_mat[temp_index_sort]                #The importance values corresponding to the indices
    
    index=np.where(temp_index_sort>=FINAL_FEATURE_EXTRACT)[0][0] #Finding out where the first occurrence of the 700th window is
    sum=np.sum(temp_vals_sort[0:index])                          #The sum of importances for the first window

    #Initializing the window and the highest sum window
    val_max=0
    window=[FEATURES_LOWER-1,FINAL_FEATURE_EXTRACT+FEATURES_LOWER-1]

    #Shifting the window using a for loop
    for i in range(1,FINAL_FEATURE_EXTRACT-index):
      temp_sum= sum-temp_vals_sort[i-1]+temp_vals_sort[index+i-1]           #Calculating the sum of importances for every window
      if(temp_sum>val_max):

        #Updating the best values
        val_max=temp_sum
        window=[temp_index_sort[i]+FEATURES_LOWER-1,FINAL_FEATURE_EXTRACT+temp_index_sort[i]+FEATURES_LOWER-1]

    return window


#This function helps calculate the window with the most number of top features
def findmaxWindow(att_mat):
    temp=att_mat.argsort()[-FINAL_FEATURE_EXTRACT:][::-1]  #Calculating top features
    temp_index_sort=np.sort(temp).reshape((-1,1))          #Array of sorted indices
    
    #Initializing the index, the maximum length and the window
    index=0
    len_max=0
    window=[FEATURES_LOWER-1,FEATURES_LOWER+FINAL_FEATURE_EXTRACT-1]

    #Shifting the window using a for loop
    for i in range(temp_index_sort.shape[0]):
      start=temp_index_sort[i][0]
      end=start+FINAL_FEATURE_EXTRACT

      #Breaking out of loop if it exceeds bounds
      if(end>=FEATURES_UPPER-FEATURES_LOWER-1):
        break
      print(end)
      try:
        index=np.where(temp_index_sort[index:]>=end)[0][0] + index
      except:
        break
      length=index-i                                               #Calculating new length

      #Updating length and window
      if(length>len_max):
        len_max=length
        window=[start+FEATURES_LOWER-1,start+FEATURES_LOWER+FINAL_FEATURE_EXTRACT-1]

    return window

#############################################################################################################################################
