# pure implementation of SANs
# Skrlj, Dzeroski, Lavrac and Petkovic.

"""
The code containing the neural network part, Skrlj, 2019
Also contains the MLP architecture suitable for DL SCA, Shivaram Srikanth, 2022
"""
#Importing all necessary libraries
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(123321)
np.random.seed(123321)

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

#This function performs one-hot encoding on the input data
def to_one_hot(lbx):
    enc = OneHotEncoder(handle_unknown='ignore')
    return enc.fit_transform(lbx.reshape(-1, 1))

#Creates a sparse matrix from the features for greater speed of computation
class E2EDatasetLoader(Dataset):
    def __init__(self, features, targets=None):  # , transform=None
        features = sparse.csr_matrix(features)
        self.features = features.tocsr()

        if targets is not None:
            self.targets = targets  # .tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        instance = torch.from_numpy(self.features[index, :].todense())
        if self.targets is not None:
            target = torch.from_numpy(np.array(self.targets[index]))
        else:
            target = None
        return instance, target

#This class contains all the information about the model parameters and is called from the 'SAN' class
class SANNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_size, dropout=0.02, num_heads=2, device="cuda",baseline=True):
        super(SANNetwork, self).__init__()

        #All fully connected layers required in the project
        self.fc1 = nn.Linear(input_size, input_size)           #Fully conncted layer before the attention layer(Not used for DL based SCA)
        self.fc2 = nn.Linear(input_size, hidden_layer_size)
        self.fc2_MLP=nn.Linear(input_size,20)
        self.fc3 = nn.Linear(hidden_layer_size, num_classes)
        self.fc3_MLP=nn.Linear(20,50)
        self.fc4_MLP=nn.Linear(50,num_classes)
        self.device = device

        #All activation functions
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)
        self.relu=nn.ReLU()                                    #Required only in the MLP architecture
        self.activation = nn.SELU()
        self.sigmoid = nn.Sigmoid()

        #Tunable parameters
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.multi_head = nn.ModuleList([nn.Linear(input_size, input_size) for k in range(num_heads)])
        self.baseline=baseline

    #This returns the output of the attention layer    
    def forward_attention(self, input_space, return_softmax=True):

        placeholder = torch.zeros(input_space.shape).to(self.device)

        #Looping to aggregate all outputs of the attention heads
        for k in range(len(self.multi_head)):
            if return_softmax:
                attended_matrix = self.multi_head[k](input_space)
            else:
                attended_matrix = self.multi_head[k](input_space) * input_space
            placeholder = torch.add(placeholder,attended_matrix)
        placeholder /= len(self.multi_head)
        out = placeholder
        if return_softmax:
            out = self.softmax(out)
        return out

    #This function is called from the method inside SAN, to calculate the attention weights for each feature
    def get_mean_attention_weights(self):
        activated_weight_matrices = []
        for head in self.multi_head:
            wm = head.weight.data
            diagonal_els = torch.diag(wm)
            activated_diagonal = self.softmax2(diagonal_els)        #Activating the diagonal elements with a softmax function
            activated_weight_matrices.append(activated_diagonal)
        output_mean = torch.mean(torch.stack(activated_weight_matrices, axis=0), axis=0)
        return output_mean

    #This function helps calculate the output for the MLP architecture
    def MLP_forward(self,input):
        out=self.fc2_MLP(input)
        out = self.dropout(out)
        out=self.relu(out)
        out = self.activation(out)
        out=self.fc3_MLP(out)
        out=self.relu(out)
        out=self.fc4_MLP(out)
        return out

    #This function determines the output for the baseline SAN model
    def Original_SAN(self,input):
        out=self.fc1(input)
        out = self.fc2(out)           # dense hidden (l2 in the paper, output)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out
    
    def forward(self, x):
        
        # attend and aggregate
        out = self.forward_attention(x)

        #This decides whether to run the baseline model or the MLP one
        ########################################################################################################################
        if(self.baseline==False):
            out=self.MLP_forward(out)
        else:
            out=self.Original_SAN(out)
        #########################################################################################################################

        out = self.sigmoid(out)
        return out

    def get_attention(self, x):
        return self.forward_attention(x, return_softmax=True)

    def get_softmax_hadamand_layer(self):
        return self.get_mean_attention_weights()

#The SAN class contains all the hyper-parameters for the model and the details of the network
#An instance of this class is first created with all the required hyper-parameters and then the fit method is called with x and y values for the model to begin the process to train.
class SAN:
    def __init__(self, batch_size=32, num_epochs=32, learning_rate=0.001, stopping_crit=10, hidden_layer_size=64,num_heads=1,
                 dropout=0.2,baseline=True):  # , num_head=1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.loss = torch.nn.CrossEntropyLoss()
        self.loss = torch.nn.BCELoss()        #The loss is set to Binary Cross Entropy Loss
        self.dropout = dropout
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.stopping_crit = stopping_crit
        self.num_epochs = num_epochs
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.num_params = None
        self.baseline=baseline
    
    #Fit method sets the model for training
    def fit(self, features, labels):  # , onehot=False
        
        label_unique=np.unique(labels) #Unique labels
        nun = len(label_unique)        #Length of unique labels
        one_hot_labels = []            #Array that will hold the One hot encoded labels

        #Looping through all labels to perform one hot encoding
        for j in range(len(labels)):
            lvec = np.zeros(nun)
            lj = labels[j]
            lvec[np.where(label_unique==lj)] = 1
            one_hot_labels.append(lvec)
        one_hot_labels = np.matrix(one_hot_labels)
        logging.info("Found {} unique labels.".format(nun))

        #Loading the dataset
        train_dataset = E2EDatasetLoader(features, one_hot_labels)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        stopping_iteration = 0
        current_loss = np.inf

        #Setting model parameters
        self.model = SANNetwork(features.shape[1], num_classes=nun, hidden_layer_size=self.hidden_layer_size, num_heads = self.num_heads,
                                dropout=self.dropout, device=self.device,baseline=self.baseline).to(self.device)                          #The values passeed into SAN go into this model to train the model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)                          #ADAM optimizer
        self.num_params = sum(p.numel() for p in self.model.parameters())                                          #Calculation of the total parameters for the model
        logging.info("Number of parameters {}".format(self.num_params))
        logging.info("Starting training for {} epochs".format(self.num_epochs))

        #Looping through the epochs 
        for epoch in range(self.num_epochs):
            if stopping_iteration > self.stopping_crit:
                logging.info("Stopping reached!")
                break
            losses_per_batch = []
            self.model.train()                                 #Initializing model
            for i, (features, labels) in enumerate(dataloader):
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)                
                outputs = self.model(features)           #Running the model. This runs the forward() function
                loss = self.loss(outputs, labels)        #Calculating the Binary Cross Entropy loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))    
            mean_loss = np.mean(losses_per_batch)        #Mean of BCE losses for all batches

            #This part of the code checks the number of iterations for which the mean_loss goes up
            if mean_loss < current_loss:
                current_loss = mean_loss
                stopping_iteration = 0
            else:
                stopping_iteration += 1
            logging.info("epoch {}, mean loss per batch {}".format(epoch, mean_loss))

    #Not used
    def predict(self, features, return_proba=False):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        with torch.no_grad():
            for features, _ in test_dataset:
                self.model.eval()
                features = features.float().to(self.device)
                representation = self.model(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        if not return_proba:
            a = [np.argmax(a_) for a_ in predictions]  # assumes 0 is 0
            return np.array(a).flatten()
        else:
            a = [a_ for a_ in predictions]
            return a

    #Not used
    def predict_proba(self, features):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in test_dataset:
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        a = [a_[1] for a_ in predictions]
        return np.array(a).flatten()

    #This function helps calculate the global attention matrix for the features
    def get_mean_attention_weights(self):
        return self.model.get_mean_attention_weights().detach().cpu().numpy()

    #This function helps ccalculate the instance based attention for every sample trace(Not used)
    def get_instance_attention(self, instance_space):
        if "scipy" in str(type(instance_space)):
            instance_space = instance_space.todense()
        instance_space = torch.from_numpy(instance_space).float().to(self.device)
        return self.model.get_attention(instance_space).detach().cpu().numpy()

