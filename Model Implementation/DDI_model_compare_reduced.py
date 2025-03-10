import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from tabulate import tabulate
import tensorflow as tf
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)

tf.config.set_visible_devices([], 'GPU')

#Basic dimensionality reduction down to a selected number of components
#have X include cols to be able to rejoin with datasets if desired
def Dim_red(X, feature_num, batch_size, col_name):
	ipca = IncrementalPCA(n_components=feature_num)

	for i in range(0, X.shape[0], batch_size):
	    X_batch = X[i:i + batch_size]
	    ipca.partial_fit(X_batch)

	X_transformed = pd.DataFrame(ipca.transform(X))
	X_transformed['cols'] = X[col_name]

	#prove that X's shape has changed
	#print("Original shape:", X.shape)
	#print("Transformed shape:", X_transformed.shape)

	return(X_transformed)

#quick function to turn a list of size 1 lists of strings into a list of strings, for later use
def delist(list_of_lists):
    list_of_strings = []
    for inner_list in list_of_lists:
        string = inner_list[0]
        list_of_strings.append(string)
    return list_of_strings

#a function to calculate layer nodes if using many
def calc_layers(X_size, Y_size):
    layers = [X_size+1]
    layer = 2
    while layer <= X_size:
        layer = int(layer * 2)
    layers.append(layer)
    if X_size > Y_size:
        while layer / 2 > Y_size and layer > 2:
            layer = layer / 2
            layers.append(int(layer))
    layers.append(Y_size)
    return layers

    #basic/rough neural network implementation functions

#a function to perform feed-forward deep neural network analysis
def DNN(X, Y, Epochs, batchsize, layernum=1, verbose=False):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=27, stratify=Y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #define layers and nodes in each layer
    input = len(X.columns)
    output = Y.shape[1]
    if layernum=='many':
        layers = calc_layers(input,output)
    elif type(layernum)==int:
        layers = [input]
        for i in range(layernum, 1, -1):
            layer = int(round((i * (input + output) / (layernum+1)), 0))
            if layer > output:
                layers.append(layer)
        layers.append(output)
    else:
        print(f"incorrect layernum {layernum}")
        return None

    model = keras.models.Sequential()

    model.add(Dense(layers[0], activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    for layer_size in layers[1:-1]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        
    model.add(Dense(layers[-1], activation='softmax'))
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy','AUC','precision','recall'])
    
    model.fit(X_train, y_train, epochs=Epochs, batch_size=batchsize, validation_split=0.1, verbose=verbose)
    
    loss, accuracy, AUC, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    if verbose:
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
        print(f"Test AUC: {AUC}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
    return model, accuracy

    #Creating GNN Model
#credit Beth Farr
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

#Using PyTorch Geometric Graph Format
def create_graph_features(features, labels):
    num_nodes = len(features)

    # creating the nodes
    edge_index = torch.tensor(
        np.array([[i, i] for i in range(num_nodes)]).T, dtype=torch.long
    )

    graphs = []
    for i in range(len(features)):
        x = torch.tensor(features[i], dtype=torch.float).unsqueeze(0)
        y = torch.tensor([labels[i]], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)
    return graphs

#Training the GNN Model
def GNN_train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(torch.device("cpu"))  # Ensure correct device
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

#Evaluating the Model
def GNN_test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(torch.device("cpu"))  # Ensure correct device
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.cpu().size(0)
    acc = correct / total
    return acc

def perform_GNN(X, Y, epochs, batchsize, verbose=False):

    #Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
                                                    
    train_graphs = create_graph_features(X_train, y_train)
    test_graphs = create_graph_features(X_test, y_test)

    # Creating data loaders
    train_loader = DataLoader(train_graphs, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batchsize, shuffle=False)

    #Initialising the Model
    input_dim = X.shape[1]
    output_dim = len(np.unique(Y))
    hidden_dim = int(round((input_dim + output_dim)/2,0))
    
    model = GNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    #Running Training
    accuracy = 0
    for epoch in range(epochs):
        loss = GNN_train(model, train_loader, optimizer, criterion)
        acc = GNN_test(model, test_loader)
        if verbose:
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Accuracy={acc:.4f}")
        if acc > accuracy:
            accuracy = acc

    #Saving the model for future use
    torch.save(model.state_dict(), "gnn_model.pth")
    return model, accuracy

    #function to optimize hyperparameters
def opt_hps(func, X, Y, epochs, batchsizes, show_progress=False):
    best_epochs = 0
    best_batchsize = 0
    best_acc = 0
    grid = pd.DataFrame(columns=epochs, index=batchsizes)
    iter = 0
    max_rep = len(epochs) * len(batchsizes)

    best_model, best_acc = func(X, Y, epochs[0], batchsizes[0])

    for e in range(0,len(epochs)):
        for b in range(0,len(batchsizes)):
            if show_progress:
                iter += 1
                pct = int(round(iter / max_rep * 100,0))
                print(f"Performing {epochs[e]} epochs on batches of {batchsizes[b]}, {pct} percent complete")
            model, acc = func(X, Y, epochs[e], batchsizes[b])

            if show_progress:
                print(f"assigning {acc} to {grid.iloc[b, e]} ")
            grid.iloc[b, e] = acc
            
            if show_progress:
                print(grid)

            if acc > best_acc:
                best_epochs = epochs[e]
                best_batchsize = batchsizes[b]
                best_acc = acc
                best_model = model


    print(f"Optimal hyperparams: {best_epochs} epochs, {best_batchsize} batchsize. Accuracy: {best_acc}")
    return grid, best_model

#initialize dataframe

table2 = pq.read_table('..\\Data Files\\DDI_red_feat.parquet')
df_red_feat = table2.to_pandas()

label_encoder = LabelEncoder()
graph_red_feat_y = label_encoder.fit_transform(df_red_feat['y'])
red_feat_y = to_categorical(df_red_feat['y'])
red_feat_x = df_red_feat.iloc[:,2:]

dnn_epochs = [7,8,9,10,11]
dnn_batchsizes = [150, 200, 250, 300, 350, 400]

gnn_epochs = [5,6,7,8,9]
gnn_batchsizes = [100, 125,150, 175, 200, 225]

#just reduced to top 20 categories
dnn_grid_red, dnn_model = opt_hps(DNN, red_feat_x, red_feat_y, dnn_epochs, dnn_batchsizes, show_progress=True)

print(dnn_grid_red)

#just reduced
gnn_grid_red, gnn_model = opt_hps(perform_GNN, red_feat_x.values, graph_red_feat_y, gnn_epochs, gnn_batchsizes, show_progress=True)

print(gnn_grid_red)

filename = 'dnn_model_red.pkl'
with open(filename, 'wb') as file:
    pickle.dump(dnn_model, file)

filename = 'gnn_model_red.pkl'
with open(filename, 'wb') as file:
    pickle.dump(gnn_model, file)