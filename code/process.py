import os
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from model import *
from train import *
from predict import *
import scanpy as sc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC

def extract_csv_sample(csv_path):
    if not os.path.exists(csv_path):
        return None
    rows = 4
    cols = 4
    df = pd.read_csv(
        csv_path,
        nrows=rows,
        encoding='utf-8',
        on_bad_lines='skip'
    )
    df = df.iloc[:, :cols] 
    
    text = "Header: " + ", ".join(df.columns.tolist()) + "\n"
    text += "Data:\n"
    text += df.to_string(index=False)
    
    return text

def data_process(address):
    cell_names = []
    gene_data = []
    with open(address, 'r', newline='') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        cell_names = first_row[1:]
        for row in reader:
            if not row:
                continue
            expressions = [float(value) for value in row[1:]]
            gene_data.append(expressions)
    x = np.array(gene_data).T

    unique_names, y = np.unique(cell_names, return_inverse=True)
    return x, y

def split_dataset(x, y, valid_size):
    X_train, X_temp, y_train, y_temp, train_indices, valid_indices = train_test_split(
        x, y, np.arange(len(y)),
        test_size=valid_size,
        random_state=0, 
        stratify=y
    )

    train_mask = np.zeros(len(y), dtype=bool)
    valid_mask = np.zeros(len(y), dtype=bool)
    
    train_mask[train_indices] = True
    valid_mask[valid_indices] = True

    return train_mask, valid_mask

def construct_G(x, device):
    len_cell = x.shape[0]
    len_gene = x.shape[1]
    edge_index = [[] for _ in range(len_cell+len_gene)]

    for i in np.arange(0,len_cell,1):
        for j in np.arange(0, len_gene,1):
            cell_id = i
            gene_id = len_cell + j
            if x[i][j] > 1e-4:
                edge_index[cell_id].append(gene_id)
                edge_index[gene_id].append(cell_id)
    src = []
    dst = []
    for node, neighbors in enumerate(edge_index):
        src.extend([node] * len(neighbors))
        dst.extend(neighbors)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA().fit(x_scaled)
    cum_evr = np.cumsum(pca.explained_variance_ratio_)
    optimal_dim = np.argmax(cum_evr >= 0.95) + 1

    pca = PCA(n_components=optimal_dim)
    x_principal = pca.fit_transform(x_scaled)
    transform_matrix = pca.components_.T
    feature = np.concatenate([x_principal, transform_matrix], axis=0)

    return feature.astype(np.float32), edge_index, optimal_dim

def tranning_by_scABiGNN(matrix, matrix_test, label, t_m, v_m, device):
    x = matrix.copy()
    x_train = matrix_test.copy()
    x = np.concatenate([x,x_train],axis=0)
    y = label.copy()
    train_mask = t_m.copy()
    valid_mask = v_m.copy()

    feature, edge_index, dim = construct_G(x, device)
    model = scABiGNN(
        in_channels=int(dim),
        out_channels=13,
        ).to(device)
    
    train_mask = np.concatenate([train_mask,np.full(x.shape[1]+x_train.shape[0], False, dtype=bool)],axis=0)
    valid_mask = np.concatenate([valid_mask,np.full(x.shape[1]+x_train.shape[0], False, dtype=bool)],axis=0)
    y = np.concatenate([y,np.full(x.shape[1]+x_train.shape[0], -1, dtype=int)],axis=0)
    feature = torch.from_numpy(feature).to(device)
    y = torch.from_numpy(y).to(device)

    train_scABiGNN(model, feature, y, edge_index, train_mask, valid_mask, epochs=300, lr=5e-4)

def predicting_by_scABiGNN(matrix, label, train_matrix, device, valid):
    x = matrix.copy()
    y = label.copy()
    train_x = train_matrix.copy()
    len_train_x = train_x.shape[0]
    x = np.concatenate([train_x,x],axis=0)

    feature, edge_index, dim = construct_G(x, device)

    model = scABiGNN(
        in_channels=int(dim),
        out_channels=13,
        ).to(device)

    feature = torch.from_numpy(feature).to(device)
    y = torch.from_numpy(y).to(device)

    pred, metrics = predict_scABiGNN(model, feature, y, edge_index, len_train_x,valid)

    return pred, metrics

def tranning_by_celltypist(matrix, label, t_m):
    x = matrix.copy()
    y = label.copy()

    adata = sc.AnnData(x)
    adata.obs["label"] = y
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    train_adata = adata[t_m].copy()
    train_celltypist(train_adata)

def predicting_by_celltypist(matrix, label):
    x = matrix.copy()
    y = label.copy()

    adata = sc.AnnData(x)
    adata.obs["label"] = y
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    pred, metrics = predict_celltypist(adata)
    return pred, metrics

def tranning_by_scVI(matrix, label, t_m):
    x = matrix.copy()
    y = label.copy()

    adata = sc.AnnData(x)
    adata.obs["label"] = y.astype(str)
    adata.obs["batch"] = "batch1"
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    train_adata = adata[t_m].copy()
    train_scVI(train_adata)

def predicting_by_scVI(matrix, label):
    x = matrix.copy()
    y = label.copy()

    adata = sc.AnnData(x)
    adata.obs["label"] = y.astype(str)
    adata.obs["batch"] = "batch1"
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    pred, metrics = predict_scVI(adata)
    return pred, metrics

def tranning_by_SVM(matrix, label, t_m):
    x = matrix.copy()
    y = label.copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    
    model = SVC(kernel="linear", class_weight="balanced", random_state=0)
    param_grid = {
        "C": [0.1, 1, 10],        
        "gamma": ["scale", "auto"]
    }

    x_scaled = x_scaled[t_m]
    y = y[t_m]
    train_SVM(model, x_scaled, y, param_grid)

def predicting_by_SVM(matrix, label, train_matrix):
    x = matrix.copy()
    y = label.copy()
    train_x = train_matrix.copy()

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    x_scaled = scaler.transform(x)

    model = SVC(kernel="linear", class_weight="balanced", random_state=0)
    pred, metrics = predict_SVM(model, x_scaled, y)
    return pred, metrics


def train_assesser(assesser, x, y, device, x_test, y_test):
    pred_scABiGNN, metrics_scABiGNN = predicting_by_scABiGNN(x_test, y, x, device,1)
    pred_celltypist, metrics_celltypist = predicting_by_celltypist(x, y)
    pred_scVI, metrics_scVI = predicting_by_scVI(x, y)
    pred_SVM, metrics_SVM = predicting_by_SVM(x, y, x)

    X_meta = np.column_stack([pred_scABiGNN,pred_celltypist,pred_scVI.astype(int),pred_SVM])
    y_meta = y
    assesser.fit(X_meta, y_meta)

    return assesser

def calculate_metrics(y_true, y_pred):

    precision = precision_score(y_true, y_pred,average='weighted')
    recall = recall_score(y_true, y_pred,average='weighted')
    f1 = f1_score(y_true, y_pred,average='weighted')
    
    metrics = {
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

    return metrics
