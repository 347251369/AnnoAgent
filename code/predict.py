import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import celltypist
import scvi
import os
import pickle

def predict_scABiGNN(model, feature, y, edge_index, len_train_x, valid):
    checkpoint = torch.load("best_scABiGNN.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        output = model(feature, edge_index)
        if valid == 1:
            pred = output[:len_train_x].max(1)[1]
        else:
            pred = output[len_train_x:len_train_x+len(y)].max(1)[1]
        precision = precision_score(pred.cpu().numpy(), y.cpu().numpy(),average='weighted')
        recall = recall_score(pred.cpu().numpy(), y.cpu().numpy(),average='weighted')
        f1 = f1_score(pred.cpu().numpy(), y.cpu().numpy(),average='weighted')
    
    metrics = {
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

    return pred.cpu().numpy(), metrics

def predict_celltypist(test_adata):
    addr = "./best_celltypist.pkl"
    if os.path.exists(addr):
        model = celltypist.models.Model.load("./best_celltypist.pkl")
    
    pred_test = celltypist.annotate(test_adata, model=model)

    y_true_val = test_adata.obs["label"]
    y_pred_val = pred_test.predicted_labels["predicted_labels"]

    precision = precision_score(y_true_val, y_pred_val,average='weighted')
    recall = recall_score(y_true_val, y_pred_val,average='weighted')
    f1 = f1_score(y_true_val, y_pred_val,average='weighted')
    
    metrics = {
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

    return y_pred_val, metrics

def predict_scVI(test_adata):
    model = scvi.model.SCANVI.load(
        "best_scanVI",
        adata = test_adata
    )

    y_pred_val = model.predict(test_adata)
    y_true_val = test_adata.obs["label"]

    precision = precision_score(y_true_val, y_pred_val,average='weighted')
    recall = recall_score(y_true_val, y_pred_val,average='weighted')
    f1 = f1_score(y_true_val, y_pred_val,average='weighted')
    
    metrics = {
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

    return y_pred_val, metrics
 
def predict_SVM(model, x_scaled, y):

    with open("best_SVM.pkl", "rb") as f:
        model = pickle.load(f) 
    y_pred_val = model.predict(x_scaled)

    precision = precision_score(y, y_pred_val,average='weighted')
    recall = recall_score(y, y_pred_val,average='weighted')
    f1 = f1_score(y, y_pred_val,average='weighted')

    metrics = {
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

    return y_pred_val, metrics