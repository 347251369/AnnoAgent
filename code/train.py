import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
import celltypist
import scvi
from sklearn.model_selection import GridSearchCV
import pickle

def train_scABiGNN(model, feature, y, edge_index, train_mask, valid_mask, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    criterion = F.nll_loss
    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(feature, edge_index)
        total_loss = criterion(output[train_mask], y[train_mask])

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(feature, edge_index)
            val_pred = val_logits[valid_mask].argmax(dim=1)
            val_f1 = f1_score(val_pred.cpu().numpy(), y[valid_mask].cpu().numpy(),average='weighted')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_val_f1': best_val_f1
                }, "best_scABiGNN.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, total Loss: {total_loss.item():.6f}")

def train_celltypist(train_adata):

    model = celltypist.train(
        train_adata,
        labels=train_adata.obs["label"],
        n_jobs=8,
        use_SGD=True,
        verbose=False
    )
    model.write("best_celltypist.pkl")

def train_scVI(train_adata):
    scvi.model.SCVI.setup_anndata(
        train_adata,
        batch_key="batch",
        labels_key="label"
    )

    vae = scvi.model.SCVI(train_adata)
    vae.train(
        max_epochs=50,
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_monitor="elbo_validation",
    )

    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae,
        unlabeled_category="Unknown",
        labels_key="label"
    )

    scanvi.train(
        max_epochs=10,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_monitor="elbo_validation",
    )

    scanvi.save("best_scanVI", overwrite=True)

def train_SVM(model, x_scaled, y, param_grid):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,                  
        scoring="f1_weighted",
        n_jobs=-1
    )
    grid_search.fit(x_scaled, y)

    model = grid_search.best_estimator_
    with open("best_SVM.pkl", "wb") as f:
        pickle.dump(model, f)