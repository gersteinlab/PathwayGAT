from utils import *
from model import *
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
import numpy as np
# Added from Shaoke (AUC modified by Weihao to improve on class imbalance)

def evaluation(dataset_file, meta_file, class_name, output_prefix, batch_size, hidden_channels, epochs, folds, learning_rate, multi_class, sample_list_dir, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dataset = torch.load(dataset_file)
    label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
    label_df['label'], _ = pd.factorize(label_df[class_name])
    if sample_list_dir is not None:
        with open(sample_list_dir, 'rb') as f:
            sample_list = pickle.load(f)
        label_df = label_df.iloc[sample_list, ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    criterion = torch.nn.CrossEntropyLoss()

    all_aucs = []
    all_auprs = []
    true_labels = []
    predicted_probs = []

    # K-Fold cross validation
    for train_indices, val_indices in kf.split(dataset):
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = PathwayGAT3(num_features=dataset[0].x.shape[1], hidden_channels=hidden_channels, num_classes=len(set(label_df['label'])), num_nodes=dataset[0].x.shape[0]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for data in train_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                data.x = data.x.to(torch.float32)            
                out = model(data.x, data.edge_index, data.batch)
                probabilities = F.softmax(out, dim=1).cpu().numpy()
                true_labels.extend(label_binarize(data.y.cpu().numpy(), classes=np.arange(model.num_classes)))
                predicted_probs.extend(probabilities)
                        
    # Get AUC and AUPR based on predicted probabilities and true labels
    true_labels = np.vstack(true_labels)
    predicted_probs=np.vstack(predicted_probs)
    np.savez(f'{output_prefix}_mat4auc_savez.npz', true=true_labels, predict=predicted_probs)
    
    auc_scores=[]
    aupr_scores=[]

    for i in range(true_labels.shape[1]):
        if len(np.unique(true_labels[:, i])) == 2:
            auc_scores.append(roc_auc_score(true_labels[:, i], predicted_probs[:, i]))
            aupr_scores.append(average_precision_score(true_labels[:, i], predicted_probs[:, i]))
        else:
            auc_scores.append(np.nan)
            aupr_scores.append(np.nan)

    if multi_class: # With multiple classes: use the original code
        # Compute mean AUC and AUPR excluding NaN values
        mean_auc = np.nanmean(auc_scores)
        mean_aupr = np.nanmean(aupr_scores)
        auc0 = roc_auc_score(true_labels, predicted_probs,multi_class='ovr', average='weighted') # Modified by Weihao to deal with imbalance class
        aupr = average_precision_score(true_labels, predicted_probs)
        all_aucs.append(auc0)
        all_auprs.append(aupr)

        print(f"Average AUC across {folds} folds: {np.mean(all_aucs):.4f}")
        print(f"Average AUPR across {folds} folds: {np.mean(all_auprs):.4f}")

        idmap = label_df[[class_name, "label"]].drop_duplicates()
        class_mapping = dict(zip(idmap['label'], idmap[class_name]))

        plt.figure(figsize=(10, 8))
        for i in range(predicted_probs.shape[1]):
            fpr, tpr, _ = roc_curve(true_labels[:, i], predicted_probs[:, i])
            plt.plot(fpr, tpr, label=f'{class_mapping[i]} (AUC = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Microbe feature prediction ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(fname=f'{output_prefix}_AUC.png', bbox_inches='tight')

        plt.figure(figsize=(10, 8))
        for i in range(predicted_probs.shape[1]):
            precision, recall, _ = precision_recall_curve(true_labels[:, i], predicted_probs[:, i])
            plt.plot(recall, precision, label=f'{class_mapping[i]} (AUPR = {average_precision_score(true_labels[:, i], predicted_probs[:, i]):.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.savefig(fname=f'{output_prefix}_AUPR.png', bbox_inches='tight')
    else: # Only two classes: feed AUC and AUPR with only one probability
        # Compute mean AUC and AUPR excluding NaN values
        mean_auc = np.nanmean(auc_scores)
        mean_aupr = np.nanmean(aupr_scores)
        auc0 = roc_auc_score(true_labels, predicted_probs[:, 1],multi_class='ovr', average='weighted') #SKL change the index of class.. Modified by Weihao to deal with imbalance class
        aupr = average_precision_score(true_labels, predicted_probs[:, 1])
        all_aucs.append(auc0)
        all_auprs.append(aupr)

        print(f"Average AUC across {folds} folds: {np.mean(all_aucs):.4f}")
        print(f"Average AUPR across {folds} folds: {np.mean(all_auprs):.4f}")

        idmap = label_df[[class_name, "label"]].drop_duplicates()
        class_mapping = dict(zip(idmap['label'], idmap[class_name]))

        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, 1])
        plt.plot(fpr, tpr, label=f'{class_mapping[i]} (AUC = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Microbe feature prediction ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(fname=f'{output_prefix}_AUC.png', bbox_inches='tight')

        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(true_labels, predicted_probs[:, 1])
        plt.plot(recall, precision, label=f'{class_mapping[i]} (AUPR = {average_precision_score(true_labels, predicted_probs[:, 1]):.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.savefig(fname=f'{output_prefix}_AUPR.png', bbox_inches='tight')
