import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

df = pd.read_pickle('bgc_embeddings_dfs/bgc0000001_embeddings.pkl')
df_ft = pd.read_pickle('bgc_embeddings_dfs/bgc0000001_embeddings_finetuned.pkl')

# logistic regression on embeddings from each layer
def five_fold_cv(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # clf = LogisticRegression(max_iter=10000, n_jobs=-1).fit(X_train, y_train)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    return accuracies

layer_accuracies = {'Layer': [], 'Accuracy': [], 'Model': []}
labels = [i for sublist in df['Narrowed_Labels'] for i in sublist]
le = LabelEncoder()
le.fit(labels)
labels_encoded = [le.transform(lbls) for lbls in df['Narrowed_Labels']]
for layer in [1, 5, 9, 13]:
    print(f'Processing Layer {layer}...')
    X = []
    for i in range(len(df)):
        embeddings = df[f'layer_{layer}_embeddings'][i]  # (seq_len, hidden_size)
        X.extend(embeddings)
    X = np.array(X)
    X = X[:y.shape[0]]  # ensure X and y have the same length
    y = np.concatenate(labels_encoded)
    acc = five_fold_cv(X, y)
    layer_accuracies['Layer'].append(layer)
    layer_accuracies['Accuracy'].append(acc)
    layer_accuracies['Model'].append('Multi-species Pretrained')
    
    # finetuned model
    X_ft = []
    for i in range(len(df_ft)):
        embeddings_ft = df_ft[f'layer_{layer}_embeddings'][i]
        X_ft.extend(embeddings_ft)
    X_ft = np.array(X_ft)
    X_ft = X_ft[:y.shape[0]]  # ensure X and y have the same length
    y_ft = np.concatenate([le.transform(lbls) for lbls in df_ft['Narrowed_Labels']])
    acc_ft = five_fold_cv(X_ft, y_ft)
    layer_accuracies['Layer'].append(layer)
    layer_accuracies['Accuracy'].append(acc_ft)
    layer_accuracies['Model'].append('GTDB-1K Finetuned')

# boxplot, x=Layer, y=Accuracy, hue=Model
acc_df = pd.DataFrame(layer_accuracies)
acc_df = acc_df.explode('Accuracy')
plt.figure(figsize=(8, 6))
sns.barplot(data=acc_df, x='Layer', y='Accuracy', hue='Model', ci='sd', palette='tab10')
plt.xlabel('Transformer Layer')
plt.title('5-Fold CV Accuracy of Logistic Regression on Embeddings')
plt.ylim(0.3, 0.7)
os.makedirs('plot', exist_ok=True)
plt.savefig('plot/layer_wise_logreg_accuracy.pdf')
plt.savefig('plot/layer_wise_logreg_accuracy.png', dpi=300)
plt.close()

# t-SNE plot of embeddings for each layer, colored by label
# each layer in one figure with subplots, left: pretrained, right: finetuned
for layer in [1, 5, 9, 13]:
    print(f'Generating t-SNE for Layer {layer}...')
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, data, title in zip(axes, [df, df_ft], ['Multi-species Pretrained', 'GTDB-1K Finetuned']):
        X = []
        y = []
        for i in range(len(data)):
            embeddings = data[f'layer_{layer}_embeddings'][i]
            X.extend(embeddings)
            y.extend(le.transform(data['Narrowed_Labels'][i]))
        X = np.array(X)
        y = np.array(y)
        # sample each class to max 1000 points to speed up t-SNE
        unique_classes, class_counts = np.unique(y, return_counts=True)
        sampled_X = []
        sampled_y = []
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) > 1000:
                sampled_indices = np.random.choice(cls_indices, size=1000, replace=False)
            else:
                sampled_indices = cls_indices
            sampled_X.append(X[sampled_indices])
            sampled_y.append(y[sampled_indices])
        sampled_X = np.vstack(sampled_X)
        sampled_y = np.hstack(sampled_y)
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(sampled_X)
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=sampled_y, cmap='tab20', s=1, alpha=1)
        ax.set_title(f'Layer {layer} - {title}')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
    handles, labels = scatter.legend_elements(num=len(le.classes_))
    legend = fig.legend(handles, le.classes_, title="Labels", bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f'plot/tsne_layer_{layer}.pdf', bbox_inches='tight')
    plt.savefig(f'plot/tsne_layer_{layer}.png', dpi=300, bbox_inches='tight')
    plt.close()
