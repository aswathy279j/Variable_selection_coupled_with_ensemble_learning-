import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data # replace with your filename 
data_file = 'myfilename.csv'
data = pd.read_csv(data_file)

# Extract features and class labels (assuming class labels in column 1)
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values  
feature_names = data.columns[2:]
base_filename = os.path.splitext(os.path.basename(data_file))[0]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA # Can also check the percentage variance captured to arrive at the optimal number of components
n_components = 5
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# DataFrame for PCs and class labels
pc_df = pd.DataFrame(data=principal_components,
                     columns=[f'PC{i+1}' for i in range(n_components)])
pc_df['ClassLabel'] = y

# Save PCA scores with class labels
pc_df.to_csv(f'pca_scores_{base_filename}.csv', index=False)

# variance plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, n_components + 1), explained_variance_ratio * 100,
        alpha=0.7, color='teal', label='Individual Explained Variance')
plt.step(range(1, n_components + 1), cumulative_variance * 100,
         where='mid', label='Cumulative Explained Variance', color='orange')
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Variance Explained (%)', fontsize=12)
plt.title('Explained Variance by Principal Components', fontsize=14)
plt.xticks(range(1, n_components + 1))
plt.ylim(0, 110)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'Variance_explained_{base_filename}.png', dpi=300)
plt.close()

# Unique classes and colors for plotting
unique_classes = np.unique(y)
color_list = plt.cm.get_cmap('tab20', len(unique_classes))
colors = {cls: color_list(i) for i, cls in enumerate(unique_classes)}

# 2D PCA scatter plot PC1 vs PC2 colored by actual class labels
plt.figure(figsize=(8, 6))
for cls in unique_classes:
    idx = pc_df['ClassLabel'] == cls
    plt.scatter(pc_df.loc[idx, 'PC1'], pc_df.loc[idx, 'PC2'],
                label=cls, alpha=0.7, edgecolors='k', s=70, c=[colors[cls]])

plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', fontsize=12)
plt.title('PCA Score Plot: PC1 vs PC2', fontsize=14)
plt.legend(title='Class Label', fontsize=10, loc='best', bbox_to_anchor=(1.05, 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'PCA_{base_filename}.png', dpi=300)
plt.close()

# Interactive 3D PCA plot with plotly - uses actual class labels
try:
    import plotly.express as px
    df_plotly = pc_df.iloc[:, :3].copy()
    df_plotly['ClassLabel'] = y

    fig = px.scatter_3d(df_plotly, x='PC1', y='PC2', z='PC3',
                        color='ClassLabel',
                        labels={
                            'PC1': f'PC1 ({explained_variance_ratio[0]*100:.2f}%)',
                            'PC2': f'PC2 ({explained_variance_ratio[1]*100:.2f}%)',
                            'PC3': f'PC3 ({explained_variance_ratio[2]*100:.2f}%)'
                        },
                        title='PCA 3D Score Plot')
    fig.show()
except ImportError:
    print("Plotly is not installed. Interactive 3D plot will be skipped")
