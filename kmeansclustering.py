import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# LOADING OF THE DATASET 
df = pd.read_csv('Mall_Customers.csv')

# FEATURES TO USE
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# NORMALIZING THE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CALCULATE WCSS FOR DIFFERENT VALUES OF K
wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ attribute is WCSS

# PLOT THE ELBOW METHOD GRAPH
plt.figure(figsize=(8,5))
plt.plot(range(1, 10), wcss, marker='o', linestyle='-', color='b')
plt.title("Elbow Method - Optimal k", fontsize=14, fontweight='bold')
plt.xlabel("Number of clusters (k)", fontsize=12)
plt.ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
plt.xticks(range(1, 10))
plt.grid(True)
plt.tight_layout()
plt.show()

# APPLY K-MEANS WITH CHOSEN NUMBER OF CLUSTERS
k = 3  # You may change this based on the elbow plot
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# DISPLAY THE FIRST FEW ROWS WITH CLUSTER LABELS
print(df.head())

# PLOT THE CLUSTERS WITH SEABORN
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='viridis', s=100, alpha=0.8)
plt.title("Customer Segmentation by K-Means Clustering", fontsize=14, fontweight='bold')
plt.xlabel("Annual Income (k$)", fontsize=12)
plt.ylabel("Spending Score (1-100)", fontsize=12)
plt.legend(title='Cluster', loc='best')
plt.tight_layout()
plt.show()
