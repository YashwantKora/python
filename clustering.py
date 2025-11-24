# 1️⃣ Import
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 2️⃣ Load
df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\Iris.csv")   # or your file path
X = df.iloc[:, :-1].values             # a3l features except species

# 3️⃣ Model (KMeans with 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# 4️⃣ Visualize clusters (first two features)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c='green', label='Cluster 3')

# 5️⃣ Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='*', s=200, c='yellow', label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
