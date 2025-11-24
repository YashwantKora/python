# 1️⃣ Import
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 2️⃣ Load
df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\income.csv")

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show() # Added to display the initial scatter plot

# 3️⃣ Prepare (scale Age and Income)
scaler = MinMaxScaler()
df[['Age', 'Income($)']] = scaler.fit_transform(df[['Age', 'Income($)']])

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show() # Added to display the scatter plot after scaling

# 4️⃣ Model
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(df[['Age', 'Income($)']])

# 5️⃣ Visualize
centers = kmeans.cluster_centers_
plt.scatter(df['Age'], df['Income($)'], c=df['cluster'])
plt.scatter(centers[:,0], centers[:,1], marker='*', s=200, label='centroids')
plt.legend()
plt.show()
