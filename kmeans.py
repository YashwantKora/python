import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\hhhhhh\python\datasets\income.csv")
print(df.head())

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show() # Added to display the initial scatter plot


scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
print(df.head())

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show() # Added to display the scatter plot after scaling

km = KMeans(n_clusters=3, n_init=10, random_state=0) # Added n_init and random_state
y_predicted = km.fit_predict(df[['Age','Income($)']])
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())

print(km.cluster_centers_)
df1 = df[df['cluster'] == 0]
df2 = df[df['cluster'] == 1]
df3 = df[df['cluster'] == 2]
plt.scatter(df1.Age,df1['Income($)'],color='green', label='Income Cluster 1') # Added labels
plt.scatter(df2.Age,df2['Income($)'],color='red', label='Income Cluster 2') # Added labels
plt.scatter(df3.Age,df3['Income($)'],color='black', label='Income Cluster 3') # Added labels
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show() # Added to display the clustering plot