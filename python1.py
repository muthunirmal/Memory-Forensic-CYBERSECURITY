import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_excel('book.xlsx')
features = df[['Memory_Usage_MB', 'Num_Threads', 'Network_Connections']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# Plot the dot plot with annotations
plt.scatter(df['Num_Threads'], df['PID'], c='blue')  # Changed color to blue
plt.xlabel('Num_Threads')
plt.ylabel('PID')
plt.title('Cluster Analysis of Malware Data')

# Annotate each point with its value in brackets
for i, row in df.iterrows():
    plt.annotate(f'({row["Num_Threads"]}, {row["PID"]})', 
                 (row['Num_Threads'], row['PID']), 
                 textcoords="offset points", 
                 xytext=(0,5), 
                 ha='center', 
                 fontsize=8)

plt.show()

print(df.head())






