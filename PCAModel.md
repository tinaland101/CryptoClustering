Step 1: Setup the Repository & Environment
Download the starter code and dataset (crypto_market_data.csv).
Rename the starter code file to Crypto_Clustering.ipynb.
Install necessary Python packages using:

pip install pandas numpy scikit-learn hvplot matplotlib
Step 2: Load and Explore the Data
Actions:
Load the dataset using pandas and inspect its structure.
Generate summary statistics to understand feature distributions.
Visualize price changes over 24 hours and 7 days.
Code:

import pandas as pd

# Load dataset
file_path = "crypto_market_data.csv"
df = pd.read_csv(file_path)

# Display first five rows
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
Step 3: Data Preprocessing
Actions:
Scale the features using StandardScaler from scikit-learn.
Create a new scaled DataFrame, maintaining the "coin_id" as the index.
Code:
python
Copy
Edit
from sklearn.preprocessing import StandardScaler

# Select numerical columns for scaling
df_market_data = df.drop(columns=["coin_id"])  

# Scale data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_market_data), 
                         columns=df_market_data.columns, 
                         index=df["coin_id"])

# Display first five rows
print(df_scaled.head())
Step 4: Determine the Best k using the Elbow Method
Actions:
Iterate k values from 1 to 11.
Compute inertia (sum of squared distances).
Plot the Elbow Curve to find the optimal k.
Code:
python
Copy
Edit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define range for k
k_values = range(1, 11)
inertia = []

# Compute inertia for each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal k')
plt.show()
Question Answer:
Based on the elbow curve, select the best k.
Step 5: Cluster Cryptocurrencies Using K-Means
Actions:
Initialize K-Means with the best k found in Step 4.
Fit K-Means to the scaled DataFrame.
Predict clusters and store them in the DataFrame.
Code:
python
Copy
Edit
# Apply K-Means clustering
best_k = 4  # Adjust based on elbow curve result
kmeans = KMeans(n_clusters=best_k, random_state=1)
df_scaled["Cluster"] = kmeans.fit_predict(df_scaled)

# Display cluster counts
print(df_scaled["Cluster"].value_counts())
Step 6: Visualize Clusters in Feature Space
Actions:
Create a scatter plot of 24-hour vs. 7-day price changes, colored by clusters.
Use hvPlot to add interactive hover information.
Code:
python
Copy
Edit
import hvplot.pandas

df_scaled.hvplot.scatter(
    x="price_change_percentage_24h", 
    y="price_change_percentage_7d", 
    by="Cluster", 
    hover_cols=["Cluster"], 
    title="Cryptocurrency Clusters (Original Data)"
)
Step 7: Apply PCA for Dimensionality Reduction
Actions:
Reduce features to three principal components using PCA.
Retrieve explained variance of each component.
Create a new PCA-transformed DataFrame.
Code:
python
Copy
Edit
from sklearn.decomposition import PCA

# Apply PCA to reduce to 3 components
pca = PCA(n_components=3)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled.drop(columns=["Cluster"])), 
                      index=df_scaled.index, 
                      columns=["PC1", "PC2", "PC3"])

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance:", explained_variance)
print("Total Explained Variance:", sum(explained_variance))
Question Answer:
Total Explained Variance: Sum of explained_variance.
Step 8: Find the Best k using PCA-Reduced Data
Actions:
Repeat the Elbow Method using the PCA-transformed data.
Code:
python
Copy
Edit
# Compute inertia for PCA-transformed data
inertia_pca = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(df_pca)
    inertia_pca.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(k_values, inertia_pca, marker='o', label="PCA Data")
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method After PCA')
plt.legend()
plt.show()
Question Answer:
Compare best k before and after PCA.
Step 9: Cluster Cryptocurrencies Using PCA Data
Actions:
Apply K-Means clustering to the PCA-reduced DataFrame.
Code:
python
Copy
Edit
best_k_pca = 4  # Adjust based on new elbow curve result
kmeans_pca = KMeans(n_clusters=best_k_pca, random_state=1)
df_pca["Cluster"] = kmeans_pca.fit_predict(df_pca)
Step 10: Visualize Clusters in PCA Space
Actions:
Create an interactive scatter plot using PC1 and PC2.
Code:
python
Copy
Edit
df_pca.hvplot.scatter(
    x="PC1", 
    y="PC2", 
    by="Cluster", 
    hover_cols=["Cluster"], 
    title="Cryptocurrency Clusters (PCA Data)"
)
Step 11: Compare Cluster Results
Actions:
Use hvPlot to compare clustering before and after PCA.
Code:
python
Copy
Edit
scatter_original + scatter_pca
Question Answer:
Discuss the impact of PCA on clustering:
Does PCA improve cluster separation?
Does PCA reduce computational cost?
Is there a trade-off in information loss?
Final Steps: Deployment
Actions:
Commit all changes to GitHub:
bash
Copy
Edit
git add .
git commit -m "Completed CryptoClustering project"
git push origin main
