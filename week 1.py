import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import warnings
import os

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置环境变量防止并发警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load the dataset
df = pd.read_csv(r'C:\Users\ADMIN\Desktop\nigerian-songs.csv')

# Data cleaning - 更彻底的数据清洗
print(f"原始数据形状: {df.shape}")
df = df.dropna()
print(f"删除空值后形状: {df.shape}")

# 移除无效流派
df = df[df['artist_top_genre'] != 'Missing']
print(f"移除无效流派后形状: {df.shape}")

# 移除0流行度和异常值
df = df[(df['popularity'] > 0) & (df['popularity'] <= 100)]
df = df[df['release_date'] >= 1970]  # 过滤异常年份

# Convert milliseconds to minutes
df['length_min'] = df['length'] / 60000

# Exploratory Data Analysis
print(f"\n数据集最终形状: {df.shape}")
print(f"\nTop genres:\n{df['artist_top_genre'].value_counts().head(5)}")
print(f"\n时间范围: {df['release_date'].min()} 至 {df['release_date'].max()}")

# Correlation analysis
plt.figure(figsize=(12, 8))
corr_matrix = df.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角掩码
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt=".2f")
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# Visualize distributions
plt.figure(figsize=(15, 10))
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'popularity']
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True, color='skyblue')
    plt.title(f'{feature}分布')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.show()

# Genre analysis
top_genres = df['artist_top_genre'].value_counts().head(5).index
genre_df = df[df['artist_top_genre'].isin(top_genres)]

plt.figure(figsize=(12, 8))
sns.boxplot(x='artist_top_genre', y='popularity', data=genre_df, palette='Set2')
plt.title('不同流派的流行度分布')
plt.xticks(rotation=45)
plt.xlabel('流派')
plt.ylabel('流行度')
plt.savefig('popularity_by_genre.png', dpi=300)
plt.show()

# Prepare data for PCA
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'tempo']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=principal_components,
                      columns=['PC1', 'PC2', 'PC3'])

# Explained variance
explained_var = pca.explained_variance_ratio_
print(f"\n主成分解释方差比: {explained_var}")
print(f"累计解释方差: {sum(explained_var):.3f}")

# Visualize PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', data=df_pca, alpha=0.7, color='steelblue')
plt.title('尼日利亚音乐特征PCA分析')
plt.xlabel(f'PC1 ({explained_var[0] * 100:.1f}%)')
plt.ylabel(f'PC2 ({explained_var[1] * 100:.1f}%)')
plt.grid(alpha=0.3)
plt.savefig('pca_visualization.png', dpi=300)
plt.show()

# Interactive 3D PCA plot
fig = px.scatter_3d(
    df_pca, x='PC1', y='PC2', z='PC3',
    title='尼日利亚音乐特征3D PCA可视化',
    labels={'PC1': f'PC1 ({explained_var[0] * 100:.1f}%)',
            'PC2': f'PC2 ({explained_var[1] * 100:.1f}%)',
            'PC3': f'PC3 ({explained_var[2] * 100:.1f}%)'},
    opacity=0.7,
    height=800
)
fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
fig.write_html('3d_pca.html')
fig.show()

# K-Means clustering - 优化聚类数选择
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

    if k > 1:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}: 轮廓系数 = {score:.4f}")

# 自动选择最优K值
optimal_k = np.argmax(silhouette_scores) + 2  # 因为k从2开始
print(f"\n根据轮廓系数选择的最优K值: {optimal_k}")

# Elbow method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')
plt.title('肘部法则 (WCSS)')
plt.xlabel('聚类数量')
plt.ylabel('WCSS')
plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)

# Silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, 'go-')
plt.title('轮廓系数')
plt.xlabel('聚类数量')
plt.ylabel('轮廓系数')
plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cluster_evaluation.png', dpi=300)
plt.show()

# Apply K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters
df_pca['cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='cluster',
    data=df_pca,
    palette='viridis',
    alpha=0.8,
    s=80
)
plt.title(f'尼日利亚歌曲聚类 (K={optimal_k})')
plt.xlabel(f'PC1 ({explained_var[0] * 100:.1f}%)')
plt.ylabel(f'PC2 ({explained_var[1] * 100:.1f}%)')
plt.legend(title='聚类')
plt.grid(alpha=0.3)
plt.savefig('clusters_pca.png', dpi=300)
plt.show()

# Analyze cluster characteristics
cluster_features = features + ['popularity', 'length_min']
cluster_analysis = df.groupby('cluster')[cluster_features].mean()

print("\n聚类特征分析:")
print(cluster_analysis)

# Visualize cluster means
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_analysis.T, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('聚类特征均值')
plt.xlabel('聚类')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=300)
plt.show()

# 聚类分布分析
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=df, palette='viridis')
plt.title('聚类分布')
plt.xlabel('聚类')
plt.ylabel('歌曲数量')
plt.savefig('cluster_distribution.png', dpi=300)
plt.show()

# 聚类与流派的关系
plt.figure(figsize=(12, 8))
cross_tab = pd.crosstab(df['cluster'], df['artist_top_genre'])
sns.heatmap(cross_tab, cmap="Blues", annot=True, fmt='d')
plt.title('聚类与流派关系')
plt.xlabel('流派')
plt.ylabel('聚类')
plt.tight_layout()
plt.savefig('cluster_genre_heatmap.png', dpi=300)
plt.show()

# Save results
df.to_csv('nigerian_songs_with_clusters.csv', index=False)
print("\n分析完成! 结果已保存。")