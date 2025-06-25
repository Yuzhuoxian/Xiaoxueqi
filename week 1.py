import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.cluster import KMeans  # K-Means聚类算法
from sklearn.metrics import silhouette_score  # 聚类效果评估指标
import plotly.express as px  # 交互式可视化库
import warnings  # 警告处理
import os  # 操作系统接口

# 忽略不必要的警告信息，保持输出整洁
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置环境变量防止并发警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ======================
# 数据加载与预处理
# ======================

# 加载数据集
df = pd.read_csv(r'C:\Users\ADMIN\Desktop\nigerian-songs.csv')

# 数据清洗 - 更彻底的数据清洗
print(f"原始数据形状: {df.shape}")

# 删除包含空值的行
df = df.dropna()
print(f"删除空值后形状: {df.shape}")

# 移除无效流派（'Missing'标签）
df = df[df['artist_top_genre'] != 'Missing']
print(f"移除无效流派后形状: {df.shape}")

# 移除无效的流行度值（0及以下或超过100）和异常年份（1970年以前）
df = df[(df['popularity'] > 0) & (df['popularity'] <= 100)]
df = df[df['release_date'] >= 1970]  # 过滤异常年份

# 将歌曲长度从毫秒转换为分钟
df['length_min'] = df['length'] / 60000

# 打印数据基本信息
print(f"\n数据集最终形状: {df.shape}")
print(f"\nTop genres:\n{df['artist_top_genre'].value_counts().head(5)}")
print(f"\n时间范围: {df['release_date'].min()} 至 {df['release_date'].max()}")

# ======================
# 探索性数据分析 (EDA)
# ======================

# 1. 特征相关性热力图
plt.figure(figsize=(12, 8))
# 选择所有数值型特征计算相关系数矩阵
corr_matrix = df.select_dtypes(include=np.number).corr()
# 创建上三角掩码，避免重复显示
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# 绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# 2. 特征分布直方图
plt.figure(figsize=(15, 10))
# 选择要分析的特征
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'popularity']
# 循环绘制每个特征的分布
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)  # 创建2行3列的子图
    sns.histplot(df[feature], kde=True, color='skyblue')  # 直方图+核密度估计
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.show()

# 3. 流派与流行度关系箱线图
plt.figure(figsize=(12, 8))
# 选择前5个最常见的流派
top_genres = df['artist_top_genre'].value_counts().head(5).index
# 过滤出这些流派的数据
genre_df = df[df['artist_top_genre'].isin(top_genres)]

# 绘制箱线图展示不同流派的流行度分布
sns.boxplot(x='artist_top_genre', y='popularity', data=genre_df, palette='Set2')
plt.title('Popularity by Genre')
plt.xticks(rotation=45)  # 旋转X轴标签避免重叠
plt.xlabel('Genre')
plt.ylabel('Popularity')
plt.savefig('popularity_by_genre.png', dpi=300)
plt.show()

# ======================
# 主成分分析 (PCA)
# ======================

# 准备PCA分析的特征
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'tempo']
X = df[features]

# 标准化特征 - PCA前必须标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用PCA，降维到3个主成分
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)
# 创建包含主成分的数据框
df_pca = pd.DataFrame(data=principal_components,
                      columns=['PC1', 'PC2', 'PC3'])

# 输出解释方差信息
explained_var = pca.explained_variance_ratio_
print(f"\n主成分解释方差比: {explained_var}")
print(f"累计解释方差: {sum(explained_var):.3f}")

# 1. 二维PCA散点图
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', data=df_pca, alpha=0.7, color='steelblue')
plt.title('PCA of Nigerian Music Features')
plt.xlabel(f'PC1 ({explained_var[0] * 100:.1f}%)')  # 显示解释方差百分比
plt.ylabel(f'PC2 ({explained_var[1] * 100:.1f}%)')
plt.grid(alpha=0.3)  # 添加浅色网格
plt.savefig('pca_visualization.png', dpi=300)
plt.show()

# 2. 三维交互式PCA图
fig = px.scatter_3d(
    df_pca,
    x='PC1',
    y='PC2',
    z='PC3',
    title='3D PCA Visualization of Nigerian Music Features',
    labels={
        'PC1': f'PC1 ({explained_var[0] * 100:.1f}%)',
        'PC2': f'PC2 ({explained_var[1] * 100:.1f}%)',
        'PC3': f'PC3 ({explained_var[2] * 100:.1f}%)'
    },
    opacity=0.7,  # 设置透明度
    height=800  # 设置图表高度
)
# 调整标记样式
fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
# 保存为HTML文件以便交互查看
fig.write_html('3d_pca.html')
fig.show()

# ======================
# K-Means聚类分析
# ======================

# 初始化存储评估指标的列表
wcss = []  # 簇内平方和
silhouette_scores = []  # 轮廓系数
k_range = range(2, 11)  # 测试的聚类数量范围 (2-10)

# 循环测试不同K值
for k in k_range:
    # 创建KMeans模型
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)  # 拟合模型
    wcss.append(kmeans.inertia_)  # 存储WCSS值

    # 轮廓系数需要至少2个聚类
    if k > 1:
        score = silhouette_score(X_scaled, kmeans.labels_)  # 计算轮廓系数
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.4f}")

# 自动选择最优K值（轮廓系数最高的K值）
optimal_k = np.argmax(silhouette_scores) + 2  # 因为k从2开始
print(f"\nOptimal K based on Silhouette Score: {optimal_k}")

# 1. 肘部法则图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')
plt.title('Elbow Method (WCSS)')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
# 标记最优K值位置
plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)

# 2. 轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, 'go-')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
# 标记最优K值位置
plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cluster_evaluation.png', dpi=300)
plt.show()

# 使用最优K值进行最终聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)  # 预测聚类标签
# 将聚类结果添加到原始数据框和PCA数据框
df['cluster'] = clusters
df_pca['cluster'] = clusters

# 可视化聚类结果（基于PCA）
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='cluster',  # 按聚类着色
    data=df_pca,
    palette='viridis',  # 使用viridis调色板
    alpha=0.8,  # 设置透明度
    s=80  # 设置点大小
)
plt.title(f'K-Means Clustering of Nigerian Songs (K={optimal_k})')
plt.xlabel(f'PC1 ({explained_var[0] * 100:.1f}%)')
plt.ylabel(f'PC2 ({explained_var[1] * 100:.1f}%)')
plt.legend(title='Cluster')
plt.grid(alpha=0.3)
plt.savefig('clusters_pca.png', dpi=300)
plt.show()

# ======================
# 聚类结果分析
# ======================

# 分析聚类特征
cluster_features = features + ['popularity', 'length_min']  # 添加额外分析特征
# 按聚类分组计算特征均值
cluster_analysis = df.groupby('cluster')[cluster_features].mean()

print("\nCluster Characteristics:")
print(cluster_analysis)

# 1. 聚类特征均值热力图
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_analysis.T, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Cluster Feature Means')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=300)
plt.show()

# 2. 聚类分布图（每个聚类的歌曲数量）
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=df, palette='viridis')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Songs')
plt.savefig('cluster_distribution.png', dpi=300)
plt.show()

# 3. 聚类与流派关系热力图
plt.figure(figsize=(12, 8))
# 创建聚类与流派的交叉表
cross_tab = pd.crosstab(df['cluster'], df['artist_top_genre'])
sns.heatmap(cross_tab, cmap="Blues", annot=True, fmt='d')
plt.title('Cluster-Genre Relationship')
plt.xlabel('Genre')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('cluster_genre_heatmap.png', dpi=300)
plt.show()

# ======================
# 结果保存
# ======================

# 保存带有聚类标签的完整数据集
df.to_csv('nigerian_songs_with_clusters.csv', index=False)
print("\nAnalysis completed! Results saved.")