import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib as mpl

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['font.size'] = 12  # 全局字体大小

# 1. 加载数据
file_path = r"C:\Users\ADMIN\Desktop\US-pumpkins.xlsx"
df = pd.read_excel(file_path, sheet_name="US-pumpkins")


# 2. 数据清洗和预处理
# 处理日期列 - 支持多种日期格式
def parse_date(date_str):
    try:
        # 尝试解析不同格式的日期
        for fmt in ('%m/%d/%y', '%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return pd.NaT
    except:
        return pd.NaT


df['Date'] = df['Date'].apply(parse_date)

# 转换价格列为数值类型
price_columns = ['Low Price', 'High Price', 'Mostly Low', 'Mostly High']
for col in price_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 创建月份列用于分析季节性
df['Month'] = df['Date'].dt.month

# 处理大小和颜色列
df['Item Size'] = df['Item Size'].str.strip().replace('', np.nan)
df['Color'] = df['Color'].str.strip().replace('', np.nan)

# 过滤掉缺失价格的数据
df = df.dropna(subset=['Low Price', 'High Price'])

# 3. 使用matplotlib进行可视化
# 可视化1: 不同城市的价格分布（箱线图）
plt.figure(figsize=(14, 8))
cities = df['City Name'].value_counts().index[:5]  # 取前5个最常见城市
filtered_df = df[df['City Name'].isin(cities)]

# 创建箱线图
filtered_df.boxplot(column='Low Price', by='City Name', grid=False, vert=True)
plt.title('不同城市的南瓜最低价格分布', fontsize=15)
plt.suptitle('')  # 移除自动生成的标题
plt.xlabel('城市', fontsize=12)
plt.ylabel('最低价格 (美元)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('city_price_boxplot.png', dpi=300)
plt.show()

# 可视化2: 价格随时间的变化趋势（折线图）
plt.figure(figsize=(14, 8))

# 仅包含有数据的月份
valid_months = df['Month'].dropna().unique()
valid_months.sort()

monthly_prices = df.groupby('Month')['Low Price'].agg(['mean', 'median', 'min', 'max'])

# 只绘制有数据的月份
plt.plot(valid_months, monthly_prices.loc[valid_months, 'mean'], 'o-', label='平均价格', linewidth=2.5)
plt.plot(valid_months, monthly_prices.loc[valid_months, 'median'], 's--', label='中位价格', linewidth=2.5)
plt.fill_between(
    valid_months,
    monthly_prices.loc[valid_months, 'min'],
    monthly_prices.loc[valid_months, 'max'],
    alpha=0.2,
    label='价格范围'
)

# 月份标签
month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
plt.xticks(valid_months, [month_names[int(m - 1)] for m in valid_months])
plt.title('南瓜价格随月份的变化趋势', fontsize=15)
plt.xlabel('月份', fontsize=12)
plt.ylabel('价格 (美元)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('price_trend_monthly.png', dpi=300)
plt.show()

# 4. 使用seaborn进行可视化
# 可视化1: 不同包装类型的平均价格（柱状图）
plt.figure(figsize=(14, 8))
top_packages = df['Package'].value_counts().index[:10]  # 取前10个最常见包装类型
package_df = df[df['Package'].isin(top_packages)]

# 修复警告问题并添加颜色
sns.barplot(
    x='Package',
    y='Low Price',
    data=package_df,
    estimator=np.mean,
    errorbar=None,
    hue='Package',  # 添加hue参数
    palette='viridis',
    dodge=False,  # 禁用分组
    legend=False  # 不显示图例
)
plt.title('不同包装类型的平均南瓜价格', fontsize=15)
plt.xlabel('包装类型', fontsize=12)
plt.ylabel('平均最低价格 (美元)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('package_price_bar.png', dpi=300)
plt.show()

# 可视化2: 品种-颜色价格热力图
plt.figure(figsize=(14, 10))

# 解决热力图空白问题
# 选择最常见的品种和颜色（扩大选择范围）
top_varieties = df['Variety'].value_counts().index[:8]  # 增加品种数量
top_colors = df['Color'].value_counts().index[:8]  # 增加颜色数量
heatmap_df = df[(df['Variety'].isin(top_varieties)) & (df['Color'].isin(top_colors))]

# 确保有足够的数据
if len(heatmap_df) > 0:
    # 创建数据透视表，填充缺失值为0
    heatmap_data = heatmap_df.pivot_table(
        index='Variety',
        columns='Color',
        values='Low Price',
        aggfunc='mean',
        fill_value=0  # 填充缺失值为0
    )

    # 过滤掉全为0的行和列
    heatmap_data = heatmap_data.loc[(heatmap_data != 0).any(axis=1)]
    heatmap_data = heatmap_data.loc[:, (heatmap_data != 0).any(axis=0)]

    if not heatmap_data.empty:
        sns.heatmap(
            heatmap_data,
            cmap='YlOrRd',
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            cbar_kws={'label': '平均最低价格 (美元)'}
        )
        plt.title('不同品种/颜色南瓜的平均价格', fontsize=15)
        plt.xlabel('颜色', fontsize=12)
        plt.ylabel('品种', fontsize=12)
        plt.tight_layout()
        plt.savefig('variety_color_heatmap.png', dpi=300)
        plt.show()
    else:
        print("热力图数据不足，无法生成图表")
else:
    print("没有足够的数据生成热力图")

# 5. 添加南瓜产地分布图
plt.figure(figsize=(14, 8))
top_origins = df['Origin'].value_counts().head(10)
top_origins.plot(kind='bar', color=sns.color_palette('viridis', len(top_origins)))
plt.title('南瓜主要产地分布', fontsize=15)
plt.xlabel('产地', fontsize=12)
plt.ylabel('记录数量', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('origin_distribution.png', dpi=300)
plt.show()

# 6. 库使用体验对比
print("\n库使用体验对比:")
print("| 特性                | matplotlib                          | seaborn                           |")
print("|---------------------|-------------------------------------|-----------------------------------|")
print("| 代码复杂度          | 需要更多代码定制细节                | 高阶封装，代码更简洁             |")
print("| 统计支持            | 需手动计算统计量                    | 内置统计函数（如误差线、聚合）    |")
print("| 默认美观度          | 基础样式较简单                      | 默认配色和样式更现代             |")
print("| 学习曲线            | 较陡峭（需掌握面向对象API）         | 对pandas用户更友好               |")
print("| 定制灵活性          | 极高（可控制每个元素）              | 中等（通过matplotlib底层定制）    |")

print("\n结论:")
print("对于快速探索性分析，seaborn更高效（如用1行代码生成统计图表）。")
print("对于需要精细控制的出版级图表，matplotlib更强大（如自定义每个坐标轴刻度）。")