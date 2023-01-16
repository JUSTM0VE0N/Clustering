# 高斯混合模型总结了一个多变量概率密度函数，顾名思义就是混合了高斯概率分布
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)
# 定义数据集
X, _ = make_classification(n_samples=1000,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           random_state=4)
# 定义模型
model = GaussianMixture(n_components=2)
# 模型拟合与聚类预测
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    # 创建这些样本的散布
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
plt.show()