# DBSCAN 聚类（其中 DBSCAN 是基于密度的空间聚类的噪声应用程序）涉及在域中寻找高密度区域，并将其周围的特征空间区域扩展为群集。
#
#     …我们提出了新的聚类算法 DBSCAN 依赖于基于密度的概念的集群设计，以发现任意形状的集群。DBSCAN 只需要一个输入参数，并支持用户为其确定适当的值
#
#     -源自：《基于密度的噪声大空间数据库聚类发现算法》，1996

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
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
model = DBSCAN(eps=0.30, min_samples=9)
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