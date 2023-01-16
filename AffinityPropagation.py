# # 亲和力传播聚类
# 我们设计了一种名为“亲和传播”的方法，它作为两对数据点之间相似度的输入度量。在数据点之间交换实值消息，直到一组高质量的范例和相应的群集逐渐出现
#
# —源自：《通过在数据点之间传递消息》2007.
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# 定义数据集
X, _ = make_classification(n_samples=1000,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           random_state=4)
# 它是通过 AffinityPropagation 类实现的，要调整的主要配置是将“ 阻尼 ”设置为0.5到1，甚至可能是“首选项".
# 定义模型
model = AffinityPropagation(damping=0.9)
# 匹配模型
model.fit(X)
# 为每个示例分配一个集群
yhat = model.predict(X)
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