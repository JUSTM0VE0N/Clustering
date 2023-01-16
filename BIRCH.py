# BIRCH 聚类（ BIRCH 是平衡迭代减少的缩写，聚类使用层次结构)包括构造一个树状结构，从中提取聚类质心。
# BIRCH
# 递增地和动态地群集传入的多维度量数据点，以尝试利用可用资源（即可用内存和时间约束）产生最佳质量的聚类。
#
# —源自：《 BIRCH ：1996年大型数据库的高效数据聚类方法》

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
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
model = Birch(threshold=0.01, n_clusters=2)
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