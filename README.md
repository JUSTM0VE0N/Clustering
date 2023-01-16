# 在 scikit-learn 机器学习库的Python中实现、适配和使用顶级聚类算法。
## scikit-learn库安装
`sudo pip install scikit-learn`
### 检查scikit-learn版本
```
import sklearn
print(sklearn.__version__)
```

## 聚类数据集
使用 make _ classification ()函数创建一个测试二分类数据集。数据集将有1000个示例，每个类有两个输入要素和一个群集。这些群集在两个维度上是可见的，因此我们可以用散点图绘制数据，并通过指定的群集对图中的点进行颜色绘制。

## 聚类算法
1. 亲和力传播 -- AffinityPropagation.py
2. 聚合聚类 -- AgglomerationClustering.py
3. BIRCH 聚类 -- BIRCH.py
4. DBSCAN 聚类 -- DBSCAN.py
5. K-均值聚类 -- K-mean.py
6. Mini-Batch K-均值聚类 -- Mini-Batch K-Mean.py
7. 均值漂移聚类 -- MeanShift.py
8. OPTICS 聚类 -- OPTICS.py
9. 光谱聚类 -- Spectral.py
10. 高斯混合模型 -- GaussianMixture.py
