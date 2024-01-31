# coding=utf-8

import numpy as np


class PCA():
    def __init__(self,n_components):
        self.n_components =n_components

    def fit(self,X):
        self.features = X.shape[1] #多少个特征
        X=X-X.mean(axis=0) #先中心化
        self.covariance=np.dot(X.T,X)/X.shape[0]

        #求特征向量和特征值
        eig_vals,eig_vec=np.linalg.eig(self.covariance)

        # 降序排序获得特征值的序号
        idx=np.argsort(-eig_vals)

        # 降维
        self.components_=eig_vec[:,idx[:self.n_components]]

        # 得到的矩阵对X进行降维
        return np.dot(X,self.components_)


# 调用
pca = PCA(n_components=2)
X = np.random.randint(-21, 20, size=(6, 5))
newX = pca.fit(X)
print(newX)  # 输出降维后的数据
