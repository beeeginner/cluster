import numpy as np

class Kmeans:
    def __init__(self,k=3,dim=3,seed=42,max_iter = 1000):
        '''
        :param k:分类的簇数目
        :param dim: 样本的维度
        :param seed: 随机种子
        :param max_iter 最大迭代次数
        '''

        self.k = k
        self.seed = seed
        self.U = np.zeros([k,dim])
        self.max_iter = max_iter
        self.Cluster = None
        self.labels = None

    def _dist(self,x,y):
        # 欧几里得距离
        tmp = abs(x - y)
        return np.sqrt(np.sum(tmp**2))

    def _train(self,X):
        # 读取样本数
        m = X.shape[0]
        cluster = None
        np.random.seed(self.seed)
        idxes = np.random.randint(0, m, size=self.k)
        self.U = X[idxes]

        for _ in range(self.max_iter):
            cluster = [[] for _ in range(self.k)]
            # 记录到每个样本簇的距离
            distances = np.zeros(self.k)
            for j in range(m):
                for k0 in range(self.k):
                    distances[k0] = self._dist(X[j], self.U[k0])
                # 确定簇标记
                lambdai = np.argmin(distances)
                # 把当前样本划分到这个蔟里面
                cluster[lambdai].append(j)
            # 记录均值向量是否更新
            cnt = 0
            for i in range(self.k):
                # 更新均值向量
                u = np.mean(X[cluster[i]],axis=0)
                self.U[i] = u
                if (u!=self.U[i]).all():
                    cnt+=1
            if cnt == 0:
                # 早停机制当均值向量都不更新的时候退出循环
                break
        # 返回蔟
        return cluster

    def fit(self,X,return_dict = False):

        self.labels = np.zeros(X.shape[0])
        cluster = self._train(X)
        res = dict(enumerate(cluster))
        if return_dict == True:
            return res
        for label in res.keys():
            for idx in res[label]:
                self.labels[idx] = label

if __name__=='__main__':
    from sklearn.datasets import load_iris
    from sklearn.metrics import silhouette_score

    # 加载 iris 数据集
    iris = load_iris()
    X, y = iris.data, iris.target

    # 创建 KMeans 对象并进行拟合
    kmeans = Kmeans(k=3,dim=4,seed=42,max_iter=1000)
    kmeans.fit(X)


    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, kmeans.labels)
    print("轮廓系数:", silhouette_avg)






