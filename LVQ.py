import numpy as np

class LVQ:
    def __init__(self,n_vectors=5,learning_rate = 0.1,random_state = 42,max_iter = 1000):
        '''
        :param learning_rate:学习率
        :param n_vectors:原型向量的个数
        :param max_iter:最大迭代次数
        :param random_state:随机数种子
        '''
        self.vectorNum = n_vectors
        self.lr = learning_rate
        self.learningVectors = None
        self.learningVectorslabels = None
        self.seed = random_state
        self.max_iter = max_iter

    def dist(self,x,y):
        return np.linalg.norm(x-y)

    def fit(self,X,y):

        np.random.seed(self.seed)
        m,n = X.shape
        idxes = np.random.randint(0,m,self.vectorNum)
        # 初始化学习向量
        # 随机初始化
        learning_vectors,learning_vectors_labels = X[idxes],y[idxes]

        for _ in range(self.max_iter):

            # 随机选取一个样本
            idx = np.random.randint(0,m)
            xj,yj = X[idx],y[idx]
            distances = np.zeros(self.vectorNum)
            for i in range(self.vectorNum):
                distances[i] = self.dist(xj,learning_vectors[i])
            i_astrix = np.argmin(distances)
            if yj == learning_vectors_labels[i_astrix]:
                learning_vectors[i_astrix] += self.lr * (xj - learning_vectors[i_astrix])
            else:
                learning_vectors[i_astrix] -= self.lr * (xj - learning_vectors[i_astrix])

        self.learningVectors = learning_vectors
        self.learningVectorslabels = learning_vectors_labels



