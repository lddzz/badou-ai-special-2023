"""
pca 主成分分析代码实现
"""
import numpy as np

class PCA(object):
    '''
    用pca求样本矩阵input_matrix的 K 阶降维矩阵 Z；
    输入的样本矩阵的要求：shape=（m,n)
    '''
    def __init__(self,input_matrix,K):
        '''

        :param input_matrix: the input sample matrix;
        :param K:the order of the ouput matrix
        '''
        '''Determine the shape of the input matrix'''
        if len(input_matrix.shape)>2:
            tempMatrix=input_matrix.reshape(input_matrix.shape[0]*input_matrix.shape[1],input_matrix.shape[2])
            self.format='pic'
            self.source_row=input_matrix.shape[0]
            self.source_col=input_matrix.shape[1]
        self.inputMatrix=tempMatrix
        self.outputOrder=K
        '''used to  the centralization of the input matrix'''
        self.centrInputMatrix=[]
        '''the convaiance matrix of the sample'''
        self.convC=[]
        '''the dimensionality reduction transformation matrix of the sample'''
        self.dimReductTranMatrix=[]
        '''the dimensionality reduction matrix of the sample'''
        self.dimReductMatrix=[]

        self.centrInputMatrix=self._centralized()
        self.convC=self._cov()
        self.dimReductTranMatrix=self._U()
        self.dimReductMatrix=self._Z()

    def _centralized(self):
        '''Centralizaiont of the matrix'''
        print('sample matrix:\n',self.inputMatrix)
        centrInputMatrix=[]
        '''the characteristic mean of the sample set'''
        mean=np.array([np.mean(attr) for attr in self.inputMatrix.T])
        print('the characteristic mean of the sample set\n',mean)
        '''Centralization of the sample set'''
        centrInputMatrix=self.inputMatrix-mean
        print('the centralization of the sample inputMatrix:\n',centrInputMatrix)
        return centrInputMatrix

    def _cov(self):
        '''the convariance of the inputmatrix'''

        '''the total number of the samples in the sample set'''
        ns=np.shape(self.centrInputMatrix)[0]
        '''the convariance of the sample matrix'''
        c=np.dot(self.centrInputMatrix.T,self.centrInputMatrix)/(ns-1)
        print('the convariance of the inputMatrix:\n',c)
        return c

    def _U(self):
        '''FInd the dimensionality reduction transformation matrix of the matrix inputMatrix.
        ---shape=(n,k),
        ---n is the total number of eigendimensions of the matrix inputmatrix;
        ---k is the characteristic dimension of the dimensionality reduction transformation matrix

            first to find the eigenvalue and eigenvector of the convariance of the inputMatrix.'''

        #the eigenvalue is assigned to a;
        #the eigenvector is assigned to b;
        a,b=np.linalg.eig(self.convC)
        print('the eigenvalue of the convariance of convC of the sample:\n',a)
        print('the eigenvector of the convariance of convC of the sample:\n',b)

        #the index of the first k eigenvalues in descending order is given.
        ind=np.argsort(-1*a)
        #the dimemsionality reduction transformation matrix  of order k dimensionality reduction is constructed.
        UT=[b[:,ind[i]] for i in range(self.outputOrder)]
        U=np.transpose(UT)
        print('%d order dimensionality reduction transformation matrix:\n'%self.outputOrder,U)
        return U
    def _Z(self):
        '''Find the dimensionality reduction transformation matrix in terms of z=xu,
        ---shape=(m,k),
        ---m is the total number of the sample;
        ---k is the total number of characteristic dimension of the  dimensionality reduction matrix '''
        Z=np.dot(self.inputMatrix,self.dimReductTranMatrix)
        #the length of the shape of the self.inputMatrix>2,we consider the input matrix is a pic
        if self.format=='pic':
            Z=np.uint8(np.absolute(Z))
            Z=Z.reshape(self.source_row,self.source_col,Z.shape[-1])
        print('inputmatrix shape:',np.shape(self.inputMatrix))
        print('dimReductTranMatrix shape:',np.shape(self.dimReductTranMatrix))
        print('ouput matrix shape:',np.shape(Z))
        print('the dimensionality reduction matrix:',Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K=np.shape(X)[1]-1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)





