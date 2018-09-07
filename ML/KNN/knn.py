#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
def file2matrix(filename):
    """
    特征转化为numpy
    :param filename:
    :return:
    """
    fr = open(filename,'r',encoding='utf-8')
    lines=fr.readlines()
    n = len(lines)
    matrix = np.zeros((n,3))
    labels = []
    for index,line in enumerate(lines):
        # print(index)
        listFromLine = line.strip('\n').split()
        assert len(listFromLine) == 4
        labels.append(int(listFromLine[-1]))
        matrix[index,:] = listFromLine[:3]

    return matrix,labels

def plot(matrix,labels):
    """
    画数据散点图
    :param matrix:
    :param labels:
    :return:
    """
    fig = plt.figure()
    #111代表画满整个画布，一行一列第一个
    ax = fig.add_subplot(111)
    #s为点的大小，c为点的颜色
    ax.scatter(matrix[:, 0], matrix[:, 1], s=15.0 * np.array(labels), c=15.0 * np.array(labels))
    plt.show()

def aotuNorm(matrix):
    """
    归一化特征
    :param matrix:
    :return:
    """
    maxArray = np.max(matrix,axis=0)
    minArray = np.min(matrix,axis=0)
    rangeArray = maxArray-minArray
    normMatrix = (matrix-minArray)/rangeArray #传播
    return normMatrix,rangeArray,minArray

def classfy(inX,matrix,labels,k):
    """
    分类
    :param inX:
    :param matrix:
    :param labels:
    :param k:
    :return:
    """
    diffmatrix = matrix - inX
    diffmatrix = diffmatrix ** 2
    diffArray = np.sum(diffmatrix,axis=1)
    diffArray = diffArray ** 0.5
    indexArray = np.argsort(diffArray)
    from collections import defaultdict
    dic_label = defaultdict(int)

    for index in indexArray[:k]:
        label = labels[index]
        dic_label[label] += 1

    sort_list = sorted(dic_label.items(),key = lambda x:x[1],reverse=True)
    return sort_list[0][0],indexArray

def plotClassfy(inX,matrix,indexArray,k):
    """
    绘制输入的点与knn的图
    :param inX:
    :param matrix:
    :param indexArray:
    :param k:
    :return:
    """
    knnMatrix = np.zeros((k, 3))
    knnLabels = []
    knncount = 0
    for index in indexArray[:k]:
        knnMatrix[knncount,:] = matrix[index,:]
        knncount+=1
        label = labels[index]
        knnLabels.append(label)
    n = len(indexArray)
    unknnMatrix = np.zeros((n-k, 3))
    unknnLabels = []
    unknncount = 0
    for index in indexArray[k:]:
        unknnMatrix[unknncount, :] = matrix[index, :]
        unknncount += 1
        label = labels[index]
        unknnLabels.append(label)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # s为点的大小，c为点的颜色
    ax.scatter(knnMatrix[:, 0], knnMatrix[:, 1], s=50, c=15.0 * np.array(knnLabels),marker='s')
    ax.scatter(unknnMatrix[:, 0], unknnMatrix[:, 1], s=15, c=15.0 * np.array(unknnLabels), marker='.')
    ax.scatter(inX[0], inX[1], s=100, c=0.0, marker='x')
    plt.show()

if __name__=='__main__':
    matrix,labels = file2matrix('datingTestSet2.txt')
    plot(matrix,labels)
    normMatrix,rangeArray,minArray = aotuNorm(matrix)
    plot(normMatrix,labels)
    inX = np.array([0.4,0.25,0.2])
    label,indexArray = classfy(inX,normMatrix,labels,5)
    print(label)
    plotClassfy(inX,normMatrix,indexArray,5)