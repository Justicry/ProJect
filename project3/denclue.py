#-*-coding:utf-8-*-
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter #求数组中每个数字出现了几次

#爬山算法函数，该函数爬到内核密度函数的“坡度”并找到代表密度吸引子的“峰值”
#attr_t:当前点，eps:X(t+1)与X(t)差值阈值
def _hill_climb(attr_t, X, W=None, h=1, eps=1e-7):
    error = 99.  #差值
    prob = 0.
    attr_l1 = np.copy(attr_t)   #X(t+1)
    #三个步骤用于确定吸引子周围的邻域半径
    #radius：密度吸引子的半径
    radius_new = 0.      #新位置的密度吸引子的半径
    radius_old = 0.      #原位置的密度吸引子的半径
    radius_twiceold = 0.
    iters = 0.      #迭代次数
    while True:
        radius_thriceold = radius_twiceold 
        radius_twiceold = radius_old
        radius_old = radius_new
        attr_l0 = np.copy(attr_l1)       #X(t)
        attr_l1, density = step(attr_l0, X, W=W, h=h)  #寻找密度吸引子
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(attr_l1 - attr_l0)  #新位置和原位置的高度差
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new  #密度吸引子半径
        iters += 1  #迭代次数增加
        if iters > 3 and error < eps:   #迭代次数不少于3次，且差值小于X(t+1)与X(t)的差值阈值时迭代结束
            break
    return [attr_l1, prob, radius]

#计算高斯核的方法
def kernelize(x, y, h, degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel

#计算X(t+1)
def step(attr_l0, X, W=None, h=1):
    n = X.shape[0]
    d = X.shape[1]
    weight = 0.  # 每一项的核的权重
    attr_l1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1))  #如果权重为0则赋值1
    else:
        W = W
    for i in range(n):
        kernel = kernelize(attr_l0, X[i], h, d)  #计算得到高斯核
        kernel = kernel * W[i] / (h ** d)        
        weight = weight + kernel
        attr_l1 = attr_l1 + (kernel * X[i])
    attr_l1 = attr_l1 / weight     #计算X（t+1）
    density = weight / np.sum(W)   #计算密度
    return [attr_l1, density]

#denclue算法
class DENCLUE(BaseEstimator, ClusterMixin):
    def __init__(self, h=0.25, eps=1e-8, min_density=0., metric='euclidean'):
        self.h = h      #高斯核的平滑参数，最佳值取决于数据。
        self.eps = eps    #密度吸引子的收敛阈值参数
        self.min_density = min_density    #最小密度
        self.metric = metric    #计算模型中实例之间的距离时使用的度量特征数组

    def fit(self, X, y=None, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        density_attractors = np.zeros((self.n_samples, self.n_features))
        radii = np.zeros((self.n_samples, 1))     #所有半径
        density = np.zeros((self.n_samples, 1))   #所有密度

        #创建初始值
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1))
        else:
            sample_weight = sample_weight
        # 把所有的点初始化为noise点
        labels = -np.ones(X.shape[0])
        #遍历完所有山，即对每个样本点进行attractor和其相应密度的计算
        for i in range(self.n_samples):
            density_attractors[i], density[i], radii[i] = _hill_climb(X[i], X, W=sample_weight,
                                                                      h=self.h, eps=self.eps)
        #初始化聚类图以完成聚类。  
        #边缘的定义是密度吸引子与我们的每个吸引子的半径所定义的邻域相同。
        #构造链接图
        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'instances': [0],
                                      'centroid': np.atleast_2d(density_attractors[0])}
        g_clusters = nx.Graph()
        for j1 in range(self.n_samples):
            g_clusters.add_node(j1, dict={'attractor': density_attractors[j1], 'radius': radii[j1],
                                               'density': density[j1]})
        #构造聚类图
        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):
                if g_clusters.has_edge(j1, j2):   
                    continue
                #利用二范数求两个密度吸引子之间的距离
                diff = np.linalg.norm(g_clusters.node[j1]["dict"]['attractor'] - g_clusters.node[j2]["dict"]['attractor'])
                if diff <= (g_clusters.node[j1]["dict"]['radius'] + g_clusters.node[j1]["dict"]['radius']):
                    g_clusters.add_edge(j1, j2)
        clusters = list(nx.connected_component_subgraphs(g_clusters))
        num_clusters = 0
        # 链接聚类
        for clust in clusters:
            #获得最大密度的吸引子及其位置
            max_instance = max(clust, key=lambda x: clust.node[x]["dict"]['density']) #密度最大的点
            max_density = clust.node[max_instance]["dict"]['density']   #当前类簇的最大密度
            max_centroid = clust.node[max_instance]["dict"]['attractor'] #当前类簇最大密度的位置
            complete = False
            c_size = len(clust.nodes())
            if clust.number_of_edges() == (c_size * (c_size - 1)) / 2.:
                complete = True
            # 构造聚类字典
            cluster_info[num_clusters] = {'instances': clust.nodes(),
                                          'size': c_size,
                                          'centroid': max_centroid,
                                          'density': max_density,
                                          'complete': complete}
            # 如果类的密度小于阈值，则为noise点
            if max_density >= self.min_density:
                labels[clust.nodes()] = num_clusters
            num_clusters += 1
        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self

data = pd.read_csv('iris.csv')
data = np.array(data)  #数组化数据
samples = np.mat(data[:,0:2])  #读取数据的前两列

d = DENCLUE(0.25, 0.0001)  #因为要在I = 0.0001的iris数据集上运行，所以设置阈值为0.0001
d.fit(samples)

true_labels=data[:,-1]
labels=list(set(true_labels))
true_ID=np.zeros((3,50))   #均分标签
#给标签赋值
index=range(len(true_labels))   
for i in range(len(labels)):   
    true_ID[i]=[j for j in index if true_labels[j]==labels[i]]
    right_num=0

#计算正确分类的个数ritght_num
set_len=[] #存放集合大小的数组
for i in range(len(d.clust_info_)):
    bestlens=0
    clust_set = set(d.clust_info_[i]['instances'])
    for j in range(len(labels)):
        true_set=set(true_ID[j]) 
        and_set= clust_set&true_set 
        if len(list(and_set))>bestlens:
            bestlens=len(list(and_set))
    set_len.append(bestlens)
    right_num+=bestlens


labels = [int(l) for l in d.labels_]
num_clusters = len(list(set(labels)))
# 输出聚类数和每个类的个数
print('类别种数: {}'.format(num_clusters))
print('每种类别的个数分别为: ')
print(dict(Counter(labels)))
print('-' * 50)
#输出每个类别的点集
cluster_info = d.clust_info_
for k,v in cluster_info.items():
    print('类别 {}, 密度吸引子: {}, 该密度吸引子周围的点集: {}'.format(k, v['centroid'], v['instances']))
print('-' * 50)
#输出纯度
print("纯度为：{0}".format(float(right_num)/len(samples)))
#绘制聚类图
#matplotlib形状颜色
li=['s','p','*']
color=['c','m','y']

for i in d.clust_info_.keys():
    for j in d.clust_info_[i]['instances']:
        plt.scatter(data[j][0], data[j][1], marker=li[i],c=color[i])
plt.show()