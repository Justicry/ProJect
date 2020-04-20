import numpy as np
import pandas as pd

data=pd.read_csv('iris.csv')
data=np.array(data)
data=np.mat(data[:,0:4])

length=len(data)

#用核函数计算核矩阵
k=np.mat(np.zeros((length,length)))   #创建一个150*150的零矩阵
for i in range(0,length):
    for j in range(i,length):
        k[i,j]=(np.dot(data[i],data[j].T))**2 #矩阵外积的二次方
        k[j,i]=k[i,j]

len_k=len(k)

#中心化核矩阵K
I=np.eye(len_k) #单位矩阵·
one=np.ones((len_k,len_k))  #全为1的矩阵
A=I-1.0/len_k*one
centered_k=np.dot(np.dot(A,k),A) #中心化核矩阵
print("中心化核矩阵结果：\n",centered_k)

#标准化核矩阵
W=np.zeros((len_k,len_k)) #标准化K左右两侧的矩阵
for i in range(0,len_k):
    W[i,i]=centered_k[i,i]**(-0.5)
normalized_k=np.dot(np.dot(W,centered_k),W)#标准化核矩阵
print ("标准化核矩阵结果：\n",normalized_k)

#第二问：使用齐次二次核将每个点x变换到特征空间ϕ（x）。 并将其标准化。

#计算ϕ(f)
f=np.mat(np.zeros((length,10)))
for i in range(0,length):
    for j in range(0,4):  #计算二次项
        f[i,j]=data[i,j]**2
    for m in range(0,3):  #计算交叉项
        for n in range(m+1,4):
            j=j+1
            f[i,j]=2**0.5*data[i,m]*data[i,n]

#高维特征空间计算内积得到核矩阵
k_f=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        k_f[i,j]=(np.dot(f[i],f[j].T))
        k_f[j,i]=k_f[i,j]

len_k_f=len(k_f)
#中心化核矩阵K
I=np.eye(len_k_f) #单位矩阵·
one=np.ones((len_k_f,len_k_f))  #全为1的矩阵
A=I-1.0/len_k_f*one
centered_k_f=np.dot(np.dot(A,k_f),A) #中心化核矩阵

#标准化核矩阵
W=np.zeros((len_k_f,len_k_f)) #标准化K左右两侧的矩阵
for i in range(0,len_k_f):
    W[i,i]=centered_k_f[i,i]**(-0.5)
normalized_k_f=np.dot(np.dot(W,centered_k_f),W)#标准化核矩阵
print ("变换到特征空间后标准化核矩阵结果：\n",normalized_k_f)

