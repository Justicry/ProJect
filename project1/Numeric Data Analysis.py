import numpy as np
import matplotlib.pyplot as plt

import math

f = open("MAGIC.txt", "r")
row = f.readlines()

list = []
for i in range(len(row)):
    column_list = row[i].strip().split(",")  # 每一行以，为分隔符split后是一个列表
    column_list.pop()#去掉最后一行属性g
    list.append(column_list)  # 加入list_source
a=np.array(list)#转化为np数组
a=a.astype(float)#转换为浮点类型
MultivariateMeanVector=np.mean(a,axis=0)#均值向量
print("多元均值向量为：",MultivariateMeanVector)


center=a-MultivariateMeanVector#中心化多元均值向量
InnerProduct=np.dot(center.T,center)
print("内积计算协方差矩阵为：")
print(InnerProduct/len(center))#求内积

OuterProduct=0
for i in range(len(center)):
    OuterProduct = OuterProduct+center[i].reshape(len(center[0]),1)*center[i]
print("外积计算协方差矩阵为：")
print(OuterProduct/len(center))#求外积

t=center.T
def cosVector(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return;
        
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    print("夹角余弦值为："+str(result1/((result2*result3)**0.5))) #结果显示
cosVector(t[0],t[1])

print("散点图：")
picture = plt.figure()
plt.scatter(t[0],t[1])
plt.xlabel('Attribute 1') #设置X轴标签
plt.ylabel('Attribute 2') #设置Y轴标签
plt.show()

# 若属性1为正态分布。则其 x 是均值和 标准差的函数
def normfun(x, mean, sd):
    pdf = np.exp(-((x - mean) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
    return pdf

#计算均值和标准差
mean=np.mean(a,axis=0)[0]#计算第一列均值
sd=np.var(a.T[0])#计算第一列方差
pic = plt.figure()

# 绘制正态分布概率密度函数
print("概率密度函数图像：")
x = np.linspace(mean - 3 * sd, mean + 3 * sd, 50)
y_pic = np.exp(-(x - mean) ** 2 / (2 * sd ** 2)) / (math.sqrt(2 * math.pi) * sd)
plt.plot(x, y_pic, "b-.", linewidth=1)
plt.vlines(mean, 0, np.exp(-(mean - mean) ** 2 / (2 * sd ** 2)) / (math.sqrt(2 * math.pi) * sd), colors="r",
           linestyles="dashed")
plt.vlines(mean + sd, 0, np.exp(-(mean + sd - mean) ** 2 / (2 * sd ** 2)) / (math.sqrt(2 * math.pi) * sd),
           colors="k", linestyles="dotted")
plt.vlines(mean - sd, 0, np.exp(-(mean - sd - mean) ** 2 / (2 * sd ** 2)) / (math.sqrt(2 * math.pi) * sd),
           colors="k", linestyles="dotted")
plt.xticks([mean - sd, mean, mean + sd], ['μ-σ', 'μ', 'μ+σ'])
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 2')
plt.show()


list=[]
for i in range(len(a[0])):
    list.append(np.var(a.T[i]))#计算每一列的方差
Maxsd=list.index(max(list))
Minsd=list.index(min(list))
print("方差最大的属性是：第",Maxsd+1,"个属性")
print("值为：",list[Maxsd])
print("方差最小的属性是：第",Minsd+1,"个属性")
print("值为：",list[Minsd])

#求矩阵一对属性的协方差
Cov={}
for i in range(9):
    for j in range(i+1,10):
        st=str(i+1)+'-'+str(j+1)
        Cov[st]= np.cov(a.T[i],a.T[j])[0][1]#遍历求协方差
print("协方差最小的两个属性为：",min(Cov, key=Cov.get))
print("协方差最大的两个属性为：",max(Cov, key=Cov.get))

