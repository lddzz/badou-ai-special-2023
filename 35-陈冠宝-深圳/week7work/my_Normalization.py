import numpy as np
import matplotlib.pyplot as plt


# 归一化的三种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(2*(float(i) - np.mean(x))) / (max(x) - min(x)) for i in x]


# 标准化得到均值为0标准差为1的正态分布数据
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = z_score(l)
print(n)
print(z)
plt.plot(l, cs,'red')
plt.plot(z, cs,'blue')
plt.show()
