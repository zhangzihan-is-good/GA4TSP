import numpy as np
import random

#最初的四个个体，在0-9号城市中旅行的顺序
s1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
s2 = [4, 5, 6, 9, 2, 1, 7, 8, 3, 0]
s3 = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6]
s4 = [1, 2, 0, 4, 5, 7, 6, 8, 9, 3]
#十个城市的初始笛卡尔坐标
locs = [(12, 13), (45, 62), (37, 55), (71, 52), (7, 8), (2, 10), (6, 55), (17, 44), (62, 33), (44, 77)]
# 交叉率
p_c = 0.7
# 变异率
p_m = 0.3
def CacDistance(a, b):#计算两点间的距离
    a = np.array(a)
    b = np.array(b)
    c = a - b
    distance = np.sqrt(np.sum(c * c))
    return distance

def Cross(p1, p2):#交叉操作
    a = np.array(p1).copy()
    b = np.array(p2).copy()
    begin = random.randint(0, 4)
    end = random.randint(5, 9)
    temp = []
    for i in range(begin, end + 1):
        for j in range(0,10):
            if a[i] == b[j]:
                temp.append(j)
    m = 0
    for i in range(begin, end + 1):
        t = a[i]
        a[i] = b[temp[m]]
        b[temp[m]] = t
        m=m+1
    return a,b

def Variation(s):#变异操作，随机互换染色体上的两个基因
    index1, index2 = random.sample(range(10), 2)
    temp = s[index1]
    s[index1] = s[index2]
    s[index2] = temp
    return s

def cost(s):#输入路径，返回路径长度的负值
    n = len(locs)
    dis_mat = np.zeros([10, 10])#获取邻接矩阵
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = CacDistance(locs[i], locs[j])
            dis_mat[i, j] = dist
    for i in range(n):
        dis_mat[:, i] = dis_mat[i, :]
    n = len(s)
    cost = 0
    for i in range(n):
        cost += dis_mat[s[i], s[(i + 1) % n]]
    return -cost

def CacAdap(ada):# adap n*4,n为行数，每行包括：个体下标,路径长度,适应度,选择概率,累积概率
    # 计算每一个个体的适应度,选择概率
    adap = []
    psum = 0
    # 计算适应度
    i = 0
    for p in ada:
        icost = np.exp(p)
        psum += icost
        # 添加个体下标
        adap.append([i])
        # 添加路径长度的相反数
        adap[i].append(p)
        # 添加适应度
        adap[i].append(icost)
        i += 1
    # 计算选择概率
    for p in adap:
        # 添加选择概率和累积概率，这里累积概率暂时等于选择概率，后面会重新计算赋值
        p.append(p[2] / psum)
        p.append(p[3])
    # 计算累计概率
    n = len(adap)
    for i in range(1, n):
        p = adap[i][4] + adap[i - 1][4]
        adap[i][4] = p
    return adap

def Chose(adap):#轮盘选择操作
    chose = []
    # 选择次数
    epochs = 20
    n = len(adap)
    for a in range(epochs):
        p = random.random()
        if adap[0][4] >= p:
            chose.append(adap[0][0])
        else:
            for i in range(1, n):
                if adap[i][4] >= p and adap[i - 1][4] < p:
                    chose.append(adap[i][0])
                    break
    return chose

def Cross_Variation(chose, population):#交叉变异操作
    # 交叉变异操作
    chose_num = len(chose)
    sample_times = chose_num // 2
    for i in range(sample_times):
        index1, index2 = random.sample(chose, 2)
        # 参与交叉的父结点
        parent1 = population[index1]
        parent2 = population[index2]
        # 这两个父结点已经交叉，后面就不要参与了，就像这两个人以及结婚，按规矩不能在与其他人结婚了，故从采样样本中移除
        chose.remove(index1)
        chose.remove(index2)

        p = random.random()
        if p_c >= p:
            child1, child2 = Cross(parent1, parent2)
            p1 = random.random()
            p2 = random.random()
            if p_m > p1:
                child1 = Variation(child1)
            if p_m > p2:
                child2 = Variation(child2)
            population.append(list(child1))
            population.append(list(child2))
    return population

population = [s1, s2, s3, s4]
epochs = 51
i = 0
while i < epochs:
    ada = []
    # 计算路径长度的相反数
    for p in population:
        icost = cost(p)
        ada.append(icost)

    # 根据路径长度的相反数计算累积概率
    adap = CacAdap(ada)
    min_cost = max(ada)
    # 选择操作
    chose = Chose(adap)
    # 交叉变异
    population = Cross_Variation(chose, population)

    if i % 10 == 0:
        print('epoch %d: loss=%.2f' % (i, -min_cost))
    i += 1
    if i == epochs:
        # 输出最优解
        p_len = len(population)
        for index in range(p_len):
            if ada[index] == min_cost:
                print('最优路径:')
                print(population[index])
                print('代价大小:')
                print(-min_cost)
                break
