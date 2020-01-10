# ^(*￣(oo)￣)^
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import operator
import csv

N_CITIES = 48          #城市数
POP_SIZE = 200         #每一代个体数
ELITE_SIZE = 100       #保留的精英个体数量
MUTATE_RATE = 0.002    #基因变异概率
N_GENERATIONS = 1000   #生成子代数量

#勾股定理求两城市间距离
def distance(city1, city2):
        x = abs(city1[0] - city2[0])
        y = abs(city1[1] - city2[1])
        distance = np.sqrt((x ** 2) + (y ** 2))
        return distance

#适应度函数，定义为总路径的倒数
def fitness(route):
    fitness= 0.0
    path_distance = 0
    for i in range(0, len(route)):
        city_from = route[i]
        city_to = None
        if i + 1 < len(route):
            city_to = route[i + 1]

        #判断是最后一个城市时，计算到第一个城市的距离
        else:
            city_to = route[0]
        dist = distance(city_from, city_to)
        path_distance += dist
    fitness = 1 / float(path_distance)
    return fitness

#创建一个城市的随机访问路径 --->个体
def create_routes(citylist):
    route = random.sample(citylist, len(citylist))
    return route

#创建足够多的个体 --->初始种群
def initial_population(population_size, citylist):
    population = []
    for i in range(0, population_size):
        population.append(create_routes(citylist))
    return population

#生成种群内每个个体的适应度，并将它们降序排列，个体用在种群中的序号代替
def rank_routes(population):
    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = fitness(population[i])
    return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = True)

#筛选个体
def select_routes(rank_result, elite_size):
    select_result = []

    #将排序完的个体转换为DataFrame格式，按顺序计算出累积和cun_sum，再得出所占总适应度百分比cum_perc
    #格式样例：
    #       Index   Fitness  cum_sum  cum_perc
    # 0       1       0.5      0.5      50.0
    # 1       0       0.3      0.8      80.0
    # 2       2       0.2      1.0     100.0
    df = pd.DataFrame(np.array(rank_result), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    #先保留精英个体，将剩下个体进行轮盘赌选择
    for i in range(0, elite_size):
        select_result.append(rank_result[i][0])
    for i in range(0, len(rank_result) - elite_size):
        pick = 100*random.random()
        for i in range(0, len(rank_result)):
            if pick <= df.iat[i,3]:
                select_result.append(rank_result[i][0])
                break
    return select_result

#将筛选完的个体的序号转化为具体的路由信息 --->生成交配池
def matingPool(population, select_result):
    matingpool = []
    for i in range(0, len(select_result)):
        index = select_result[i]
        matingpool.append(population[index])
    return matingpool

#两个个体间通过交叉产生后代
def crossover(parent1, parent2):
    child = []
    p1 = []
    P2 = []

    #从第一个个体中随机选择一段城市，拼接第二个个体中所有不重复的城市，生成的新的个体
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    start = min(geneA, geneB)
    end = max(geneA, geneB)
    for i in range(start, end):
        p1.append(parent1[i])
    p2 = [item for item in parent2 if item not in p1]
    child = p1 + p2
    return child

#在整个种群上，先保留精英个体，剩下的个体两两交叉直到生成足够的数量
def crossover_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,elite_size):
        children.append(matingpool[i])

    for i in range(0, length):
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#个体中的每个城市都有概率与其他城市发生顺序交换  ---> 变异
def mutate(route, mutation_rate):
    for swap1 in range(len(route)):
        if(random.random() < mutation_rate):
            swap2 = int(random.random() * len(route))
            city1 = route[swap1]
            city2 = route[swap2]
            route[swap1] = city2
            route[swap2] = city1
    return route

#种群中所有个体都有机会变异
def mutate_population(population, mutation_rate):
    mutate_population = []

    for route in range(0, len(population)):
        mutate_route = mutate(population[route], mutation_rate)
        mutate_population.append(mutate_route)
    return mutate_population

#繁衍：父代种群 => 排序 -> 选择 -> 交配池 -> 交叉 -> 变异 =>子代种群
def multiply(current_generation, elite_size, mutation_rate):
    rank = rank_routes(current_generation)
    select = select_routes(rank, elite_size)
    mate = matingPool(current_generation, select)
    children = crossover_population(mate, elite_size)
    next_generation = mutate_population(children, mutation_rate)
    return next_generation

#绘图
def plotting(route, best_route, show_time):
    x = []
    y = []
    plt.cla()
    for i in range(0, len(route)):
        x.append(route[i][0])
        y.append(route[i][1])
    x.append(route[0][0])
    y.append(route[0][1])
    plt.plot(x, y, 'r-s', markerfacecolor = 'y')
    plt.text(0, 5500, "Total distance = %.3f" % best_route, fontdict={'size': 20, 'color': 'blue'})
    plt.pause(show_time)

#读取城市坐标
def read_city_file(citylist, N_CITIES):
    csv_file = csv.reader(open('city-' + str(N_CITIES) + '.csv'))
    for row in csv_file:
        citylist.append(row)
    for i in range(0, len(citylist)):
        for j in range(0, 2):
            citylist[i][j] = int(citylist[i][j])
    return citylist

#输出结果到文件中
def output_city_path(route, best_route, N_CITIES):
    output = open('result-' + str(N_CITIES) + '.txt' ,'w')
    output.write('----------Distance----------\n\n')
    output.write(str(best_route) + '\n\n')
    output.write('---------Circle Path--------\n\n')
    for i in route:
        output.write(str(i) + '\n')
    output.write(str(route[0]))

#繁衍
def multiply_of_each_generation(generation, population, elite_size, mutation_rate):
    population = multiply(population, elite_size, mutation_rate)
    best = 1 / rank_routes(population)[0][1]
    print('Gen:', generation,'|| best fit:', best)
    #注释下一行可以只绘制最终结果的图像
    plotting(population[0], best, 0.01)
    return population

#繁衍最后一代
def multiply_of_last_generation(generation, population, city_size, elite_size, mutation_rate):
    population = multiply(population, elite_size, mutation_rate)
    best = 1 / rank_routes(population)[0][1]
    print('Gen:', generation,'|| best fit:', best)
    #最后一张绘图停留两秒后消失
    plotting(population[0], best, 2)
    print("Final Distance = %.3f" % best)
    output_city_path(population[0], best, city_size)

def GA(city_size, pop_size, elite_size, mutation_rate, generation):
    citylist = []
    read_city_file(citylist, city_size)
    population = initial_population(pop_size, citylist)
    for gen in range(1, generation):
        population = multiply_of_each_generation(gen, population, elite_size, mutation_rate)
    multiply_of_last_generation(generation, population, city_size, elite_size, mutation_rate)

#-------------------------------------------——————————————————————------------------------------#

GA(N_CITIES, POP_SIZE, ELITE_SIZE, MUTATE_RATE, N_GENERATIONS)

#-----------------------------------------------------------------------------------------------#
