import pandas as pd
from fun import *
import numpy as np
import random
import copy
import timeit


def get_fitness(ori_w_p):
    # 设置适应度函数
    return -fun(ori_w_p[0], ori_w_p[1])


def initial_group(num):
    # 随机起始灰狼种群，根据输入的num进行设置，并输出其（x,y,z）值
    wolves_dict = {}
    ori_w_range_x = np.arange(x1, x2, 0.1)
    ori_w_range_y = np.arange(y1, y2, 0.1)
    for i in range(num):
        ori_w_p_x = random.choice(ori_w_range_x)
        ori_w_p_y = random.choice(ori_w_range_y)
        ori_w_p = [ori_w_p_x, ori_w_p_x]
        wolves_dict[i] = {'x': ori_w_p_x, 'y': ori_w_p_y, 'z': get_fitness(ori_w_p)}
    return wolves_dict


def sort_wolves(wolves_dict):
    # 将灰狼种群分类，分为a,b,c,d四个等级
    wolves = pd.DataFrame(wolves_dict)
    wolves = wolves.T
    wolves.sort_values(by='z', ascending=False, inplace=True)
    wolves.reset_index(drop=True, inplace=True)
    wolves_level = []
    wolves_level.extend(list('a'))
    wolves_level.extend(list('b'*4))
    wolves_level.extend(list('c'*16))
    wolves_level.extend(list('d'*29))
    wolves_id = ['wolf_'+str(i) for i in range(len(wolves_level))]
    wolves['level'] = wolves_level
    wolves['id'] = wolves_id
    wolves_T = wolves.T
    return wolves_T.to_dict()


def get_iteration_param(t_n, total_nums):
    v_r1 = random.random()
    v_r2 = random.random()
    v_c = 2 * v_r2
    prey_range_x = np.arange(-30, 15, 0.1)
    prey_range_y = np.arange(-30, 15, 0.1)
    v_xp = np.array([random.choice(prey_range_x), random.choice(prey_range_y)])
    v_D = np.absolute(v_c * v_xp - v_xp)
    v_a = 2 - t_n/total_nums * 2
    v_A = 2 * v_a * v_r1 - v_a
    return v_A, v_D


def get_iteration_param_2(t_n, total_nums, wolve_a):
    # 拿到迭代参数
    v_r1 = random.random()
    v_r2 = random.random()
    v_c = 2 * v_r2
    v_xp = np.array([wolve_a['x'], wolve_a['y']])
    v_D = np.absolute(v_c * v_xp - v_xp)
    v_a = 2 - t_n/total_nums * 2
    v_A = 2 * v_a * v_r1 - v_a
    return v_A, v_D


def omega_find_leader(wolves_group):
    # 发现omega狼周围的领导者
    # 离其最近的三匹高级狼
    # a等级一匹
    # b等级一匹，距离最短的
    # c等级一匹，距离最短的
    a_w = []
    b_w = []
    c_w = []
    d_w = []

    for i in wolves_group.keys():
        if wolves_group[i]['level'] == 'a':
            a_w.append(wolves_group[i])
        elif wolves_group[i]['level'] == 'b':
            b_w.append(wolves_group[i])
        elif wolves_group[i]['level'] == 'c':
            c_w.append(wolves_group[i])
        elif wolves_group[i]['level'] == 'd':
            d_w.append(wolves_group[i])

    for i in d_w:
        i['leader_group'] = [a_w[0]['id']]
        i['leader_group'].append(find_shortest_leader(i, b_w)['id'])
        i['leader_group'].append(find_shortest_leader(i, c_w)['id'])
    return d_w


def find_shortest_leader(worker, leaders):
    leader_id = []
    leader_dist = []
    for l in leaders:
        dist = np.linalg.norm(
            np.array([l['x'], l['y']]) - np.array([worker['x'], worker['y']]))
        leader_id.append(l)
        leader_dist.append(dist)
    return leader_id[leader_dist.index(min(leader_dist))]


def iteration(wolves_group, t_n, total_nums, wolve_a):
    # 根据文本中的公式进行迭代
    wolves_group_cg = wolves_group
    for i in wolves_group_cg:
        ind = wolves_group_cg[i]
        if ind['level'] == 'a':
            x_p = np.array([ind['x'], ind['y']])
            v_A, v_D = get_iteration_param_2(t_n, total_nums, wolve_a)
            v_x = x_p - (v_A * v_D)
            if(v_x[0] > x2):
                v_x[0] = x2
            elif(v_x[0] < x1):
                v_x[0] = x1
            if(v_x[1] > y2):
                v_x[1] = y2
            elif(v_x[1] < y1):
                v_x[1] = y1
            ind['x'] = v_x[0]
            ind['y'] = v_x[1]
            ind['z'] = get_fitness([v_x[0], v_x[1]])
        elif ind['level'] == 'b':
            x_p = np.array([ind['x'], ind['y']])
            v_A, v_D = get_iteration_param_2(t_n, total_nums, wolve_a)
            v_x = x_p - (v_A * v_D)
            if(v_x[0] > x2):
                v_x[0] = x2
            elif(v_x[0] < x1):
                v_x[0] = x1
            if(v_x[1] > y2):
                v_x[1] = y2
            elif(v_x[1] < y1):
                v_x[1] = y1
            ind['x'] = v_x[0]
            ind['y'] = v_x[1]
            ind['z'] = get_fitness([v_x[0], v_x[1]])
        elif ind['level'] == 'c':
            x_p = np.array([ind['x'], ind['y']])
            v_A, v_D = get_iteration_param_2(t_n, total_nums, wolve_a)
            v_x = x_p - v_A * v_D
            if(v_x[0] > x2):
                v_x[0] = x2
            elif(v_x[0] < x1):
                v_x[0] = x1
            if(v_x[1] > y2):
                v_x[1] = y2
            elif(v_x[1] < y1):
                v_x[1] = y1
            ind['x'] = v_x[0]
            ind['y'] = v_x[1]
            ind['z'] = get_fitness([v_x[0], v_x[1]])
        elif ind['level'] == 'd':
            del ind

    wolves_group_cg_t = copy.deepcopy(wolves_group_cg)
    count = len(wolves_group_cg_t)
    # print(count)
    wolves_list = [i for i in wolves_group_cg_t]
    for i in wolves_list:
        if wolves_group_cg_t[i]['level'] == 'd':
            del wolves_group_cg_t[i]

    count = len(wolves_group_cg_t)
    # print(count)

    omega_leaders = omega_find_leader(wolves_group)
    omega_leaders_cg = get_omega_next(omega_leaders, wolves_group_cg_t)
#    print('omega_leaders_cg',omega_leaders_cg)
    for i in range(len(omega_leaders_cg)):
        wolves_group_cg_t[i + count + 1] = omega_leaders_cg[i]

    return wolves_group_cg_t


def get_omega_next(omega_leaders, wolves_group_cg):
    # 拿到omega狼的下一步位置
    for i in omega_leaders:
        leader_group = [[x, wolves_group_cg[x]]
                        for x in wolves_group_cg if wolves_group_cg[x]['id'] in i['leader_group']]
        i['x'] = (leader_group[0][1]['x']+leader_group[1][1]['x']+leader_group[2][1]['x'])/3
        i['y'] = (leader_group[0][1]['y']+leader_group[1][1]['y']+leader_group[2][1]['y'])/3
        del i['leader_group']
    return omega_leaders


if __name__ == '__main__':
    # 进入主函数阶段
    # 起始生成种群数33匹狼
    # a等级：1
    # b等级：4
    # c等级：16
    # d等级：29
    l1=[]
    l2=[]
    for test_n in range(test_num):
        fit_list = []
        initial_wolves = initial_group(50)
        start0=timeit.default_timer()
        wolves_sort = sort_wolves(initial_wolves)
        wolves_group_cg = iteration(wolves_sort, 1, 100, wolves_sort[0])
        for x in wolves_group_cg:
            temp = (wolves_group_cg[x]['x'], wolves_group_cg[x]['y'], wolves_group_cg[x]['z'])
            fit_list.append(temp)
        # 查看第一代狼群中的最优值
        temp1 = max(fit_list, key=lambda x: x[2])
        # print('iteration:',0,'\tz:',-fit_list[loc][2])
        # print('iter,x,y,z')
        # print(0, ',', temp1[0], ',', temp1[1], ',', -temp1[2])
        end0=timeit.default_timer()
        l1.append({'test_n':test_n,'iter':0,'x':temp1[0],'y':temp1[1],'z':-temp1[2],'time':end0-start0})
        # 设置一个狼群记忆库，记录迭代拿到的最大值
        wolves_memory = []
        wolves_memory.append(temp1)

        # 进行迭代
        count = 0
        # 设置记数器
        for i in range(100):
            start=timeit.default_timer()
            wolves_sort = sort_wolves(wolves_group_cg)
            wolves_group_cg = iteration(wolves_group_cg, i+2, 100, wolves_group_cg[0])
            fit_list = []
            for x in wolves_group_cg:
                temp2 = (wolves_group_cg[x]['x'], wolves_group_cg[x]['y'], wolves_group_cg[x]['z'])
                fit_list.append(temp2)
            # 如果生成了比记忆库还好的值，植入记忆库中
            if max(fit_list, key=lambda x: x[2])[2] > max(wolves_memory, key=lambda x: x[2])[2]:
                # print('wolves get better')
                wolves_memory.append(max(fit_list, key=lambda x: x[2]))
            # 如果生成了比记忆库最小值还坏的值，记数器加1
            elif max(fit_list, key=lambda x: x[2])[2] <= min(wolves_memory, key=lambda x: x[2])[2]:
                count = count + 1
                # print('wolves get bad')
            temp_val = max(wolves_memory, key=lambda x: x[2])
            end=timeit.default_timer()
            l1.append({'test_n':test_n,'iter':i+1,'x':temp_val[0],'y':temp_val[1],'z':-temp_val[2],'time':end-start})
            # print(i+1, ',', temp_val[0], ',', temp_val[1], ',', -temp_val[2])
        for x in wolves_group_cg:
            l2.append({'test_n':test_n,'x':wolves_group_cg[x]['x'],'y':wolves_group_cg[x]['y'],'z':-wolves_group_cg[x]['z']})
        print('test_n:',test_n)
    pd.DataFrame(l1).to_csv('./data/'+str(fun_index)+'GWO.csv')
    pd.DataFrame(l2).to_csv('./data/'+str(fun_index)+'GWO_final_population.csv')