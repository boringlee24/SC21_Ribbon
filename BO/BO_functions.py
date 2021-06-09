import sys
#from matplotlib.ticker import MultipleLocator
import json
import os
os.chdir('../query')
sys.path.append('../query')
import pandas
import numpy as np
import pdb
from bayes_opt import BayesianOptimization
from scipy.spatial import distance
import distributor as distr

# read in price.csv as a dict
price_file = 'price.csv'
price = pandas.read_csv(price_file)
instance_list = price[price.columns[0]].tolist()
price_list = price[price.columns[1]].tolist()
price_dict = {}
for i in range(len(instance_list)):
    price_dict[instance_list[i]] = price_list[i]

def max_price(model):
    path = f'configs/{model}.json'
    with open(path) as f:
        read = json.load(f)
    ins_types = read['ins_types']
    max_num = read['max_num']
    highest_price = 0
    for ins, num in zip(ins_types, max_num):
        highest_price += price_dict[ins] * num
    return highest_price

def max_instance(model):
    path = f'configs/{model}.json'
    with open(path) as f:
        read = json.load(f)
    ins_types = read['ins_types']
    max_num = read['max_num']
    num1 = int(max_num[0])
    num2 = int(max_num[1])
    num3 = int(max_num[2])
    return num1, num2, num3

def get_ins_type(model):
    path = f'configs/{model}.json'
    with open(path) as f:
        read = json.load(f)
    ins_types = read['ins_types']
    return ins_types[0], ins_types[1], ins_types[2]

violate_dict = {}
obj_range = 2

def total_price(model, num1, num2, num3):
    num1 = int(round(num1))
    num2 = int(round(num2))
    num3 = int(round(num3))
    ins1, ins2, ins3 = get_ins_type(model)
    return price_dict[ins1]*num1 + price_dict[ins2]*num2 + price_dict[ins3]*num3

def eval_qos(model, num1, num2, num3):
    assert type(num1) == int
    assert type(num2) == int
    assert type(num3) == int

    if num1 == 0 and num2 == 0 and num3 == 0:
        return 0, 0
    path = f'configs/{model}.json'
    with open(path) as f:
        read = json.load(f)
    ins_types = read['ins_types']
    types = {ins_types[0]:num1, ins_types[1]:num2, ins_types[2]:num3}
    kwargs = {'num_samp':read['num_samp'], 'inter_arrival':read['inter_arrival'], 'qos':read['qos'], 'rm':read['rm']}
    price, rate1 = distr.run(types, **kwargs)
    price, rate2 = distr.run(types, **kwargs)
    price, rate3 = distr.run(types, **kwargs)
    rate = min(rate1, rate2, rate3)
    return price, rate

def objective_int(model, num1, num2, num3):
    assert type(num1) == int
    assert type(num2) == int
    assert type(num3) == int
    global violate_dict
    highest_price = max_price(model)
    key = f'{num1}, {num2}, {num3}'

    if num1 == 0 and num2 == 0 and num3 == 0:
        violate_dict[key] = True # violation
        return 0 / obj_range
    price, rate = eval_qos(model, num1, num2, num3)    
    if rate < 99: # region 1, from 2 to 1
        violate_dict[key] = True # violation
        val = rate / 99 / obj_range 
        return val
    else: # region 0, from 1 to 0
        violate_dict[key] = False 
        val = (1-price / highest_price)/obj_range + 1/obj_range
        return val

def obj_function_3D(x,y,z):
    num1 = int(round(x))
    num2 = int(round(y))
    num3 = int(round(z))
    return objective_int(model, num1, num2, num3)

def get_violate_dict():
    global violate_dict
    return violate_dict

class BO(BayesianOptimization):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.pruned = []

    def bo_prune(self, violate_dict):
        # update p_upper and p_lower after each update
        x_prev = np.round(self._space._params[-1])
        y_prev = self._space._target[-1]

        x_bound = int(x_prev[0])
        y_bound = int(x_prev[1])
        z_bound = int(x_prev[2])
        key =  f'{x_bound}, {y_bound}, {z_bound}'
        xmax = int(self._space._bounds[0,1] + 1)
        ymax = int(self._space._bounds[1,1] + 1)
        zmax = int(self._space._bounds[2,1] + 1)

        if violate_dict[key] == False:
            ins1, ins2, ins3 = get_ins_type(model)
            standard_price = round(price_dict[ins1]*x_bound + price_dict[ins2]*y_bound + price_dict[ins3]*z_bound,2)
            point = [x_bound, y_bound, z_bound] # don't forget itself
            self._pruned.append(point)
            for x in range(0, xmax):
                for y in range(0, ymax): # y dimension
                    for z in range(0, zmax):
                        point = [x,y,z]
                        if point not in self._pruned:
                            t_price = round(price_dict[ins1]*x + price_dict[ins2]*y + price_dict[ins3]*z,2)
                            if t_price > standard_price:
                                self._pruned.append(point)
        elif violate_dict[key] == True:
            point = [x_bound, y_bound, z_bound] # don't forget itself
            self._pruned.append(point)
            if model == 'dien':
                threshold = 98.6/99/2
            else:
                threshold = 98.85/99/2
            if y_prev > threshold: 
                actual_xbound = x_bound
            else: # violates by a lot
                actual_xbound = x_bound+1
            for x in range(0, actual_xbound):
                for y in range(0, y_bound+1):
                    for z in range(0, z_bound+1):
                        point = [x,y,z]
                        if point not in self._pruned:
                            self._pruned.append(point)

class Random_Prune():
    def __init__(self, remain, seed=1):
        self.remain = remain[:] # remaining search space
        self.best_price = max_price(model)
        self.max_config = max_instance(model)
        np.random.seed(seed)
        self.num_iter = 0
        self.rate_history = []
        self.num_pruned = 0
        self.config_history = []
    def eval_config(self, num1, num2, num3):
        key = f'{num1}, {num2}, {num3}'
        price, rate = eval_qos(model, num1, num2, num3)
        if price < self.best_price and rate >= 99:
            self.best_price = price
        self.remain.remove((num1, num2, num3))
        self.num_iter += 1
        self.rate_history.append(rate)
        self.config_history.append([int(num1),int(num2),int(num3)])
        # prune
        xmax,ymax,zmax = self.max_config
        if rate < 99:
            for x in range(0,num1):
                for y in range(0,num2):
                    for z in range(0,num3):
                        point=(x,y,z)
                        if point in self.remain:
                            self.remain.remove(point)
                            self.num_pruned += 1
        else:
            for x in range(num1+1,xmax+1):
                for y in range(num2+1,ymax+1):
                    for z in range(num3+1,zmax+1):
                        point=(x,y,z)
                        if point in self.remain:
                            self.remain.remove(point)
                            self.num_pruned += 1
        return price, rate
                
    def iterate(self):
        ind = np.random.randint(0, len(self.remain), size=1)[0]
        next_point = self.remain[ind]
        self.eval_config(*next_point)
        return

class RSM_gradient():
    def __init__(self, remain, seed):
        self.remain = remain[:] # remaining search space
        self.best_price = max_price(model)
        self.max_config = max_instance(model)
        np.random.seed(seed)
        self.num_iter = 0
        self.curr_point = (0, 0, 0)
        self.curr_price = 0
        self.curr_rate = 0
        self.starting_order = [] # order of starting points, always start with first one
        self.point_qos = {}
        self.point_rate = {}
        self.rate_history = []
        self.config_history = []
    def gen_ccf(self): # Central Composite Face-centered
        ccf_list = []
        points = [[0,0,0],[0,1,0],[1,0,0],[1,1,0],[0.5,0.5,0],
                [0,.5,.5],[1,.5,.5],[.5,0,.5],[.5,1,.5],[.5,.5,.5],
                [0,0,1],[0,1,1],[1,0,1],[1,1,1],[0.5,0.5,1]]
        points = [np.array(x) for x in points]
        max_ins = np.array(self.max_config)
        for point in points:
            ccf_list.append(np.round(point*max_ins).astype(int))
        return ccf_list
    def eval_config(self, update, num1, num2, num3):
        key = f'{num1}, {num2}, {num3}'
        price, rate = eval_qos(model, num1, num2, num3)
        if update:
            self.curr_price = price
            self.curr_rate = rate
            if price < self.best_price and rate >= 99:
                self.best_price = price
        self.curr_point = (num1,num2,num3)
        # don't count evalated CCF points in iteration number
        if (num1,num2,num3) in self.remain:
            self.remain.remove((num1, num2, num3))
            self.num_iter += 1
            self.rate_history.append(rate)
            self.config_history.append([int(num1), int(num2), int(num3)])
        return price, rate
    def force_update(self, price, rate):
        self.curr_price = price
        self.curr_rate = rate
        if self.curr_price < self.best_price and self.curr_rate >= 99:
            self.best_price = self.curr_price

    def rank_iter(self, point):
        price, rate = self.eval_config(True, *point)
        if rate >= 99:
            self.point_qos[point] = price # the lower the better
        else:
            self.point_rate[point] = -rate # the lower the better

    def rank_ccf(self): # evaluate and rank all points by price then rate
        qos_sort = [k for k,v in sorted(self.point_qos.items(), key=lambda item: item[1])]
        rate_sort = [k for k,v in sorted(self.point_rate.items(), key=lambda item: item[1])]
        self.starting_order = qos_sort + rate_sort
    def get_sp(self):
        if len(self.starting_order) > 0:
            starting_point = self.starting_order[0]
            self.starting_order.pop(0)
            return starting_point
        else:
            ind = np.random.randint(0, len(self.remain), size=1)[0]
            starting_point = self.remain[ind]
            return starting_point
    def initialize(self, starting_point):
        self.eval_config(True, *starting_point)
    def iterate(self): 
        # just return if reached optimal 
        x, y, z = self.curr_point
        next_point = self.curr_point
        if self.curr_rate >= 99:
            # generate next point candidates
            candi = []
            if x > 0 and (x-1,y,z) in self.remain:
                candi.append((x-1,y,z)) 
            if y > 0 and (x,y-1,z) in self.remain:
                candi.append((x,y-1,z)) 
            if z > 0 and (x,y,z-1) in self.remain:
                candi.append((x,y,z-1))
            np.random.shuffle(candi)
            for point in candi:
                price, rate = self.eval_config(False, *point)
                if price <= self.curr_price and rate >= 99:
                    self.force_update(price, rate)
                    return 
        else: # violates QoS
            candi = []
            if x < self.max_config[0] and (x+1,y,z) in self.remain:
                candi.append((x+1,y,z)) 
            if y < self.max_config[1] and (x,y+1,z) in self.remain:
                candi.append((x,y+1,z)) 
            if z < self.max_config[2] and (x,y,z+1) in self.remain:
                candi.append((x,y,z+1))
            np.random.shuffle(candi)
            for point in candi:
                price, rate = self.eval_config(False, *point)
                if rate >= self.curr_rate: # this is rate at descent point
                    # use this as next point
                    self.force_update(price, rate)
                    return 
        # if there is no descent point
        next_point = self.get_sp()
        self.eval_config(True, *next_point)
        return 
         
class GradientDescent():
    def __init__(self, remain, seed):
        self.remain = remain[:] # remaining search space
        self.best_price = max_price(model)
        self.max_config = max_instance(model)
        np.random.seed(seed)
        self.num_iter = 0
        self.curr_point = (0, 0, 0)
        self.curr_price = 0
        self.curr_rate = 0
        self.rate_history = []
        self.config_history = []
    def get_sp(self): # generate a random starting point from remaining points
        ind = np.random.randint(0, len(self.remain), size=1)[0]
        starting_point = self.remain[ind]
        return starting_point
    def eval_config(self, update, num1, num2, num3): # returns price and rate
        key = f'{num1}, {num2}, {num3}'
        price, rate = eval_qos(model, num1, num2, num3)
        if update:
            self.curr_price = price
            self.curr_rate = rate
            if self.curr_price < self.best_price and self.curr_rate >= 99:
                self.best_price = self.curr_price
        self.curr_point = (num1, num2, num3)
        self.remain.remove(self.curr_point)
        self.num_iter += 1
        self.rate_history.append(rate)
        self.config_history.append([num1, num2, num3])

        return price, rate
    def force_update(self, price, rate):
        self.curr_price = price
        self.curr_rate = rate
        if self.curr_price < self.best_price and self.curr_rate >= 99:
            self.best_price = self.curr_price

    def initialize(self, starting_point):
        # first evaluate starting point:
        self.eval_config(True, *starting_point)
    def iterate(self): 
        # returns num_iters used
        # just return if reached optimal 
        x, y, z = self.curr_point
        next_point = self.curr_point
        num_iter = 0
        if self.curr_rate >= 99:
            # generate next point candidates
            candi = []
            if x > 0 and (x-1,y,z) in self.remain:
                candi.append((x-1,y,z)) 
            if y > 0 and (x,y-1,z) in self.remain:
                candi.append((x,y-1,z)) 
            if z > 0 and (x,y,z-1) in self.remain:
                candi.append((x,y,z-1))
            np.random.shuffle(candi)
            for point in candi:
                price, rate = self.eval_config(False, *point)
                num_iter += 1
                if price <= self.curr_price and rate >= 99:
                    self.force_update(price, rate)
                    return num_iter
        else: # violates QoS
            candi = []
            if x < self.max_config[0] and (x+1,y,z) in self.remain:
                candi.append((x+1,y,z)) 
            if y < self.max_config[1] and (x,y+1,z) in self.remain:
                candi.append((x,y+1,z)) 
            if z < self.max_config[2] and (x,y,z+1) in self.remain:
                candi.append((x,y,z+1))
            np.random.shuffle(candi)
            for point in candi:
                price, rate = self.eval_config(False, *point)
                num_iter += 1
                if rate >= self.curr_rate: # this is rate at descent point
                    # use this as next point
                    self.force_update(price, rate)
                    return num_iter
        # if there is no descent point
        next_point = self.get_sp()
        self.eval_config(True, *next_point)
        num_iter += 1
        return num_iter

