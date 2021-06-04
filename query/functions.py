import numpy as np

def random_samp(num_samp): 
#    mu, sigma = 4.3, 0.5 # mean and standard deviation
    mu, sigma = 5.1, 0.2 # mean and standard deviation
    samples = np.random.lognormal(mu, sigma, num_samp)
    #pdb.set_trace()
    
    for i in range(len(samples)):
        rand_num = np.random.randint(low=1,high=100,size=1)
        if rand_num <= 2: # 2% chance that the size falls uniformly between 0-100
            samples[i] = np.random.randint(low=1,high=100,size=1)
        elif rand_num <= 12: # 8% chance that the size falls uniformly between 300-500
            samples[i] = np.random.randint(low=250,high=500,size=1)
        elif samples[i] > 500:
            samples[i] = np.random.randint(low=400,high=500,size=1)
    return samples

# create instance class
class Instance:
    def __init__(self, index, **kwargs): # curr_time, ins_type and price_dict
        self.index = index # ith instance
        self.ins_type = kwargs['ins_type'][index]
        self.price = kwargs['price_dict'][self.ins_type]
        self.curr_query = None
        self.available = True
        self.avail_time = kwargs['curr_time']
    def start(self, query, latency, curr_time):
        self.curr_query = query
        self.available = False
        self.avail_time = latency + curr_time
    def update(self, curr_time): # run this after each clock tick
        if self.available:
            self.avail_time = curr_time
        else:
            if self.avail_time <= curr_time:
                self.available = True
                self.curr_query = None
                self.avail_time = curr_time

# create query class
class Query:
    def __init__(self, query, curr_time, **kwargs): # curr_time, ins_type and price_dict
        self.query = query # ith instance
        self.batch = kwargs['batch']
        #self.lat_c5a = kwargs['lat_c5a']
        #self.lat_r4 = kwargs['lat_r4']
        #self.lat_m5 = kwargs['lat_m5']
        self.lats = kwargs['lats']
        self.lats_est = kwargs['lats_est']
        self.t_arrive = curr_time
        self.qos = kwargs['qos']
        self.t_qos = self.qos # time left till QoS
        self.wait = 0
    def update(self, curr_time): # run this to update time till qos, do this before starting query 
        t_passed = curr_time - self.t_arrive
        self.wait = t_passed
        if t_passed >= self.qos: # jobs are all the same once violated
            self.t_qos = 0
        else:
            self.t_qos = self.qos - t_passed

class Future_query:
    def __init__(self, query, lats, arr):
        self.query = query
        self.lats = lats
        self.arr = arr

class Oracle_query:
    def __init__(self, query, lats, qos):
        self.query = query
        self.lats = lats
        self.qos = qos

# create instance class
class Ins_Bipartite:
    def __init__(self, index, **kwargs): # curr_time, ins_type and price_dict
        self.index = index # ith instance
        self.ins_type = kwargs['ins_type'][index]
        self.price = kwargs['price_dict'][self.ins_type]
        self.curr_query = None
        self.available = True
        self.avail_time = kwargs['curr_time']
        self.t_left = 0
        self.skew = 1
    def start(self, query, latency, curr_time):
        self.curr_query = query
        self.available = False
        self.avail_time = latency + curr_time
        self.t_left = latency
    def update(self, curr_time): # run this after each clock tick
        if self.available: # there is no query
            self.avail_time = curr_time
        else: # there is a query running
            self.t_left = self.avail_time - curr_time
            if self.avail_time <= curr_time: # when the query is done
                self.available = True
                self.curr_query = None
                self.avail_time = curr_time
                self.t_left = 0

# create instance class
class Ins_Bipartite_V2:
    def __init__(self, index, **kwargs): # curr_time, ins_type and price_dict
        self.index = index # ith instance
        self.ins_type = kwargs['ins_type'][index]
        self.price = kwargs['price_dict'][self.ins_type]
        self.curr_query = None
        self.available = True
        self.avail_time = kwargs['curr_time'] # time when next query in buffer can start
        self.t_left = 0 # time left to finish current query and everything in buffer
        self.buf = []
    def add(self, query_obj, latency): # add query object instances to buffer 
        self.buf.append(query_obj)
        self.t_left += latency
    ###################################################
    # process: add to buffer -> check start (for all) -> increase time -> update (for all)
    ##################################################
    def check_start(self, curr_time): # check if current query is done and if so start job in buffer
        if self.available == True and len(self.buf) > 0:
            q_serve = self.buf.pop(0)
            self.curr_query = q_serve.query
            self.available = False
            latency = q_serve.lats[self.ins_type]
            self.avail_time = latency + curr_time
    ##########################################
    # how to update t_left: t_left = t_left - time_skipped (only when ins is unavailable)
    #########################################
    def update(self, curr_time, t_skip): # run this after each clock tick
        if self.available: # there is no query
            self.avail_time = curr_time
            if self.t_left > 0:
                print('error: there should not be a query in buffer')
                sys.exit()
            elif self.t_left < 0:
                print('error: skipped too much time')
                sys.exit()
        else: # there is a query running
            self.t_left -= t_skip
            if self.avail_time <= curr_time: # when the query is done
                self.available = True
                self.curr_query = None
                self.avail_time = curr_time

