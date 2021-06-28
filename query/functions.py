import numpy as np

def random_samp(num_samp): 
    mu, sigma = 5.1, 0.2 # mean and standard deviation
    samples = np.random.lognormal(mu, sigma, num_samp)
    
    for i in range(len(samples)):
        rand_num = np.random.randint(low=1,high=100,size=1)
        if rand_num <= 2: 
            samples[i] = np.random.randint(low=1,high=100,size=1)
        elif rand_num <= 12: 
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

