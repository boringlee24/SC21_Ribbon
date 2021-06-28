import pandas
import pdb
from datetime import datetime
import matplotlib
import numpy as np
import matplotlib
import glob
import sys
import json
import os
import math
import random
import time
import functions
from functions import Instance
import argparse

def run(types, num_samp=20000, inter_arrival=5, p=False, qos=100, rm='model'):

    # read in price.csv as a dict
    price_file = 'price.csv'
    price = pandas.read_csv(price_file)
    instance_list = price[price.columns[0]].tolist()
    price_list = price[price.columns[1]].tolist()
    price_dict = {}
    for i in range(len(instance_list)):
        price_dict[instance_list[i]] = price_list[i]
    
    queue_time = [] # this is the arrival time of each query
    arrival_time = 0 
    for i in range(num_samp):
        arrival_time += np.random.poisson(inter_arrival) # 10 ms inter-arrival time
        queue_time.append(arrival_time)
    query = 0 # pointer to next query to arrive
    
    samples = functions.random_samp(num_samp) 
    
    time_zero = 0
    curr_time = 0
    
    ins_type = []
    for k,v in types.items():
        ins_type += [k]*v
    num_ins = len(ins_type)
     
    lat_dict = {}
    for ins in types: 
        lat_dict[ins] = []
        for batch in range(10, 501, 10):
            path = f'../characterization/logs/{ins}/{rm}_{batch}_1.json'
            with open(path, 'r') as f:
                lat_list = json.load(f)
            lat_dict[ins].append((lat_list))

    violation = [] # 0: no violation. 1: violation because latency 2: violation because of wait 3: violation because of both wait and latency
    
    ####### now start the simulation, simulate by 1 ms step ###############
    
    pending_q = [] # queue for queries that arrive
    pending_w = [] # queue wait time for queries in pending_q
    
    ins_list = [Instance(i, ins_type=ins_type, price_dict=price_dict, curr_time=curr_time) for i in range(num_ins)]
    
    # record number of queries that cannot start immediately
    # record number of queries that violated QoS because instance not fast enough, or because of waiting for available instance
    
    time_start = time.time()
   
    while True:
        # FIFO queue, always makes sure queries get served by coming order
        while True: # when query arrive at same time, add all to queue
            if query < num_samp:
                if queue_time[query] == curr_time:
                    pending_q.append(query)
                    pending_w.append(0)
                    query += 1
                else:
                    break
            else:
                break
    
        # start the first query in pending queue
        avail_ins = [k for k in ins_list if k.available == True]
        
        for ins in avail_ins:
            if len(pending_q) > 0:
                q_serve = pending_q.pop(0)
                q_wait = pending_w.pop(0)
                
                # need to calculate the latency of the query
                batch = round(samples[q_serve])
                batch_ind = math.ceil(batch / 10) - 1 # index to refer for latency
                lat = round(random.choice(lat_dict[ins.ins_type][batch_ind]))
    
                ins.start(q_serve, lat, curr_time)

                if lat + q_wait <= qos:
                    violation.append(0)
                elif lat > qos: # even if there is no wait it still cannot make it
                    violation.append(1)
                elif q_wait > qos: # even if latency is 0 it still cannot make it
                    violation.append(2)
                else: # both lat and wait are within qos, but sum of them is not
                    violation.append(3)
            else:
                break
        
    
        # skip ahead when all queries are received and all instances are unavailable
    
        avail_ins = [k for k in ins_list if k.available == True]
       
        if query == num_samp and len(avail_ins) == 0:
            next_time = min([k.avail_time for k in ins_list])
            skip_time = next_time - curr_time
            curr_time += skip_time
            [k.update(curr_time) for k in ins_list]
            pending_w = [k+skip_time for k in pending_w[:]]
        else: # don't skip ahead
            curr_time += 1
            [k.update(curr_time) for k in ins_list]
            pending_w = [k+1 for k in pending_w[:]]
    
        # condition to stop simulation: no more queries, pending_q is empty
        if query == num_samp and len(pending_q) == 0:
            break
    
    total_price = sum([(k.price) for k in ins_list])
    vio_array = np.array(violation)
    non_vio = (vio_array == 0).sum() / len(vio_array) * 100
    lat_vio = (vio_array == 1).sum() / len(vio_array) * 100
    wait_vio = (vio_array == 2).sum() / len(vio_array) * 100
    sum_vio = (vio_array == 3).sum() / len(vio_array) * 100
    
    Tstart = queue_time[0]
    throughput = 1000 * num_samp / (curr_time - Tstart) # qps
   
    if p:
        print(f'number of instances: {num_ins}')
        print(f'instance type: {types}')
        print(f'total price: ${total_price}')
        print(f'QoS satisfaction rate: {non_vio}%')
   
    output = {}
    output['total_price'] = round(total_price,2)
    output['non_vio'] = round(non_vio,2)
    output['lat_vio'] = round(lat_vio,2)
    output['wait_vio'] = round(wait_vio,2)
    output['sum_vio'] = round(sum_vio,2)
    output['throughput'] = round(throughput,2)
    
    return output['total_price'], output['non_vio']

def main():
    parser = argparse.ArgumentParser(description='put in instance configuration to evaluate QoS rate.')
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--type1', type=int, help='number of type1 instances')
    parser.add_argument('--type2', type=int, help='number of type2 instances')
    parser.add_argument('--type3', type=int, help='number of type3 instances')
    args = parser.parse_args()
    with open(f'configs/{args.model}.json') as f:
        config = json.load(f)
    types = config['ins_types']
    ins_pool = {types[0]: args.type1, types[1]: args.type2, types[2]: args.type3} 
    run(ins_pool, num_samp=20000, inter_arrival=config['inter_arrival'], p=True, qos=config['qos'], rm=args.model) 
   
if __name__ == '__main__':
    main()
