import sys
import pdb
import subprocess
import json
import os
import argparse
import time
import datetime
from pathlib import Path

RM_dict =  {'mtwnd': ['multi_task_wnd', 'mtwnd'],
            'dien': ['din', 'dien']}

parser = argparse.ArgumentParser(description='Recommendation')
parser.add_argument('--testcase', metavar='TC', type=str)
args = parser.parse_args()

instance = args.testcase 
if 'g4dn' in instance:
    accel = '--use_accel'
else:
    accel = '' 
Path(f'logs/{instance}').mkdir(parents=True, exist_ok=True)

for rm, cfg in RM_dict.items():
    cwd = '.'
    cmd = f'taskset -a -c 0 python {cfg[0]}.py --instance={instance} --inference_only --num_batches 10 --caffe2_net_type async_dag --config_file "../configs/{cfg[1]}.json" --enable_profiling --engine "prof_dag" --nepochs 10 --max_mini_batch_size 1 {accel}'
    print(cmd)
    subprocess.check_call([cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd = cwd)
    time.sleep(5)

