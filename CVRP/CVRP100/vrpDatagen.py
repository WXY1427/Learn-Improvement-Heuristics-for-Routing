# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import numpy as np
import argparse
import json


def sample_pos():
	return np.random.rand(), np.random.rand()


def gen(num_samples, num_customers, capacity):
	np.random.seed(None)
	samples = []
	for _ in range(num_samples):
		cur_sample = {}
		cur_sample['customers'] = []
		cur_sample['capacity'] = capacity
		dx, dy = sample_pos()
		cur_sample['depot'] = (dx, dy)
		for i in range(num_customers):
			cx, cy = sample_pos()
			demand = np.random.randint(1, 10)
			cur_sample['customers'].append({'position': (cx, cy), 'demand': demand})
		samples.append(cur_sample)

	data_size = len(samples)
	print(data_size)
# 	fout_res = open(args.res_file, 'w')
	fout_res = open(os.path.join('data', 'vrp', "vrp_20_30.json"), 'w')
	json.dump(samples, fout_res)
