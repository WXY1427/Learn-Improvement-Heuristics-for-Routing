
# coding: utf-8

import torch
import time
from torch.utils.data import DataLoader
import json
# import util
# import operator
# from audioop import reverse
# import new
# from ortools.constraint_solver._pywrapcp import new_BaseLns
# from itertools import repeat
from itertools import combinations
import math


def capacityValid(existing,new):
    totalCap = batch['demand'].clone()[0][new]
    for c in existing:
        totalCap+=batch['demand'].clone()[0][c]

    return totalCap <= 1.


#compute Savings for depot and i,j where i <> j
def computeSaving(depot, i,j):
    iDepot = util.euclideanDistance(i, depot)
    jDepot = util.euclideanDistance(depot, j)
    ijDist = util.euclideanDistance(i, j)
    
    return (iDepot + jDepot - ijDist)
def inPrevious(new,existing):
    start = existing[0]
    end = existing[len(existing)-1]
    if new == start:
        return 1
    elif new == end:
        return 0
    else:
        return -1

    
def depot(saving, saving_lis, dem, parts):
    gs = len(dem)
    adding_number = len(saving_lis) 
#     for i in range(adding_number-2):
#         for j in range(i+1,adding_number-1):
    for i in list(combinations([c[0] for c in saving_lis], parts)):
        ads = []
#         adding = 0
        for j in list(i):
            ads.append(j[0])
#             adding+=saving[j]
        ads = sorted(ads)
        lis = []
        for l in range(parts-1):
            lis.append(dem[ads[l]+1:ads[l+1]+1].sum())
        lis.append(dem[:ads[0]+1].sum())
        lis.append(dem[ads[-1]+1:].sum())
            
        if all(e <= 1. for e in lis):

            return ads+(gs,)*(gs-len(ads))
    return depot(saving, saving_lis, dem, parts+1)

def addings(batch, rec): ######## given opts and part number
    bs, gs = batch['demand'].size()
    parts = torch.ceil(batch['demand'].sum(1)).long().tolist()
    dem_now = batch['demand'].gather(1,rec.long()-1)

    loc_now = batch['loc'].gather(1,rec[...,None].expand_as(batch['loc']))
    lis = []
    for d in range(bs):
        saving = dict()
        for i in range(gs-1):
            j = i+1
            saving[(i, j)] = computeSaving(batch['depot'][d], loc_now[d][i], loc_now[d][j])

        saving_lis = sorted(saving.items(), key=lambda kv: kv[1])

        depot_loc = depot(saving, saving_lis, dem_now[d], parts[d]-1)
        lis.append(depot_loc)
    return torch.tensor(lis)

# parts = math.ceil(batch['demand'].sum())
# parts







