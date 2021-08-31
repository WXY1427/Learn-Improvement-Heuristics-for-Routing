from torch.utils.data import Dataset
import torch
import os
import pickle


class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, exchange, rec, pre_length=None, pi_ori=None):  ###### exchange instead of pi: n_heads, batch_size, 2
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param exchange: (n_heads, batch_size, 2) meaning exchange two nodes;
        :return: (batch_size) reward: improving of the length
        """

        exchange_one_head = exchange.detach()   ############ batch_size, 2 

        
        batch_size, graph_size = dataset['demand'].size()

#         CAPACITY = 1.

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)   ########## batch_size, graph+1, 2
        
        ########## length of previous
        if pre_length is None:        
            d_ori = loc_with_depot.gather(1, pi_ori[..., None].expand(*pi_ori.size(), loc_with_depot.size(-1)))
            length_pre =  (d_ori[:, 1:] - d_ori[:, :-1]).norm(p=2, dim=2).sum(1) + (d_ori[:, 0] - dataset['depot']).norm(p=2, dim=1) + (d_ori[:, -1] - dataset['depot']).norm(p=2, dim=1)
        else:
            length_pre = pre_length 

        ############# change the record    
        rec_new = rec.clone()
        pas_rec = rec_new[torch.arange(batch_size), exchange_one_head[:,0]].clone()
        rec_new[torch.arange(batch_size), exchange_one_head[:,0]] = rec_new[torch.arange(batch_size), exchange_one_head[:,1]].clone()
        rec_new[torch.arange(batch_size), exchange_one_head[:,1]] = pas_rec
        
        
   
        pi = CVRP.seq_tensor(dataset, rec_new)             
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))                                 
        length_now = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - dataset['depot']).norm(p=2, dim=1) + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  
        
        
        
        comp = torch.cat((length_pre[None,:],length_now[None,:]),0)
        choice_ind = comp.min(0)[1]

        choice = torch.cat((rec[None,:], rec_new[None,:]),0)

        rec_final = choice.gather(0,choice_ind[:,None][None,:].expand(1, batch_size, graph_size))


        return length_pre, None, length_now, rec_final.squeeze(), pi
#         return length_pre, None, length_now, rec_new, pi    
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)
    
    
    
    @staticmethod
    def seq_tensor(input, rec, capacity=1.):
        bs, gs = input['demand'].size()
        demand = input['demand'].clone()
        demand_node = demand.gather(1, rec.long()-1)
        matrix_z = torch.zeros(bs,gs).cuda()
        matrix_i = torch.ones(bs,3*gs).long().cuda()
        matrix_acc = torch.zeros(bs,3*gs).long().cuda()
        
        while not (demand_node==0.).all():

            for i in range(gs):
                matrix_z[:,i]=demand_node[:,:i+1].sum(1).clone()
            first_over_one = (matrix_z <= 1.).sum(dim=1)
            demand_node[matrix_z<=1.] = 0.
            matrix_i[torch.arange(bs).cuda(), first_over_one]=0 
            
        matrix_i[:,gs]=1    
        matrix_z_i = matrix_i.clone()
        for i in range(3*gs):
            matrix_acc[:,i]=matrix_i[:,:i+1].sum(1).clone()
        
        matrix_acc[matrix_acc>gs]=0
        matrix_acc[matrix_z_i==0]=0
        
        dic = torch.cat((torch.zeros(bs,1).cuda(), rec),1)
        
        
        return dic.gather(1, matrix_acc).long()
    
#     @staticmethod
#     def seq_tensor(input, rec, capacity=1.):
#         CAPACITY = capacity
#         demand = input['demand']
#         batch_size, graph_size = input['demand'].size()
#         seq_with_depot = []            
#         for bat in range(batch_size):
#             cap = 0
#             i = 0
#             seq_single = rec[bat].clone()
#             indice_for_demand = seq_single.long()
#             for ind_add_zero in range(graph_size):
#                 cap += demand[bat][indice_for_demand[ind_add_zero]-1]
#                 if cap > CAPACITY:
#                     where_zero = ind_add_zero + i
#                     seq_single = torch.cat((seq_single[:where_zero], torch.zeros(1).cuda(), seq_single[where_zero:]),0)
#                     i += 1
#                     cap = demand[bat][indice_for_demand[ind_add_zero]-1].clone()
#             line_ori = torch.cat((seq_single, torch.zeros(3*graph_size-len(seq_single)).cuda()),0).long()
#             seq_with_depot.append(line_ori)   
#         seq_tensor = torch.stack(seq_with_depot, 0)  ######### bs*(3*graph_size)
#         return seq_tensor


class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        CAPACITY = 1.
        demands = torch.cat(
            (
                dataset['demand'][:, :1] * 0 - CAPACITY,  # Hacky
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = dataset['demand'][:, 0] * 0  #  torch.zeros(batch_size, 1, out=dataset['demand'].data.new())
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0))[:, None]] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) from each next location from its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000):
        super(VRPDataset, self).__init__()

        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'loc': torch.FloatTensor(loc),
                        'demand': torch.FloatTensor(demand) / capacity,  # Normalize so capacity = 1
                        'depot': torch.FloatTensor(depot),
                    }
                    for depot, loc, demand, capacity in data[:num_samples]
                ]
        else:
            self.data = [
                # {
                #     'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                #     'demand': torch.FloatTensor(size).uniform_(0, 1),
                #     'depot': torch.FloatTensor(2).uniform_(0, 1)
                # }
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }

                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
