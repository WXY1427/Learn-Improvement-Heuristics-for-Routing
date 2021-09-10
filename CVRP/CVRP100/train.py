import os
import time
from tqdm import tqdm
import torch
import math

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import DataParallel

# from attention_model import set_decode_type
from log_utils import log_values
import torch.nn.functional as F
from utils import load_problem
from problems import TSP, CVRP, SDVRP
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def emdedding(input, seq_tensor):
        
    loc_with_depot = input['loc'] ########## batch_size, graph+1, 2
        
    bs, graph_2 = seq_tensor.size()
        
    enc = position_encoding_init(graph_2, 128)      
        
    enc = enc.cuda()
        
    enc_b = enc.expand(bs,graph_2,128)

    seq_tensor_index = torch.cat((seq_tensor[:,-1][:,None], seq_tensor, seq_tensor[:,0][:,None]),1)
        
    cor_3_demand = []
        
    pos_enc = []
        
    for i in range(1, graph_2+1):
            
        cor = torch.nonzero(seq_tensor == i)        
            
        pre = seq_tensor_index[cor[:,0], cor[:,1]]
            
        mid = seq_tensor_index[cor[:,0], cor[:,1]+1]
            
        las = seq_tensor_index[cor[:,0], cor[:,1]+2]            
           
        dem = input['demand'].gather(1, mid[:,None]-1) 
            
        cor_indice = torch.cat((pre[:,None]-1, mid[:,None]-1, las[:,None]-1),1)
            
        cor_single = loc_with_depot.gather(1, cor_indice[..., None].expand(*cor_indice.size(), loc_with_depot.size(-1)))  
            
        cor_single_fla = cor_single.view(cor_single.size(0),-1)
            
        cor_with_demand = torch.cat((cor_single_fla, dem),1)
            
        cor_3_demand.append(cor_with_demand)

        single_pos = enc_b.gather(1, cor[:,1][:,None][..., None].expand(bs, 1, 128))  

        pos_enc.append(single_pos.squeeze())
            
    return torch.stack(cor_3_demand, 1), torch.stack(pos_enc, 1)

def validate(baseline, model, dataset, dic, cl, opts):
    # Validate
    print('Validating...')
    model.eval()
    dataset = DataLoader(dataset, batch_size=opts.eval_batch_size, num_workers=0)
    batch_to_link = []
    return_m = []
    best_solution =[]
    problem = load_problem(opts.problem)
    s_time = time.time()
    for batch_id_e, batch_e in enumerate(dataset):

        batch_size, _ = batch_e['demand'].size()
    
        dem = batch_e['demand'].clone()
    
        rec_v = dem.long()
    
        rec_v[dem!=0.]=torch.tensor([i+1 for i in range(opts.graph_size)]*batch_size)
    
        rec_v[dem==0.]=torch.tensor([i+opts.graph_size+1 for i in range(28)]*batch_size)
    
        batch_e['demand'] = batch_e['demand'].gather(1, rec_v.sort()[1]) 
    
        batch_e['loc'] = batch_e['loc'].gather(1, rec_v.sort()[1][...,None].expand(*rec_v.sort()[1].size(),2)) 
    
        rec_v = make_var(rec_v, opts.use_cuda)
        
        batch_e = make_var(batch_e, opts.use_cuda)
        cost_total = torch.zeros(opts.eval_batch_size).cuda()
        
        current_length = None
        current_ten = None
        total_return = []
        length_current = []
        exchange = None
        rec_reward=[]
    
        for T in range(1000):

            if T==0:
                pre_length, _, che_mask = problem.get_costs(batch_e, rec_v, dic, cl)
                pre_best = pre_length
            else:
                pre_length = current_length
                pre_best = now_best  
        
            input_info, pos = emdedding(batch_e, rec_v) 

            _, current_length, rec_v, exchange, _, che_mask = model(batch_e, rec_v, input_info, pos, exchange, che_mask, dic, cl, action=None, test = False) 

            
            cost = pre_length-current_length
            
            now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
            reward = pre_best - now_best 
            
            rec_reward.append(reward)

            cost_total = cost_total + cost
            
            total_return.append(cost_total)            
            
            length_current.append(current_length)
            
            pre_length = current_length
             
            
        current_m = torch.stack(length_current, 0)
        best_solution.append(current_m.min(0)[0])
        
        return_total = torch.stack(total_return, 0) 
        return_max = return_total.max(0)[0]
        return_m.append(return_max)
        batch_to_link.append(cost_total)
        total_cost = torch.cat(batch_to_link,0)
        return_return = torch.cat(return_m,0)
        best = torch.cat(best_solution,0)
        duration = time.time() - s_time
        print("Test, took {} s".format(time.strftime('%H:%M:%S', time.gmtime(duration))))
    return total_cost, return_return, best


def make_var(val, cuda=False, **kwargs):
    if isinstance(val, dict):
        return {k: make_var(v, cuda, **kwargs) for k, v in val.items()}
    var = Variable(val, **kwargs)
    if cuda:
        var = var.cuda()
    return var


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, policy_old, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, dic, cl, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)

    start_time = time.time()
    lr_scheduler.step(epoch)

    # Generate new training data for each epoch

    training_dataset = baseline.wrap_dataset(problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            policy_old,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            dic,
            cl,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
                'policy_old': policy_old.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    total_cost, return_return, best = validate(baseline, model, val_dataset, dic, cl, opts)

    print('Improving: {} +- {}'.format(
    total_cost.mean().item(), torch.std(total_cost) / math.sqrt(len(total_cost))))

    print('Improving: {} +- {}'.format(
    return_return.mean().item(), torch.std(return_return) / math.sqrt(len(return_return))))

    print('Improving: {} +- {}'.format(
    best.mean().item(), torch.std(best) / math.sqrt(len(best))))

def train_batch(
        model,
        policy_old,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        dic,
        cl,
        opts
):

    batch_size, _ = batch['demand'].size()
    
    dem = batch['demand'].clone()
    
    rec = dem.long()
    
    rec[dem!=0.]=torch.tensor([i+1 for i in range(opts.graph_size)]*batch_size)
    
    rec[dem==0.]=torch.tensor([i+opts.graph_size+1 for i in range(28)]*batch_size)
    
    batch['demand'] = batch['demand'].gather(1, rec.sort()[1]) 
    
    batch['loc'] = batch['loc'].gather(1, rec.sort()[1][...,None].expand(*rec.sort()[1].size(),2)) 
    
    rec = make_var(rec, opts.use_cuda)
    
    x = make_var(batch, opts.use_cuda)
    
    current_length = None
    exchange = None

    # Evaluate model, get costs and log probabilities
    I = 1
    Y = 0.999

    for U in range(48):##40
        reward_r = []
        log_like_r = []
        ret = []
        rec_rec = []
        rec_ii = []
        rec_act = []
        rec_pos = []
        rec_mask = []
        ### rollout   
        for T in range(10):  ### 12
            
            if U==0 and T==0:

                now_best, _, che_mask = CVRP.get_costs(x, rec, dic, cl)            
            input_info, pos_enc = baseline.eval(x, rec, i=None, p=None)        ####### value

            rec_mask.append(che_mask.view(batch_size,-1))    
           
        
            log_likelihood, current_length, rec, exchange, act, che_mask = policy_old(x, rec, input_info, pos_enc, exchange, che_mask, dic, cl, action=None, test = False)

            
            pre_best = now_best.clone()
            now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
            reward = pre_best - now_best
    
            rec_act.append(act)
            rec_pos.append(pos_enc)
            rec_rec.append(rec)
    
            rec_ii.append(input_info)

            reward_r.append(reward)

            log_like_r.append(log_likelihood.detach())
                       
            
        reward_r_r = reward_r[::-1]
        input_info, pos_enc = baseline.eval(x, rec, i=None, p=None)
        next_return, _  = baseline.eval(x, rec, i=input_info, p=pos_enc)   
    
        for r in range(len(reward_r)):
            
            this_return = next_return * Y + reward_r_r[r]
            
            ret.append(this_return)
            
            next_return = this_return
            
        rol_act = torch.cat(rec_act)
        
        rol_rec = torch.cat(rec_rec)
        
        rol_input = torch.cat(rec_ii)
        
        rol_pos = torch.cat(rec_pos)
        
        rol_mask = torch.cat(rec_mask)       
            
        val_tru = torch.cat(ret[::-1])
        
        log_like = torch.cat(log_like_r).detach() 
        
        sampler = BatchSampler(
            SubsetRandomSampler(range(len(rol_act))),
            960,
            drop_last=True)
        exchange=None
        for k in sampler:                        
            
            bl_val, val = baseline.eval(x, rec, i=rol_input[k], p=rol_pos[k])        ####### value

            ll = model(x, rec, rol_input[k], rol_pos[k], exchange, rol_mask[k].flatten(), dic, cl, action=rol_act[k], test = False)  

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(ll - log_like[k])

            # Finding Surrogate Loss:
            advantages = val_tru[k] - bl_val
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-0.2, 1+0.2) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5*(val-val_tru[k]).pow(2).mean() 
            
            ###optimize
        
            optimizer.zero_grad() 
            loss.backward()

            grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
            optimizer.step() 
        policy_old.load_state_dict(model.state_dict())