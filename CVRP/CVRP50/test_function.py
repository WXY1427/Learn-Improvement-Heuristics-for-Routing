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

def emdedding1(input, seq_tensor, rec):
        
    loc_with_depot = torch.cat((input['depot'][:, None, :], input['loc']), 1)   ########## batch_size, graph+1, 2
        
    bs, graph_2 = seq_tensor.size()
        
    enc = position_encoding_init(graph_2, 128)      
        
    enc = enc.cuda()
        
    enc_b = enc.expand(bs,graph_2,128)
       
    col_zero = torch.zeros(bs,1).long().cuda()
        
    seq_tensor_index = torch.cat((col_zero, seq_tensor),1)
#     seq_tensor_index = torch.cat((col_zero, rec),1)        
    cor_3_demand = []
        
    pos_enc = []
        
    for i in range(1, graph_2//2+1):
            
        cor = torch.nonzero(seq_tensor_index == i)

#             pre_pre = seq_tensor_index[cor[:,0], cor[:,1]-2]            
            
        pre = seq_tensor_index[cor[:,0], cor[:,1]-1]
            
        mid = seq_tensor_index[cor[:,0], cor[:,1]]
            
        las = seq_tensor_index[cor[:,0], cor[:,1]+1]
            
#             las_las = seq_tensor_index[cor[:,0], cor[:,1]+2]
           
        dem = input['demand'].gather(1, mid[:,None]-1) 
            
        cor_indice = torch.cat((pre[:,None], mid[:,None], las[:,None]),1)
            
        cor_single = loc_with_depot.gather(1, cor_indice[..., None].expand(*cor_indice.size(), loc_with_depot.size(-1)))  
            
        cor_single_fla = cor_single.view(cor_single.size(0),-1)
            
        cor_with_demand = torch.cat((cor_single_fla, dem),1)
            
        cor_3_demand.append(cor_with_demand)
#         cor_pos = torch.nonzero(rec == i)
        single_pos = enc_b.gather(1, cor[:,1][:,None][..., None].expand(bs, 1, 128))  

        pos_enc.append(single_pos.squeeze())
            
    return torch.stack(cor_3_demand, 1), torch.stack(pos_enc, 1)


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



def validate(model, dataset, dic, cl, opts):
    # Validate
    print('Validating...')
    model.eval()
    dataset = DataLoader(dataset, batch_size=opts.eval_batch_size, num_workers=0)
    batch_to_link = []
    return_m = []
    best_solution =[]
    problem = load_problem(opts.problem)

    for batch_id_e, batch_e in enumerate(dataset):

        batch_size, _ = batch_e['demand'].size()
    
        dem = batch_e['demand'].clone()
    
        rec_v = dem.long()
    
        rec_v[dem!=0.]=torch.tensor([i+1 for i in range(opts.graph_size)]*batch_size)
    
        rec_v[dem==0.]=torch.tensor([i+opts.graph_size+1 for i in range(opts.graph_size)]*batch_size)
    
        batch_e['demand'] = batch_e['demand'].gather(1, rec_v.sort()[1]) 
    
        batch_e['loc'] = batch_e['loc'].gather(1, rec_v.sort()[1][...,None].expand(*rec_v.sort()[1].size(),2)) 
    
        rec_v = make_var(rec_v, opts.use_cuda)
        
        batch_e = make_var(batch_e, opts.use_cuda)
        cost_total = torch.zeros(opts.eval_batch_size).cuda()
        
#         rec_sin = torch.cat([i+torch.tensor([1,1+opts.graph_size]) for i in range(opts.graph_size)], 0)
#         rec_v = rec_sin.repeat(opts.eval_batch_size, 1)        
#         rec_v = make_var(rec_v, opts.use_cuda)
        current_length = None
        current_ten = None
        total_return = []
        length_current = []
        exchange = None
        rec_reward=[]
        seg = opts.graph_size *2 //4  
        rec_best = rec_v.clone() 
        s_time = time.time()        
        for T in range(1000):

            if T==0:
                pre_length, _, che_mask = problem.get_costs(batch_e, rec_v, dic, cl)
#                 pre_best = pre_length
            else:
#                 seq_ten = current_ten
                pre_length = current_length
#                 pre_best = now_best  
        
            input_info, pos = emdedding(batch_e, rec_v)
            

            _, current_length, rec_v, exchange, _, che_mask = model(batch_e, rec_v, input_info, pos, exchange, che_mask, dic, cl, action=None, test = False) 
#             pre_best = now_best.clone() 
#             now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
            
            cost = pre_length-current_length
            
#             now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
#             reward = pre_best - now_best             
#             rec_reward.append(reward)
#             rec_best[reward>0]=rec_v[reward>0]
            
            cost_total = cost_total + cost
            
            total_return.append(cost_total)            
            
            length_current.append(current_length)
            
            pre_length = current_length
            
        duration = time.time() - s_time
        print("Test, took {} s".format(time.strftime('%H:%M:%S', time.gmtime(duration))))
        current_m = torch.stack(length_current, 0)
        best_solution.append(current_m.min(0)[0])
        
        return_total = torch.stack(total_return, 0) 
        return_max = return_total.max(0)[0]
        return_m.append(return_max)
        batch_to_link.append(cost_total)
        total_cost = torch.cat(batch_to_link,0)
        return_return = torch.cat(return_m,0)
        best = torch.cat(best_solution,0)

    return total_cost, return_return, best



def validate1(model, dataset, opts):
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
        batch_e = make_var(batch_e, opts.use_cuda)
        cost_total = torch.zeros(opts.eval_batch_size).cuda()
        rec_v = torch.linspace(1, opts.graph_size, steps=opts.graph_size).expand(opts.eval_batch_size, opts.graph_size)
        rec_v = make_var(rec_v, opts.use_cuda)
        current_length = None
        current_ten = None
        total_return = []
        length_current = []
        exchange = None
        em = torch.zeros(opts.eval_batch_size, opts.graph_size, opts.embedding_dim).cuda()
        
        for T in range(opts.steps):

            if T==0:
#                 seq_ten = problem.seq_tensor(batch_e, rec_v)
                pre_length, _, seq_ten = problem.get_costs(batch_e, rec_v)
#                 now_best = pre_length
            else:
                seq_ten = current_ten
                pre_length = current_length
            
            input_info, pos = emdedding(batch_e, seq_ten, rec_v.long())
            
#             pre_length, log_likelihood, current_length, rec_v, current_ten = model(batch_e, rec_v, input_info, test = False, pre_length=current_length) 

            log_likelihood, current_length, rec_v, current_ten, em, exchange = model(batch_e, rec_v, em, input_info, pos, exchange, test = False) 
#             pre_best = now_best.clone() 
#             now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
            
            cost = pre_length-current_length
            
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


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)


    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
#     set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
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
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    total_cost, return_return, best = validate(model, val_dataset, opts)
    print(total_cost)
    print('Improving: {} +- {}'.format(
    total_cost.mean().item(), torch.std(total_cost) / math.sqrt(len(total_cost))))
    
    print(return_return)
    print('Improving: {} +- {}'.format(
    return_return.mean().item(), torch.std(return_return) / math.sqrt(len(return_return))))
    
    print(best)
    print('Improving: {} +- {}'.format(
    best.mean().item(), torch.std(best) / math.sqrt(len(best))))

#     if not opts.no_tensorboard:
#         tb_logger.log_value('val_avg_reward', avg_reward, step)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):

#     x, _ = baseline.unwrap_batch(batch)
    x = make_var(batch, opts.use_cuda)
#     bl_val = make_var(bl_val, opts.use_cuda) if bl_val is not None else None
    
    batch_size, graph_size = batch['demand'].size()
    rec = torch.linspace(1, graph_size, steps=graph_size).expand(batch_size, graph_size)
    rec = make_var(rec, opts.use_cuda)
    current_length = None
    current_ten = None
    em = torch.zeros(batch_size, graph_size, opts.embedding_dim).cuda()
    # Evaluate model, get costs and log probabilities
    I = 1
    Y = 0.99

    for U in range(50):
        reward_r = []
        bl_r = []
        bl_detach_r = []
        log_like_r = []
        ret = []
        ### rollout   
        for T in range(10):  
            
            bl_val, val = baseline.eval(x, rec, ten_val=current_ten)        ####### value

            pre_length, log_likelihood, current_length, rec, current_ten, em = model(x, rec, em, test = False, pre_length=current_length, ten=current_ten) 
            
            if U==0 and T==0:
                pre_best = pre_length.clone()
                now_best = torch.cat((pre_length[None,:], current_length[None,:]),0).min(0)[0]
            else:
                pre_best = now_best.clone()
                now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
        
            reward = pre_best - now_best
            
#             reward = pre_length-current_length

            bl_r.append(val)

            bl_detach_r.append(bl_val)

            reward_r.append(reward)

            log_like_r.append(log_likelihood)
        
#         bl_val, val = baseline.eval(x, rec, ten_val=current_ten)
        reward_r_r = reward_r[::-1]
        next_return = bl_val
        ret.append(next_return)
        for r in range(1,len(reward_r)):
            
            this_return = next_return * Y + reward_r_r[r]
            
            ret.append(this_return)
            
            next_return = this_return
            
        val_tru = torch.stack(ret[::-1], 0)
        
        val_est = torch.stack(bl_r,0)
        
        val_est_det = torch.stack(bl_detach_r,0)
        
        log_like = torch.stack(log_like_r,0)
        
        
        bl_loss = (val_tru - val_est).pow(2).mean()

        reinforce_loss = ((val_tru - val_est_det)*log_like).mean()
        
        loss =  bl_loss - reinforce_loss
        
        optimizer.zero_grad() 
        loss.backward()

        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step() 
