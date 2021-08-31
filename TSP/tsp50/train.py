import os
import time
from tqdm import tqdm
import torch
import math

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from log_utils import log_values
import torch.nn.functional as F
from insertion import insertion
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

def emded(input, rec):
        
        ########## input: batch_size, graph, 2
    bs, gs = rec.size()
        
    enc = position_encoding_init(gs, 128)      
        
    enc = enc.cuda()
        
    enc_b = enc.expand(bs,gs,128)
        
#         seq_tensor_index = torch.cat((rec.long()[:,-1][:,None], rec.long()), 1)
    seq_tensor_index = rec.long()
    
    node_2_cor = []
        
    pos_enc = []
        
    for i in range(gs):
            
        cor = torch.nonzero(rec.long() == i)
            
        pre = seq_tensor_index[cor[:,0], cor[:,1]]
            
#             mid = seq_tensor_index[cor[:,0], cor[:,1]+1]
            
#             cor_indice = torch.cat((pre[:,None], mid[:,None]),1)
        cor_indice = pre[:,None]    
    
#             cor_single = input.gather(1, cor_indice[..., None].expand(*cor_indice.size(), 2))  
        cor_single = input.gather(1, cor_indice[..., None].expand(*cor_indice.size(), 2))    
    
        cor_single_fla = cor_single.view(cor_single.size(0),-1)
            
        node_2_cor.append(cor_single_fla)
            
        single_pos = enc_b.gather(1, cor[:,1][:,None][..., None].expand(bs, 1, 128))  

        pos_enc.append(single_pos.squeeze())

    return torch.stack(node_2_cor, 1), torch.stack(pos_enc, 1)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def rollout(model, dataset, steps, opts):
    # rollout
    model.eval()
#     dataset = DataLoader(dataset, batch_size=opts.batch_size, num_workers=0)
#     for batch_id_e, batch_e in enumerate(data 

#     batch_e = make_var(batch_e, opts.use_cuda)
    batch_e = dataset
    rec_v = torch.linspace(0, opts.graph_size-1, steps=opts.graph_size).expand(opts.batch_size, opts.graph_size)
    rec_v = make_var(rec_v, opts.use_cuda)
    current_length = None
    exchange = None
    em = torch.zeros(opts.batch_size, opts.graph_size, opts.embedding_dim).cuda()
    for T in range(steps):     
            
        pre_length, log_likelihood, current_length, rec_v, em, exchange, _ = model(batch_e, rec_v, em, exchange, test = False, pre_length=current_length)
            
    return rec_v, em

def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    model.eval()
    dataset = DataLoader(dataset, batch_size=opts.eval_batch_size, num_workers=0)
    batch_to_link = []
    return_m = []
    best_solution =[]
    for batch_id_e, batch_e in enumerate(dataset):

#         rec_v = insertion(batch_e, met='farthest')
#         rec_v = rec_v.cuda()
        batch_e = make_var(batch_e, opts.use_cuda)
        cost_total = torch.zeros(opts.eval_batch_size).cuda()
        rec_v = torch.linspace(0, opts.graph_size-1, steps=opts.graph_size).expand(opts.eval_batch_size, opts.graph_size)
        rec_v = make_var(rec_v, opts.use_cuda)
        current_length = None
        exchange = None
        total_return = []
        length_current = []
        em = torch.zeros(opts.eval_batch_size, opts.graph_size, opts.embedding_dim).cuda()
        s_time = time.time()
        for T in range(1000):
            
            input_info, pos_enc = emded(batch_e, rec_v)
            
            pre_length, log_likelihood, current_length, rec_v, em, exchange, entropy = model(batch_e, rec_v, input_info, pos_enc, em, exchange, test = False, pre_length=current_length)
            
            cost = pre_length-current_length
            
            cost_total = cost_total + cost.detach()
            
            total_return.append(cost_total)
            
            length_current.append(current_length)
            
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
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
#     if not opts.no_tensorboard:
#         tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
#     training_dataset = torch.load('myval.pt')
#     training_dataset = training_dataset[0:1500]
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
    print('Improving: {} +- {}'.format(
    total_cost.mean().item(), torch.std(total_cost) / math.sqrt(len(total_cost))))
    
    print('Best Improving: {} +- {}'.format(
    return_return.mean().item(), torch.std(return_return) / math.sqrt(len(return_return))))
    
    print('Best solutions: {} +- {}'.format(
    best.mean().item(), torch.std(best) / math.sqrt(len(best))))

#     if not opts.no_tensorboard:
#         tb_logger.log_value('val_avg_reward', avg_reward, step)
#    torch.save(best,os.path.join(opts.epo_res, 'epoch-{}.pt'.format(epoch)))    
#    print('Saving done')

def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
):
#     rec = insertion(batch, met='farthest')
#     rec = rec.cuda()
    x = make_var(batch, opts.use_cuda)
    
    batch_size, graph_size, _ = batch.size()
    rec = torch.linspace(0, graph_size-1, steps=graph_size).expand(batch_size, graph_size)
    rec = make_var(rec, opts.use_cuda)
#     rec, em = rollout(model, x, epoch, opts)
#     model.train()
    current_length = None

    em = torch.zeros(batch_size, graph_size, opts.embedding_dim).cuda()
    exchange = None
    I = 1
    Y = 0.99
    for U in range(50):

        reward_r = []
        bl_r = []
        bl_detach_r = []
        log_like_r = []
        ret = []
        entr = []
#         acc = 0
        ### rollout   
        for T in range(4):  
            bl_val, val, input_info, pos_enc = baseline.eval(x, rec)        ####### value

            pre_length, log_likelihood, current_length, rec, em, exchange, entropy = model(x, rec, input_info, pos_enc, em, exchange, test = False, pre_length=current_length)   
            
            if U==0 and T==0:
                pre_best = pre_length.clone()
                now_best = torch.cat((pre_length[None,:], current_length[None,:]),0).min(0)[0]
            else:
                pre_best = now_best.clone()
                now_best = torch.cat((pre_best[None,:], current_length[None,:]),0).min(0)[0]
        
            reward = pre_best - now_best

            
#             reward = pre_best-current_length
#             reward[reward<0] = -0.1           
#             reward = pre_length-current_length
#             reward[reward<=0] = -0.001
#             pre_length=current_length.clone()
            
            
#             rm = pre_length-current_length
# #             rm[rm<0] = 0
# #             reward = ((pre_best/now_best)**5)*(rm)
#             reward = rm
    
    
#             reward = ((pre_best/now_best)*(1+1/pre_best)**2)*(rm)
            
#             print(reward)

            bl_r.append(val)

            bl_detach_r.append(bl_val)

            reward_r.append(reward)
            
#             acc = acc+reward

            log_like_r.append(log_likelihood)
    
#             entr.append(entropy)
            
#             del bl_val, log_likelihood
            
#             torch.cuda.empty_cache()
      
        reward_r_r = reward_r[::-1]
#         next_return = bl_detach_r[-1]
        next_return, _, _, _  = baseline.eval(x, rec)
#         next_return = bl_val
#         ret.append(next_return)  

        for r in range(len(reward_r)):                  
            this_return = next_return * Y + reward_r_r[r]                   
            ret.append(this_return)                
            next_return = this_return            
            
        val_tru = torch.stack(ret[::-1], 0)
        val_est = torch.stack(bl_r,0)
        val_est_det = torch.stack(bl_detach_r,0)
        
        log_like = torch.stack(log_like_r,0)        
        
        bl_loss = (val_tru - val_est).pow(2).mean()

        reinforce_loss = ((val_tru - val_est_det)*log_like).mean()
#         entropy_loss = torch.stack(entr,0).mean()

        loss =  bl_loss - reinforce_loss #+ 0.005*entropy_loss
        
        optimizer.zero_grad() 
        loss.backward()
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step() 