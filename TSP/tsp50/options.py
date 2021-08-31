import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'cvrp'")
    parser.add_argument('--graph_size', type=int, default=50, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of instances per batch during training') 
    parser.add_argument('--epoch_size', type=int, default=10240, help='Number of instances per epoch during training')  
    parser.add_argument('--val_size', type=int, default=1000,
                         help='Number of instances used for reporting validation performance')  
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    ### training
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay per epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--eval_batch_size', type=int, default=1000,
    help="Batch size to use during (baseline) evaluation")   
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
    help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--lambda', type=float, default=0.8,
                        help='decrease future reward')
    parser.add_argument('--baseline', default='critic',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--steps', type=int, default=10, help='number of steps to swap')
    
    ###### misc
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epo_res_dir', default='epo_res', help='Directory to write output models to')     
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')

    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0,
    help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
    help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    
    opts.epo_res = os.path.join(
        opts.epo_res_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    ) 
    
    return opts
if __name__ == "__main__":
    opts=get_options()
    
