from torch.utils.data import Dataset
import torch
import os
import pickle


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, exchange, rec, pre_length=None):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param rec: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(rec.size(1), out=rec.data.new()).view(1, -1).expand_as(rec) ==
            rec.data.sort(1)[0]
        ).all(), "Invalid tour"
        
        batch_size, graph_size, s = dataset.size()
        
        #calculate the previous length
        if pre_length is None:   
            # Gather dataset in order of tour
            d = dataset.gather(1, rec.long().unsqueeze(-1).expand_as(dataset))
            length_pre =  (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)
        else:
            length_pre = pre_length 
            
        ############# change the record    
#         rec_new = rec.clone()
#         pas_rec = rec_new[torch.arange(batch_size), exchange[:,0]].clone()
#         rec_new[torch.arange(batch_size), exchange[:,0]] = rec_new[torch.arange(batch_size), exchange[:,1]].clone()
#         rec_new[torch.arange(batch_size), exchange[:,1]] = pas_rec
        
        rec_new = rec.clone().cpu()
        exchange_sort = exchange.sort(1)[0].cpu()

        for i in range(batch_size):
            inter = torch.narrow(rec_new[i], 0, exchange_sort[i][0], exchange_sort[i][1]-exchange_sort[i][0]+1).clone()
            rec_new[i][exchange_sort[i][0]:exchange_sort[i][1]+1] = torch.flip(inter,[0])
        rec_new= rec_new.cuda()


        dn = dataset.gather(1, rec_new.long().unsqueeze(-1).expand_as(dataset))
        length_now = (dn[:, 1:] - dn[:, :-1]).norm(p=2, dim=2).sum(1) + (dn[:, 0] - dn[:, -1]).norm(p=2, dim=1)
        
#         comp = torch.cat((length_now[None,:], length_pre[None,:]),0)

#         choice_ind = comp.min(0)[1]

#         choice = torch.cat((rec_new[None,:], rec[None,:]),0)

#         rec_final = choice.gather(0,choice_ind[:,None][None,:].expand(1, batch_size, graph_size))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return length_pre, length_now, rec_new #rec_final.squeeze()

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)   



class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in data[:num_samples]]
        else:
            # Sample points randomly in [0, 1] square
             self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
#            with open('TSPlib/data/eil101.pkl', 'rb') as f:    
#               data = pickle.load(f)

#            self.data = [torch.FloatTensor(data).squeeze() for i in range(num_samples)]   
            
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
