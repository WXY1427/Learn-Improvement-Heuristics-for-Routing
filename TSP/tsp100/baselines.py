import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from scipy.stats import ttest_rel
import copy
from train import get_inner_model ########## no need for rollout now

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass



class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, rec):
        v, input_info, pos_enc = self.critic(x, rec)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach().squeeze(), v.squeeze(), input_info, pos_enc

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)
