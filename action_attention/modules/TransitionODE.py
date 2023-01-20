import torch
from torch import nn
from torchdyn.models import NeuralODE

from action_attention.utils import to_one_hot


class TransitionModel(nn.Module):

    def __init__(self, embedding_dim, num_objects, action_dim, hidden_dim=256, ignore_action=False, copy_action=False):
        super().__init__()
        self.ffc_state = nn.Sequential(
            nn.Linear(embedding_dim * num_objects, hidden_dim),
            nn.SELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.action_dim = action_dim
        self.num_objects = num_objects

        self.ffc_action = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.ffc_out = (nn.Linear(hidden_dim, embedding_dim*num_objects))
    def forward(self, state_tuple):
        state, action, viz = state_tuple
        if len(state.shape) == 2:
            state = state.reshape(-1)
        else:
            state = state.reshape(-1, -1)

        state = self.ffc_state(state)
        if not self.ignore_action:
            action = self.process_action_(action, viz)
            action = self.ffc_action(action)
            state = state * action
        out = self.ffc_out(state)
        out.reshape(state.shape)
        return out


    def process_action_(self, action, viz=False):

        if self.copy_action:
            if len(action.shape) == 1:
                # action is an integer
                action_vec = to_one_hot(action, self.action_dim).repeat(1, self.num_objects)
            else:
                # action is a vector
                action_vec = action.repeat(1, self.num_objects)

            # mix node and batch dimension
            action_vec = action_vec.reshape(-1, self.action_dim).float()
        else:
            # we have a separate action for each node
            if len(action.shape) == 1:
                # index for both object and action
                action_vec = to_one_hot(action, self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)
            else:
                action_vec = action.reshape(action.size(0), self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)

        return action_vec

class TransitionODE(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_objects,
                 action_dim,
                 hidden_dim,
                 ignore_action=False,
                 copy_action=False,
                 timesteps=5):
        super().__init__()
        self.timespan = torch.range(0, timesteps)
        self.net = TransitionModel(embedding_dim,
                                   num_objects,
                                   action_dim,
                                   hidden_dim,
                                   ignore_action,
                                   copy_action)
        self.ode = NeuralODE(self.net, sensitivity='adjoint', solver='dopri5')

    def forward(self, state_tuple):
        _, trajectory = self.ode(state_tuple, self.timespan)
        return trajectory[-1], state_tuple[1], state_tuple[2]

