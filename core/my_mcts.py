from torch.cuda.amp import autocast as autocast
import torch
import numpy as np

class MCTSES:
    def __init__(self,config,model,hidden_state_roots,reward_hidden_roots):
        self.device=config.device
        self.env_nums=config.p_mcts_nums
        self.action_space_size=config.action_space_size
        self.num_simulations=config.num_simulations
        self.model=model
        self.mctses=[MCTS(self.action_space_size,self.num_simulations,hidden_state_roots[i],reward_hidden_roots[i])for i in range(self.env_nums)]
    def run(self):
        with torch.no_grad():
            self.model.eval()
            for sim in range(self.num_simulations):
                l=[mcts.run() for mcts in self.mctses]

                hidden_states = torch.from_numpy(np.asarray([l[i][0]for i in len(l)])).to(self.device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray([l[i][1]for i in len(l)])).to(self.device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray([l[i][2]for i in len(l)])).to(self.device).unsqueeze(0)
                last_actions = torch.from_numpy(np.asarray([l[i][3]for i in len(l)])).to(self.device).unsqueeze(1).long()

                # evaluation for leaf nodes
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = self.model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)
                else:
                    network_output = self.model.recurrent_inference(hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions)
                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_nodes = network_output.reward_hidden

                for i,mcts in enumerate(self.mctses):
                    mcts.expand()
    def get_roots_values(self):
        return [self.mctses[i].get_root_value() for i in range(self.env_nums)]
    def get_roots_distributions(self):
        return [self.mctses[i].get_root_distribution() for i in range(self.env_nums)]


class MCTS:
    def __init__(self):
        self.nodes=[Node(-1)]
    def run(self):
        pass
    def select_action
    def expand(self):
        pass
    def get_root_value(self):
        return self.nodes[0].value
    def get_root_distribution(self):
        return self.nodes[0].distribution

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac