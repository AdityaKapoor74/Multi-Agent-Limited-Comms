import torch
import torch.nn.functional as F
from torch import nn
import torch.quantization as quantization

import sys 
import numpy as np
sys.path.append("..") 

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.device = "cpu"

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.hid_size, 1)
        
        if args.use_fake_quantization:
            # Configure quantization parameters
            self.quant_bits = args.quant_bits
            self.quant_min = 0
            self.quant_max = 2**args.quant_bits - 1

    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone() # clone gives the full tensor and avoid the error

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state
    

    def apply_fake_quantization(self, tensor, mask):
        """
        Manual implementation of fake quantization with proper gradient flow
        """
        # Create a copy of the input tensor to avoid modifying the original
        original = tensor.clone()
        
        # Expand mask to match tensor dimensions
        if mask.shape != tensor.shape:
            mask_expanded = mask.expand_as(tensor)
        else:
            mask_expanded = mask
            
        binary_mask = mask_expanded.bool()
        
        # Get min/max values safely
        if torch.any(binary_mask):
            # Detach for min/max calculation to avoid affecting gradient computation
            masked_values = tensor[binary_mask].detach()
            min_val = masked_values.min()
            max_val = masked_values.max()
        else:
            min_val = torch.tensor(0.0, device=tensor.device)
            max_val = torch.tensor(1.0, device=tensor.device)
        
        if min_val == max_val:
            min_val = min_val - 0.01
            max_val = max_val + 0.01
        
        # Calculate scale and zero_point from detached values
        scale = (max_val - min_val) / (self.quant_max - self.quant_min)
        zero_point = torch.round(self.quant_min - min_val / scale)
        
        # Apply quantization formula without modifying the original
        scaled = tensor / scale + zero_point
        rounded = torch.round(scaled)
        clamped = torch.clamp(rounded, self.quant_min, self.quant_max)
        shifted = clamped - zero_point
        dequantized = shifted * scale
        
        # Use a non-in-place operation for the final result
        result = torch.where(binary_mask, dequantized, original)
        
        return result


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.unsqueeze(-1).expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        comm_loss = 0
        comm_bits = 0

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state

            # Get the next communication vector based on next hidden state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

            # Create mask for masking self communication
            mask = self.comm_mask.view(1, n, n)
            mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)

            # unnecessary, pytorch broadcasting would take care of the desired computation, therefore commented line below
            # mask = mask.expand_as(comm)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            send_graph = agent_mask.squeeze(-1) * agent_mask_transpose.squeeze(-1) * mask.squeeze(-1)

            for b in range(batch_size):
                active = send_graph[b].sum(-1) > 0
                if active.sum() <= 1:  # if there is only one active agent, then it would communicate with just itself
                    continue
                else:
                    indices = torch.where(active)[0]

                    # masking the first agent that is active => This acts as the center of the star graph
                    masked_active = active
                    masked_active[indices[0]] = False
                    masked_active = masked_active.reshape(-1, 1)

                    if self.args.use_comms_channel:
                        # obtain noise
                        noise =  self.get_comms_noise(mask=masked_active, target=comm[b])

                        # add noise to the comms value
                        comm[b] = comm[b] + noise

                        # calculate comms loss
                        comm_loss += self.compute_component_log_loss(comm[b], masked_active)
                        comm_bits += self.compute_num_bits_used(comm[b], masked_active)

                    elif self.args.use_fake_quantization:
                        # Apply fake quantization
                        comm[b] = self.apply_fake_quantization(comm[b], masked_active)
                        comm_loss += self.compute_quantization_loss(comm[b], masked_active)
                        comm_bits += self.compute_quantization_bits(comm[b], masked_active)
                    
                    else:
                        # calculate the comms loss -> 32 * non-zero floating values in the comms vectors
                        comm_loss += self.compute_num_bits_used(comm[b], masked_active)
                        comm_bits += self.compute_num_bits_used(comm[b], masked_active)


            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = comm.sum(dim=1)
            
            # Mean is sent from agent 0 to all other agents
            # So comms loss is calculated as the number of bits for all agents i, where i != 0
            
            # mask just the first agent
            mean_mask = torch.tensor([True if i != 0 else False for i in range(n)], 
                                     dtype=torch.bool, 
                                     device=comm_sum.device).unsqueeze(-1)
            mean_mask = mean_mask.expand(batch_size, n, 1)

            if self.args.use_comms_channel:
                noise = self.get_comms_noise(mean_mask, comm_sum)
                comm_sum = comm_sum + noise
                comm_loss += self.compute_component_log_loss(comm_sum, mean_mask)
                comm_bits += self.compute_num_bits_used(comm_sum, mean_mask)

            elif self.args.use_fake_quantization:
                comm_sum = self.apply_fake_quantization(comm_sum, mean_mask)
                comm_loss += self.compute_quantization_loss(comm_sum, mean_mask)
                comm_bits += self.compute_quantization_bits(comm_sum, mean_mask)

            else:
                comm_loss += self.compute_num_bits_used(comm_sum, mean_mask)
                comm_bits += self.compute_num_bits_used(comm_sum, mean_mask)

            c = self.C_modules[i](comm_sum)

            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c

                inp = inp.view(batch_size * n, self.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            # action = [F.log_softmax(head(h), dim=-1) for head in self.heads]
            raw_action_logits = [head(h) for head in self.heads]

            masked_actions = []

            # Handle the first head (environment action) with masking
            if 'avail_actions' in info:
                # Handle the environment action head (first head)
                env_action_logits = raw_action_logits[0]
                
                avail_mask = info['avail_actions']
                    
                # Convert to tensor if needed
                if not isinstance(avail_mask, torch.Tensor):
                    avail_mask = torch.as_tensor(avail_mask, dtype=torch.bool, device=env_action_logits.device)
                else:
                    avail_mask = avail_mask.bool()

                # Apply the mask before softmax
                mask_value = -1e8
                masked_logits = env_action_logits.clone()
                
                # Expand mask to match batch dimension
                masked_logits = masked_logits.masked_fill(~avail_mask, mask_value)
                
                # Apply softmax after masking
                masked_actions.append(F.log_softmax(masked_logits, dim=-1))
            else:
                # No masks available, just apply softmax
                masked_actions.append(F.log_softmax(raw_action_logits[0], dim=-1))

            # Always add the communication head without masking
            masked_actions.append(F.log_softmax(raw_action_logits[1], dim=-1))

            action = masked_actions

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone()), (comm_loss, comm_bits)
        else:
            return action, value_head, (comm_loss, comm_bits)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def compute_component_log_loss(self, z, mask):
        """
        Computes the communication penalty loss given by log2(2 * |M| * |z| + 1)
        """
        M = self.args.num_messages
        
        loss = torch.log2(2 * M * z.abs() + 1)
        loss = loss * mask
        return torch.sum(loss)
    
    def compute_num_bits_used(self, target, mask):
        bits_used = mask.expand_as(target) * 32
        return torch.sum(bits_used)
    
    def get_comms_noise(self, mask, target):
        # calculate noise as per comms protocol
        noise = (torch.rand_like(target, device=target.device) - 0.5) * 2
        delta = (1 / self.args.num_messages)
        noise = noise * delta * 0.5

        # masking noise wherever not needed
        noise = noise * mask

        return noise
    
    def compute_quantization_loss(self, tensor, mask):
        """
        Compute loss based on the quantized tensor values
        """
        # Expand mask to match tensor dimensions if needed
        if mask.shape != tensor.shape:
            mask_expanded = mask.expand_as(tensor)
        else:
            mask_expanded = mask
        
        binary_mask = mask_expanded.bool()
        
        # Only consider masked values
        if not torch.any(binary_mask):
            return torch.tensor(0.0, device=tensor.device)
        
        masked_tensor = tensor[binary_mask]
        
        # Compute the loss
        loss = torch.log2(2 * self.quant_max * torch.abs(masked_tensor) + 1).sum()
        
        return loss

    def compute_quantization_bits(self, tensor, mask):
        """
        Compute the actual bits used based on the tensor values
        """
        # Count number of values that would be transmitted
        # num_values = mask.sum()
        # Expand mask to match tensor dimensions if needed
        if mask.shape != tensor.shape:
            mask_expanded = mask.expand_as(tensor)
        else:
            mask_expanded = mask
            
        num_values = mask_expanded.sum()
        
        # Each value uses quant_bits bits
        total_bits = self.quant_bits * num_values
        
        return total_bits

