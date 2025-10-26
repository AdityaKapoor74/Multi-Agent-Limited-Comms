"""
Revised from CommNetMLP
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import sys 
sys.path.append("..") 


class GACommNetMLP(nn.Module):
    def __init__(self, args, num_inputs):
        super(GACommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.qk_hid_size = args.qk_hid_size
        self.device = args.device
        
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            # support multi action
            self.heads = nn.ModuleList([nn.Linear(args.hid_size*2, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents).to(self.device)
        else:
            self.comm_mask = (torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)).to(self.device)

        self.encoder = nn.Linear(num_inputs, args.hid_size)

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

        # Our main function for converting current hidden state to next state
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        self.value_head = nn.Linear(2*self.hid_size, 1)

        # hard attention layers to form the graph 
        self.lstm = nn.LSTM(args.hid_size * 2, args.hid_size * 2, bidirectional=True)
        self.linear = nn.Linear(args.hid_size * 4, 2) # *4: after bidirectional output
        
        # soft attention layers 
        self.wq = nn.Linear(args.hid_size, args.qk_hid_size)
        self.wk = nn.Linear(args.hid_size, args.qk_hid_size)

        self.comm_loss = None

        # Add fake quantization configuration
        if hasattr(args, 'use_fake_quantization') and args.use_fake_quantization:
            # Configure quantization parameters
            self.quant_bits = args.quant_bits
            self.quant_min = 0
            self.quant_max = 2**args.quant_bits - 1

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


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask.to(self.device)

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x.to(self.device))

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state.to(self.device), cell_state.to(self.device) if cell_state is not None else None

    def forward(self, x, info={}):
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

        if self.args.recurrent:
            input_device = x[0].device
        else:
            input_device = x.device
        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        if self.args.recurrent:
            inp = x 
            inp = inp.view(batch_size * n, self.hid_size)
            output = self.f_module(inp, (hidden_state, cell_state))
            hidden_state = output[0]
            cell_state = output[1]
        else:
            # Get next hidden state from f node - NOTE: This is not completely correct in GACommNetMLP
            # It appears there's a variable 'c' that's not defined here, but in the original code
            # For now, I'll comment this part out since we'll focus on the attention mechanisms
            # hidden_state = sum([x, self.f_modules[i](hidden_state), c])
            # hidden_state = self.tanh(hidden_state)
            # Just process hidden state with f_module
            hidden_state = self.f_modules[0](hidden_state)
            hidden_state = self.tanh(hidden_state)

        z = hidden_state.reshape(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
        
        comms_loss = 0
        comms_bits = 0

        if not self.args.comm_mask_zero:
            if not self.args.comm_action_one:
                # Initialize h0 and c0 for LSTM
                h0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True).to(self.device)
                c0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True).to(self.device)

                # ----- IMPORTANT CHANGE: Apply communication noise to hidden states before hard attention -----
                # Apply noise or quantization to hidden state before hard attention calculation
                hidden_state_comm = hidden_state.clone()
                
                # Create a full mask for all hidden state values
                full_mask = torch.ones_like(hidden_state_comm, device=hidden_state_comm.device)
                
                if hasattr(self.args, 'use_comms_channel') and self.args.use_comms_channel:
                    noise = self.get_comms_noise(full_mask, hidden_state_comm)
                    hidden_state_comm = hidden_state_comm + noise
                    comms_loss += self.compute_component_log_loss(hidden_state_comm, full_mask)
                    comms_bits += self.compute_num_bits_used(hidden_state_comm, full_mask)
                elif hasattr(self.args, 'use_fake_quantization') and self.args.use_fake_quantization:
                    hidden_state_comm = self.apply_fake_quantization(hidden_state_comm, full_mask)
                    comms_loss += self.compute_quantization_loss(hidden_state_comm, full_mask)
                    comms_bits += self.compute_quantization_bits(hidden_state_comm, full_mask)
                else:
                    comms_bits += self.compute_num_bits_used(hidden_state_comm, full_mask)
                    comms_loss += self.compute_num_bits_used(hidden_state_comm, full_mask)

                # according to the paper, there is no self-attention
                list1 = []
                for a in range(batch_size * n):
                    # Use the noisy/quantized hidden state for hard attention calculation
                    list2 = [torch.cat([hidden_state_comm[a], hidden_state_comm[b]]) for b in range(batch_size * n) if b != a]
                    list1.append(torch.stack(list2))
                    
                # hard_attn_input: size of (N-1) x N x (hid_size*2)
                hard_attn_input = torch.stack(list1, dim=1)
                # hard_attn_output: size of (N-1) x N x 2, the third dimension is one-hot vector
                hard_attn_output = self.lstm(hard_attn_input, (h0, c0))[0]
                hard_attn_output = self.linear(hard_attn_output)
                hard_attn_output = F.gumbel_softmax(hard_attn_output, hard=True) 
                # hard_attn_output: size of (N-1) x N x 1
                hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
                # hard_attn_output: size of N x (N-1)
                hard_attn_output = hard_attn_output.permute(1, 0, 2).squeeze()
            else:
                hard_attn_output = self.get_hard_attn_one(agent_mask)

            comm_density1 = hard_attn_output.nonzero().size(0) / (n*n)
            comm_density2 = hard_attn_output.nonzero().size(0) / (n*(n-1))

            # Calculate query and key for soft attention
            q = self.wq(hidden_state)
            k = self.wk(hidden_state)

            mask = self.convert_hard_attn_to_mask(hard_attn_output).unsqueeze(-1)

            if hasattr(self.args, 'use_comms_channel') and self.args.use_comms_channel:
                k = k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size)
                noise = self.get_comms_noise(mask, k)
                k = k + noise

                comms_loss += self.compute_component_log_loss(k, mask)
                comms_bits += self.compute_num_bits_used(k, mask)

                soft_attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / np.sqrt(self.qk_hid_size)
                soft_attn = soft_attn.squeeze(1)

            elif hasattr(self.args, 'use_fake_quantization') and self.args.use_fake_quantization:
                k = k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size)
                k = self.apply_fake_quantization(k, mask)
                comms_loss += self.compute_quantization_loss(k, mask)
                comms_bits += self.compute_quantization_bits(k, mask)
                
                soft_attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / np.sqrt(self.qk_hid_size)
                soft_attn = soft_attn.squeeze(1)

            else:
                comms_bits += self.compute_num_bits_used(k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size), mask)
                comms_loss += self.compute_num_bits_used(k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size), mask)
                # size of N x N
                soft_attn = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.qk_hid_size)
            
            # size of N x (N-1)
            soft_attn = torch.stack([torch.cat([soft_attn[l][:l], soft_attn[l][l+1:batch_size*n]]) for l in range(batch_size*n)])
            soft_attn = soft_attn * hard_attn_output
            soft_attn = F.softmax(soft_attn, dim=1)
            attn = soft_attn * hard_attn_output

            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
            comm = comm * agent_mask

            if hasattr(self.args, 'use_comms_channel') and self.args.use_comms_channel:
                noise = self.get_comms_noise(agent_mask, comm)
                comm = comm + noise

                comms_loss += self.compute_component_log_loss(comm, agent_mask)
                comms_bits += self.compute_num_bits_used(comm, agent_mask)

            elif hasattr(self.args, 'use_fake_quantization') and self.args.use_fake_quantization:
                comm = self.apply_fake_quantization(comm, agent_mask)
                comms_loss += self.compute_quantization_loss(comm, agent_mask)
                comms_bits += self.compute_quantization_bits(comm, agent_mask)

            else:
                comms_loss += self.compute_num_bits_used(comm, agent_mask)
                comms_bits += self.compute_num_bits_used(comm, agent_mask)

            comm = comm.view(batch_size * n, self.hid_size)
            # can also add 0 for self-connections in attn, and act like tar_comm, can try to cross-verify
            comm = torch.stack([(attn[l].reshape(batch_size*n-1,1)*torch.cat([comm[:l], comm[l+1:batch_size*n]], dim=0)).sum(dim=0) for l in range(batch_size*n)], dim=0)
            comm = comm.view(batch_size, n, self.hid_size)
            comm = comm * agent_mask
        else:
            comm = torch.zeros(batch_size, n, self.hid_size) if self.args.recurrent else torch.zeros(hidden_state.size())
            comm_density1 = 0
            comm_density2 = 0

        value_head = self.value_head(torch.cat((hidden_state, comm.view(batch_size*n, self.hid_size)), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)

        # if self.continuous:
        #     action_mean = self.action_mean(h)
        #     action_log_std = self.action_log_std.expand_as(action_mean)
        #     action_std = torch.exp(action_log_std)
        #     # will be used later to sample
        #     action = (action_mean.to(input_device), action_log_std.to(input_device), action_std.to(input_device))
        # else:
        #     # discrete actions
        #     action = [F.log_softmax(head(torch.cat((h, comm), dim=-1)), dim=-1).to(input_device) for head in self.heads]

        if self.continuous:
            action_mean = self.action_mean(torch.cat((h, comm), dim=-1))
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            # action = [F.log_softmax(head(h), dim=-1) for head in self.heads]
            raw_action_logits = [head(torch.cat((h, comm), dim=-1)) for head in self.heads]

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
            # masked_actions.append(F.log_softmax(raw_action_logits[1], dim=-1))

            action = masked_actions
        
        
        if self.args.recurrent:
            return action, value_head.to(input_device), (hidden_state.clone().to(input_device), cell_state.clone().to(input_device)), [comm_density1, comm_density2], (comms_loss, comms_bits)
        else:
            return action, value_head.to(input_device), [comm_density1, comm_density2], (comms_loss, comms_bits)

    def get_hard_attn_one(self, agent_mask):
        n = self.args.nagents
        adj = torch.ones(n, n).to(agent_mask.device)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        hard_attn = torch.stack([torch.cat([adj[x][:x], adj[x][x+1:n]]) for x in range(n)])

        return hard_attn
    
    def convert_hard_attn_to_mask(self, hard_attn):
        n = hard_attn.size(0)
        # Create an n x n zero matrix
        A_full = torch.zeros(n, n).to(hard_attn.device)

        # Use indexing to fill in the values
        mask = torch.ones(n, n, dtype=torch.bool).to(hard_attn.device)
        mask.fill_diagonal_(False)  # False on the diagonal
        A_full[mask] = hard_attn.flatten()  # Fill non-diagonal elements

        return A_full
    
    def compute_component_log_loss(self, z, mask):
        """
        Computes the communication penalty loss given by log2(2 * |M| * |z| + 1)
        """
        M = self.args.num_messages if hasattr(self.args, 'num_messages') else 256
        
        loss = torch.log2(2 * M * z.abs() + 1)
        loss = loss * mask
        return torch.sum(loss)
    
    def compute_num_bits_used(self, target, mask):
        bits_used = mask.expand_as(target) * 32
        return torch.sum(bits_used)
    
    def get_comms_noise(self, mask, target):
        # calculate noise as per comms protocol
        noise = (torch.rand_like(target, device=target.device) - 0.5) * 2
        delta = (1 / (self.args.num_messages if hasattr(self.args, 'num_messages') else 256))
        noise = noise * delta * 0.5

        # masking noise wherever not needed
        noise = noise * mask

        return noise

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

# class GACommNetMLP(nn.Module):
#     def __init__(self, args, num_inputs):
#         super(GACommNetMLP, self).__init__()
#         self.args = args
#         self.nagents = args.nagents
#         self.hid_size = args.hid_size
#         self.comm_passes = args.comm_passes
#         self.recurrent = args.recurrent
#         self.qk_hid_size = args.qk_hid_size
#         self.device = args.device
        

#         self.continuous = args.continuous
#         if self.continuous:
#             self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
#             self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
#         else:
#             # support multi action
#             self.heads = nn.ModuleList([nn.Linear(args.hid_size*2, o)
#                                         for o in args.naction_heads])
#         self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

#         # Mask for communication
#         if self.args.comm_mask_zero:
#             self.comm_mask = torch.zeros(self.nagents, self.nagents).to(self.device)
#         else:
#             self.comm_mask = (torch.ones(self.nagents, self.nagents) \
#                             - torch.eye(self.nagents, self.nagents)).to(self.device)

#         self.encoder = nn.Linear(num_inputs, args.hid_size)

#         if args.recurrent:
#             self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

#         if args.recurrent:
#             self.init_hidden(args.batch_size)
#             self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

#         else:
#             if args.share_weights:
#                 self.f_module = nn.Linear(args.hid_size, args.hid_size)
#                 self.f_modules = nn.ModuleList([self.f_module
#                                                 for _ in range(self.comm_passes)])
#             else:
#                 self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
#                                                 for _ in range(self.comm_passes)])
#         # else:
#             # raise RuntimeError("Unsupported RNN type.")

#         # Our main function for converting current hidden state to next state
#         # self.f = nn.Linear(args.hid_size, args.hid_size)
#         if args.share_weights:
#             self.C_module = nn.Linear(args.hid_size, args.hid_size)
#             self.C_modules = nn.ModuleList([self.C_module
#                                             for _ in range(self.comm_passes)])
#         else:
#             self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
#                                             for _ in range(self.comm_passes)])
#         # self.C = nn.Linear(args.hid_size, args.hid_size)

#         # initialise weights as 0
#         if args.comm_init == 'zeros':
#             for i in range(self.comm_passes):
#                 self.C_modules[i].weight.data.zero_()
#         self.tanh = nn.Tanh()

#         # print(self.C)
#         # self.C.weight.data.zero_()
#         # Init weights for linear layers
#         # self.apply(self.init_weights)

#         self.value_head = nn.Linear(2*self.hid_size, 1)

#         # hard attention layers to form the graph 
#         self.lstm = nn.LSTM(args.hid_size * 2, args.hid_size * 2, bidirectional=True)
#         self.linear = nn.Linear(args.hid_size * 4, 2) # *4: after bidirectional output
#         # soft attention layers 
#         self.wq = nn.Linear(args.hid_size, args.qk_hid_size)
#         self.wk = nn.Linear(args.hid_size, args.qk_hid_size)

#         self.comm_loss = None

#         # Add fake quantization configuration
#         if hasattr(args, 'use_fake_quantization') and args.use_fake_quantization:
#             # Configure quantization parameters
#             self.quant_bits = args.quant_bits
#             self.quant_min = 0
#             self.quant_max = 2**args.quant_bits - 1

#     def apply_fake_quantization(self, tensor, mask):
#         """
#         Manual implementation of fake quantization with proper gradient flow
#         """
#         # Create a copy of the input tensor to avoid modifying the original
#         original = tensor.clone()
        
#         # Expand mask to match tensor dimensions
#         if mask.shape != tensor.shape:
#             mask_expanded = mask.expand_as(tensor)
#         else:
#             mask_expanded = mask
            
#         binary_mask = mask_expanded.bool()
        
#         # Get min/max values safely
#         if torch.any(binary_mask):
#             # Detach for min/max calculation to avoid affecting gradient computation
#             masked_values = tensor[binary_mask].detach()
#             min_val = masked_values.min()
#             max_val = masked_values.max()
#         else:
#             min_val = torch.tensor(0.0, device=tensor.device)
#             max_val = torch.tensor(1.0, device=tensor.device)
        
#         if min_val == max_val:
#             min_val = min_val - 0.01
#             max_val = max_val + 0.01
        
#         # Calculate scale and zero_point from detached values
#         scale = (max_val - min_val) / (self.quant_max - self.quant_min)
#         zero_point = torch.round(self.quant_min - min_val / scale)
        
#         # Apply quantization formula without modifying the original
#         scaled = tensor / scale + zero_point
#         rounded = torch.round(scaled)
#         clamped = torch.clamp(rounded, self.quant_min, self.quant_max)
#         shifted = clamped - zero_point
#         dequantized = shifted * scale
        
#         # Use a non-in-place operation for the final result
#         result = torch.where(binary_mask, dequantized, original)
        
#         return result

#     def compute_quantization_loss(self, tensor, mask):
#         """
#         Compute loss based on the quantized tensor values
#         """
#         # Expand mask to match tensor dimensions if needed
#         if mask.shape != tensor.shape:
#             mask_expanded = mask.expand_as(tensor)
#         else:
#             mask_expanded = mask
        
#         binary_mask = mask_expanded.bool()
        
#         # Only consider masked values
#         if not torch.any(binary_mask):
#             return torch.tensor(0.0, device=tensor.device)
        
#         masked_tensor = tensor[binary_mask]
        
#         # Compute the loss
#         loss = torch.log2(2 * self.quant_max * torch.abs(masked_tensor) + 1).sum()
        
#         return loss

#     def compute_quantization_bits(self, tensor, mask):
#         """
#         Compute the actual bits used based on the tensor values
#         """
#         # Count number of values that would be transmitted
#         num_values = mask.sum()
        
#         # Each value uses quant_bits bits
#         total_bits = self.quant_bits * num_values
        
#         return total_bits


#     def get_agent_mask(self, batch_size, info):
#         n = self.nagents

#         if 'alive_mask' in info:
#             agent_mask = torch.from_numpy(info['alive_mask'])
#             num_agents_alive = agent_mask.sum()
#         else:
#             agent_mask = torch.ones(n)
#             num_agents_alive = n

#         agent_mask = agent_mask.view(n, 1).clone()

#         return num_agents_alive, agent_mask.to(self.device)

#     def forward_state_encoder(self, x):
#         hidden_state, cell_state = None, None

#         if self.args.recurrent:
#             x, extras = x
#             x = self.encoder(x.to(self.device))

#             if self.args.rnn_type == 'LSTM':
#                 hidden_state, cell_state = extras
#             else:
#                 hidden_state = extras
#             # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
#         else:
#             x = self.encoder(x)
#             x = self.tanh(x)
#             hidden_state = x

#         return x, hidden_state.to(self.device), cell_state.to(self.device)


#     def forward(self, x, info={}):
#         # TODO: Update dimensions
#         """Forward function for CommNet class, expects state, previous hidden
#         and communication tensor.
#         B: Batch Size: Normally 1 in case of episode
#         N: number of agents
#         Arguments:
#             x {tensor} -- State of the agents (N x num_inputs)
#             prev_hidden_state {tensor} -- Previous hidden state for the networks in
#             case of multiple passes (1 x N x hid_size)
#             comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)
#         Returns:
#             tuple -- Contains
#                 next_hidden {tensor}: Next hidden state for network
#                 comm_out {tensor}: Next communication tensor
#                 action_data: Data needed for taking next action (Discrete values in
#                 case of discrete, mean and std in case of continuous)
#                 v: value head
#         """

#         # if self.args.env_name == 'starcraft':
#         #     maxi = x.max(dim=-2)[0]
#         #     x = self.state_encoder(x)
#         #     x = x.sum(dim=-2)
#         #     x = torch.cat([x, maxi], dim=-1)
#         #     x = self.tanh(x)
#         if self.args.recurrent:
#             input_device = x[0].device
#         else:
#             input_device = x.device
#         x, hidden_state, cell_state = self.forward_state_encoder(x)

#         batch_size = x.size()[0]
#         n = self.nagents

#         num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

#         # # Hard Attention - action whether an agent communicates or not
#         # if self.args.hard_attn:
#         #     comm_action = torch.tensor(info['comm_action'])
#         #     comm_action_mask = comm_action.expand(batch_size, n).unsqueeze(-1)
#         #     # action 1 is talk, 0 is silent i.e. act as dead for comm purposes. ???
#         #     agent_mask *= comm_action_mask.double()            

#         if self.args.recurrent:
#             inp = x 

#             inp = inp.view(batch_size * n, self.hid_size)

#             output = self.f_module(inp, (hidden_state, cell_state))

#             hidden_state = output[0]
#             cell_state = output[1]
#         else: # MLP|RNN
#             # Get next hidden state from f node
#             # and Add skip connection from start and sum them
#             # bugs to be fixed 
#             hidden_state = sum([x, self.f_modules[i](hidden_state), c])
#             hidden_state = self.tanh(hidden_state)

#         z = hidden_state.reshape(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
        
#         comms_loss = 0
#         comms_bits = 0

#         if not self.args.comm_mask_zero:
#             if not self.args.comm_action_one:
#                 # TO DO: should do batch-wise
#                 h0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True).to(self.device)
#                 c0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True).to(self.device)

#                 # according to the paper, there is no self-attention
#                 list1 = []
#                 for a in range(batch_size * n):
#                     list2 = [torch.cat([hidden_state[a], hidden_state[b]]) for b in range(batch_size * n) if b != a]
#                     # for b in range(batch_size * n):
#                     #     if a != b:
#                     #         list2.append(torch.cat([hidden_state[a], hidden_state[b]]))
#                     list1.append(torch.stack(list2))
#                 # hard_attn_input: size of (N-1) x N x (hid_size*2)
#                 hard_attn_input = torch.stack(list1, dim=1)
#                 # hard_attn_output: size of (N-1) x N x 2, the third dimension is one-hot vector
#                 hard_attn_output = self.lstm(hard_attn_input, (h0, c0))[0]
#                 hard_attn_output = self.linear(hard_attn_output)
#                 hard_attn_output = F.gumbel_softmax(hard_attn_output, hard=True) 
#                 # hard_attn_output: size of (N-1) x N x 1
#                 hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
#                 # hard_attn_output: size of N x (N-1)
#                 hard_attn_output = hard_attn_output.permute(1, 0, 2).squeeze()
#             else:
#                 hard_attn_output = self.get_hard_attn_one(agent_mask)

#             comm_density1 = hard_attn_output.nonzero().size(0) / (n*n)
#             comm_density2 = hard_attn_output.nonzero().size(0) / (n*(n-1))

#             # calculate query and key for soft attention
#             q = self.wq(hidden_state)
#             k = self.wk(hidden_state)

#             mask = self.convert_hard_attn_to_mask(hard_attn_output).unsqueeze(-1)

#             if self.args.use_comms_channel:
#                 k = k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size)
#                 noise = self.get_comms_noise(mask, k)
#                 k = k + noise

#                 comms_loss += self.compute_component_log_loss(k, mask)
#                 comms_bits += self.compute_num_bits_used(k, mask)

#                 soft_attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / np.sqrt(self.qk_hid_size)
#                 soft_attn = soft_attn.squeeze(1)

#             elif hasattr(self.args, 'use_fake_quantization') and self.args.use_fake_quantization:
#                 k = k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size)
#                 k = self.apply_fake_quantization(k, mask)
#                 comms_loss += self.compute_quantization_loss(k, mask)
#                 comms_bits += self.compute_quantization_bits(k, mask)
                
#                 soft_attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / np.sqrt(self.qk_hid_size)
#                 soft_attn = soft_attn.squeeze(1)

#             else:
#                 comms_bits += self.compute_num_bits_used(k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size), mask)
#                 comms_loss += self.compute_num_bits_used(k.unsqueeze(1).expand(batch_size * n, n, self.qk_hid_size), mask)
#                 # size of N x N
#                 soft_attn = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.qk_hid_size)
            
#             # size of N x (N-1)
#             soft_attn = torch.stack([torch.cat([soft_attn[l][:l], soft_attn[l][l+1:batch_size*n]]) for l in range(batch_size*n)])
#             soft_attn = soft_attn * hard_attn_output
#             soft_attn = F.softmax(soft_attn, dim=1)
#             attn = soft_attn * hard_attn_output

#             # Choose current or prev depending on recurrent
#             comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
#             comm = comm * agent_mask

#             if self.args.use_comms_channel:
#                 noise = self.get_comms_noise(agent_mask, comm)
#                 comm = comm + noise

#                 comms_loss += self.compute_component_log_loss(comm, agent_mask)
#                 comms_bits += self.compute_num_bits_used(comm, agent_mask)

#             elif hasattr(self.args, 'use_fake_quantization') and self.args.use_fake_quantization:
#                 comm = self.apply_fake_quantization(comm, agent_mask)
#                 comms_loss += self.compute_quantization_loss(comm, agent_mask)
#                 comms_bits += self.compute_quantization_bits(comm, agent_mask)

#             else:
#                 comms_loss += self.compute_num_bits_used(comm, agent_mask)
#                 comms_bits += self.compute_num_bits_used(comm, agent_mask)

#             comm = comm.view(batch_size * n, self.hid_size)
#             # can also add 0 for self-connections in attn, and act like tar_comm, can try to cross-verify
#             comm = torch.stack([(attn[l].reshape(batch_size*n-1,1)*torch.cat([comm[:l], comm[l+1:batch_size*n]], dim=0)).sum(dim=0) for l in range(batch_size*n)], dim=0)
#             comm = comm.view(batch_size, n, self.hid_size)
#             comm = comm * agent_mask
#         else:
#             comm = torch.zeros(batch_size, n, self.hid_size) if self.args.recurrent else torch.zeros(hidden_state.size())
#             comm_density1 = 0
#             comm_density2 = 0

#         value_head = self.value_head(torch.cat((hidden_state, comm.view(batch_size*n, self.hid_size)), dim=-1))
#         h = hidden_state.view(batch_size, n, self.hid_size)

#         if self.continuous:
#             action_mean = self.action_mean(h)
#             action_log_std = self.action_log_std.expand_as(action_mean)
#             action_std = torch.exp(action_log_std)
#             # will be used later to sample
#             action = (action_mean.to(input_device), action_log_std.to(input_device), action_std.to(input_device))
#         else:
#             # discrete actions
#             action = [F.log_softmax(head(torch.cat((h, comm), dim=-1)), dim=-1).to(input_device) for head in self.heads]

#         if self.args.recurrent:
#             return action, value_head.to(input_device), (hidden_state.clone().to(input_device), cell_state.clone().to(input_device)), [comm_density1, comm_density2], (comms_loss, comms_bits)
#         else:
#             return action, value_head.to(input_device), [comm_density1, comm_density2], (comms_loss, comms_bits)

#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             m.weight.data.normal_(0, self.init_std)

#     def init_hidden(self, batch_size):
#         # dim 0 = num of layers * num of direction
#         return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
#                        torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
#     def get_hard_attn_one(self, agent_mask):
#         n = self.args.nagents
#         adj = torch.ones(n, n).to(agent_mask.device)
#         agent_mask = agent_mask.expand(n, n)
#         agent_mask_transpose = agent_mask.transpose(0, 1)
#         adj = adj * agent_mask * agent_mask_transpose
#         hard_attn = torch.stack([torch.cat([adj[x][:x], adj[x][x+1:n]]) for x in range(n)])

#         return hard_attn
    
#     def convert_hard_attn_to_mask(self, hard_attn):
#         n = hard_attn.size(0)
#         # Create an n x n zero matrix
#         A_full = torch.zeros(n, n).to(hard_attn.device)

#         # Use indexing to fill in the values
#         mask = torch.ones(n, n, dtype=torch.bool).to(hard_attn.device)
#         mask.fill_diagonal_(False)  # False on the diagonal
#         A_full[mask] = hard_attn.flatten()  # Fill non-diagonal elements

#         return A_full
    
#     def compute_component_log_loss(self, z, mask):
#         """
#         Computes the communication penalty loss given by log2(2 * |M| * |z| + 1)
#         """
#         M = self.args.num_messages
        
#         loss = torch.log2(2 * M * z.abs() + 1)
#         loss = loss * mask
#         return torch.sum(loss)
    
#     def compute_num_bits_used(self, target, mask):
#         bits_used = mask.expand_as(target) * 32
#         return torch.sum(bits_used)
    
#     def get_comms_noise(self, mask, target):
#         # calculate noise as per comms protocol
#         noise = (torch.rand_like(target, device=target.device) - 0.5) * 2
#         delta = (1 / self.args.num_messages)
#         noise = noise * delta * 0.5

#         # masking noise wherever not needed
#         noise = noise * mask

#         return noise


