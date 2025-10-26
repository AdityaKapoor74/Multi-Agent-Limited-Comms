import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention

class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, args):
        super(MAGIC, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.device = args.device
        
        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize, device=self.device)
        self.sub_processor2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize, device=self.device)
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize, device=self.device)

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)

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

    def forward(self, x, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        # n: number of agents

        obs, extras = x
        input_device = obs.device

        comms_loss = 0
        comms_bits = 0

        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs.to(self.device))
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask = agent_mask.to(self.device)

        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1).to(self.device)

        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state.to(self.device), cell_state.to(self.device)))

        # comm: [n * hid_size]
        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)


        # use the communication channel to exchange messages
        if self.args.use_comms_channel:
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
        
        # mask communcation from dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        # sub-scheduler 1
        # if args.first_graph_complete == True, sub-scheduler 1 will be disabled
        if not self.args.first_graph_complete:
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state1 = self.gat_encoder(comm, adj_complete)
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)

        # sub-processor 1
        comm = self.sub_processor1(comm, adj1)

        comm = F.elu(comm)

        # use the communication channel to exchange messages
        if self.args.use_comms_channel:
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
        
        # sub-scheduler 2
        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                if self.args.first_graph_complete:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    encoded_state2 = encoded_state1
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self.get_complete_graph(agent_mask)
            
        # sub-processor 2
        comm = self.sub_processor2(comm, adj2)
        
        # mask communication to dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1).to(input_device) for action_head in self.action_heads]

        # Handle available actions masking
        if 'avail_actions' in info:
            avail_actions = info['avail_actions']
            
            # For list format (separate tensors per action dimension)
            if isinstance(action_out, list):
                # Create a new list to avoid modifying the original if necessary
                masked_action_out = []
                
                for i, act_logits in enumerate(action_out):
                    # Skip if this agent doesn't have available actions info
                    if i >= len(avail_actions):
                        masked_action_out.append(act_logits)
                        continue
                    
                    # Convert avail_actions to appropriate tensor
                    if isinstance(avail_actions[i], torch.Tensor) and avail_actions[i].device == act_logits.device:
                        avail_mask = avail_actions[i].bool()
                    else:
                        avail_mask = torch.as_tensor(avail_actions[i], dtype=torch.bool, device=act_logits.device)
                    
                    # Create masked version safely (no in-place operation)
                    # This creates a new tensor with gradients properly tracked
                    mask_value = -1e8
                    masked_logits = act_logits.clone()
                    masked_logits = masked_logits.masked_fill(~avail_mask, mask_value)
                    masked_action_out.append(masked_logits)
                
                # Replace the original action_out with the masked version
                action_out = masked_action_out
            else:
                # For single tensor output
                if isinstance(avail_actions, np.ndarray):
                    avail_mask = torch.as_tensor(avail_actions, dtype=torch.bool, device=action_out.device)
                else:
                    avail_mask = avail_actions.bool().to(action_out.device)
                
                # Safely create masked version (no in-place operation)
                mask_value = -1e8
                
                # Handle different tensor dimensions
                if action_out.dim() == 2 and avail_mask.dim() == 2:
                    # [num_agents, action_dim] and [num_agents, action_dim]
                    action_out = action_out.masked_fill(~avail_mask, mask_value)
                elif action_out.dim() == 3 and avail_mask.dim() == 2:
                    # [batch, num_agents, action_dim] and [num_agents, action_dim]
                    action_out = action_out.masked_fill(~avail_mask.unsqueeze(0), mask_value)
    

        return action_out, value_head.to(input_device), (hidden_state.clone().to(input_device), cell_state.clone().to(input_device)), (comms_loss, comms_bits)

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    
    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n).to(self.device)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj

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

        return noise.to(self.device)
