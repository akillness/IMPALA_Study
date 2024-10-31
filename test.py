import torch
import torch.nn as nn
import torch.nn.functional as F

class InstructionProcessor(nn.Module):
    def __init__(self, num_hash_buckets=1000, embedding_size=20, lstm_hidden_size=64):
        super(InstructionProcessor, self).__init__()
        self.num_hash_buckets = num_hash_buckets
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(num_hash_buckets, embedding_size)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True)

    def forward(self, instruction):
        # Split string and convert to hash buckets
        splitted = instruction.split()
        buckets = [hash(word) % self.num_hash_buckets for word in splitted]
        buckets = torch.tensor(buckets, dtype=torch.long).unsqueeze(0)  # Add batch dimension

        # Embed the instruction
        embedding = self.embedding(buckets)

        # Pad to make sure there is at least one output
        if embedding.size(1) == 0:
            embedding = F.pad(embedding, (0, 0, 0, 1))

        # LSTM processing
        lengths = torch.tensor([len(splitted)], dtype=torch.long)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Return last output
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2)).unsqueeze(1)
        last_output = output.gather(1, idx).squeeze(1)

        return last_output

# # Example usage
# instruction_processor = InstructionProcessor()
# instruction = "your instruction string here"
# output = instruction_processor(instruction)
# print(output)

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

def residual_block(x, num_ch):
    identity = x
    x = F.relu(x)
    x = conv_block(num_ch, num_ch)(x)
    x = F.relu(x)
    x = conv_block(num_ch, num_ch)(x)
    x += identity
    return x

def convnet_forward(x):
    x = x.unsqueeze(1).unsqueeze(2)  # Add channel and height dimensions
    x = conv_block(1, 16)(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Adjust the input channels to match the expected channels
    x = conv_block(16, 32)(x)
    
    for i, (num_ch, num_blocks) in enumerate([(32, 2), (32, 2), (32, 2)]):
        x = conv_block(num_ch, num_ch)(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        for j in range(num_blocks):
            x = residual_block(x, num_ch)
    
    return x

# Example usage with time series data:
# Assuming time_series_data is a 2D tensor of shape (batch_size, sequence_length)
time_series_data = torch.randn(4, 4)  # Example time series data
output = convnet_forward(time_series_data)
print(output.shape)

