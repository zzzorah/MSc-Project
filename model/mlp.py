from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, dropout_rate=0.5, output_size=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        # outputs = outputs.view(outputs.size(0), -1)
        return outputs
