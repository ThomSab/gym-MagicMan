import torch
import torch.nn as nn

class PlayNet(nn.Module):

    def __init__(self, input_size = 2*334 ,num_actions = 60):
        super().__init__()
                 
        self.fc = nn.Sequential(
              nn.Linear(input_size,256),
              nn.ReLU(),
              nn.Linear(256, num_actions),
              nn.Softmax(dim=-1)
          )

    def forward(self, x):
        x = torch.flatten(x)
        if torch.cuda.is_available():
            x = torch.tensor(x, dtype=torch.float).cuda()
        #if len(x.size()) == 3:
        #  x = x.unsqueeze(dim=0)
        #x = self.features(x)
        #x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
        
class BidNet(nn.Module):

    def __init__(self, input_size = 84, num_actions = 1):
        super().__init__()
           
        self.fc = nn.Sequential(
              nn.Linear(input_size,256),
              nn.ReLU(),
              nn.Linear(256, num_actions),
              nn.Sigmoid()
          )

    def forward(self, x):
        if torch.cuda.is_available():
            x = torch.tensor(x, dtype=torch.float).cuda()
        #if len(x.size()) == 3:
        #  x = x.unsqueeze(dim=0)
        #x = self.features(x)
        #x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x