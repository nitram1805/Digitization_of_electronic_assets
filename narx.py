from torch import nn

class NARX_4(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(14, 30)
                self.lin2 = nn.Linear(30, 64)
                self.lin3 = nn.Linear(64, 64)
                self.lin4 = nn.Linear(64, 64)
                self.lin5 = nn.Linear(64, 30)              
                self.lin6 = nn.Linear(30, 1)
                self.tanh = nn.RReLU()
                

            def forward(self, xb):
                print(xb.size)
                print(type(xb))
                #xb = xb.reshape(-1,13)
                z = self.lin(xb)
                z = self.tanh(z)
                z = self.lin2(z)
                z = self.tanh(z)
                z = self.lin3(z)
                z = self.tanh(z)
                z = self.lin4(z)
                z = self.tanh(z)
                z = self.lin5(z)
                z = self.tanh(z)
                z = self.lin6(z)
                z = self.tanh(z)                    
                return z