import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_out, bias=False)
        self.dec = nn.Linear(d_out, d_in, bias=False)
    def encode(self, x): return self.enc(x)
    def forward(self, x): z = self.enc(x); return z, self.dec(z)

class ShallowAE(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__(); hid = max(d_out*4, 128)
        self.enc = nn.Sequential(nn.Linear(d_in,hid), nn.ReLU(), nn.Linear(hid,d_out))
        self.dec = nn.Sequential(nn.Linear(d_out,hid), nn.ReLU(), nn.Linear(hid,d_in))
    def encode(self,x): return self.enc(x)
    def forward(self,x): z=self.enc(x); return z, self.dec(z)

class DeepAE(nn.Module):
    def __init__(self, d_in:int, d_out:int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,d_out))
        self.dec = nn.Sequential(nn.Linear(d_out,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,d_in))
    def encode(self,x): return self.enc(x)
    def forward(self,x): z=self.enc(x); return z, self.dec(z)
