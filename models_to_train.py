import torch.nn as nn

class BaseCompressor(nn.Module):
    """Base class for all compression models with common interface"""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
    def encode(self, x):
        """Encode input to compressed representation"""
        return self.enc(x)
        
    def decode(self, x):
        """Decode compressed representation back to original space"""
        return self.dec(x)
        
    def forward(self, x):
        """Forward pass: encode and decode"""
        z = self.encode(x)
        return z, self.decode(z)
    
    def get_compressed(self, x):
        """Get only compressed representation"""
        return self.encode(x)
    
    def get_reconstructed(self, x):
        """Get only reconstructed representation"""
        z = self.encode(x)
        return self.decode(z)

class LinearCompressor(BaseCompressor):
    """Simple linear projection with decoder"""
    def __init__(self, d_in: int, d_out: int):
        super().__init__(d_in, d_out)
        self.enc = nn.Linear(d_in, d_out, bias=False)
        self.dec = nn.Linear(d_out, d_in, bias=False)

class ShallowAE(BaseCompressor):
    """Shallow autoencoder with one hidden layer"""
    def __init__(self, d_in: int, d_out: int):
        super().__init__(d_in, d_out)
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_in//2),
            nn.ReLU(),
            nn.Linear(d_in//2, d_out)
        )
        self.dec = nn.Sequential(
            nn.Linear(d_out, d_in//2),
            nn.ReLU(),
            nn.Linear(d_in//2, d_in)
        )

class DeepAE(BaseCompressor):
    """Deep autoencoder with multiple hidden layers"""
    def __init__(self, d_in: int, d_out: int):
        super().__init__(d_in, d_out)
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_in//2),
            nn.ReLU(),
            nn.Linear(d_in//2, d_in//4),
            nn.ReLU(),
            nn.Linear(d_in//4, d_out)
        )
        self.dec = nn.Sequential(
            nn.Linear(d_out, d_in//4),
            nn.ReLU(),
            nn.Linear(d_in//4, d_in//2),
            nn.ReLU(),
            nn.Linear(d_in//2, d_in)
        )
