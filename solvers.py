"""
Contains solvers and solver blocks for MOL, MOL+, and MoDL.
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import spectral_norm_chen as chen

def complex_to_real(x):
    xr = torch.cat((torch.real(x),torch.imag(x)),dim=1)
    xr = xr.type(torch.float32)
    return xr

def real_to_complex(x):
    re,im = torch.split(x,[1,1],dim=1)
    xc = re + 1j*im
    xc = xc.type(torch.complex64)
    return xc


# Convolutional layer
class convlayer(nn.Module):
    
    def __init__(self, input_channels, output_channels, last, sn=False):
        super(convlayer, self).__init__()
        
        if sn:
            self.conv = chen.spectral_norm(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1,bias=True))
        else:
            self.conv = nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1,bias=True)
        
        # self.conv = nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.relu = nn.ReLU()
        self.last = last
        
    def forward(self,x):
        x = self.conv(x)
        if not self.last:
            x = self.relu(x)
        return x


# CNN denoiser
class dwblock(nn.Module):
    def __init__(self, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(dwblock, self).__init__()
    
        self.num_layers = number_of_layers
        layers = []
        layers.append(convlayer(input_channels*2, features, False, spectral_norm))
        for i in range(1, self.num_layers-1):
            layers.append(convlayer(features, features, False, spectral_norm)) # conv layer
        layers.append(convlayer(features, output_channels*2, True, spectral_norm))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        out = real_to_complex(self.net(complex_to_real(x)))
        return out


#
# DE-Grad block for MnM-MoDL
#
class deGrad(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(deGrad, self).__init__()
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        self.A = A
        self.alpha = torch.tensor(0.1,dtype=torch.float32)
        # self.Lmax = Lmax
        
    def forward(self, x, Atb, csm, mask):
        gamma = 0.1/self.lam
        #gamma = 2/self.lam/(1+ self.Lmax**2)
        z = self.A.ATA(x,csm, mask)
        rhs = self.lam*(z-Atb) +  self.dw(x)
        x = x - gamma*rhs
        return x
    
    def Q(self, x, csm, mask):
        
        return x-self.A.ATA(x,csm,mask) - self.dw(x)/self.lam


#
# Forward-backward block for MOL
#
class fwdbwdBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(fwdbwdBlock, self).__init__()
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        self.A = A
        self.alpha = torch.tensor(0.1,dtype=torch.float32)
    
        
    def forward(self, x, Atb, csm, mask):
        z = self.dw(x)
        rhs = (1 - self.alpha)*x + self.alpha*z + self.alpha*self.lam*Atb
        x = self.A.inv(x, rhs, self.alpha*self.lam, csm, mask)
        return x
    
    def Q(self, x, csm, mask):
        return self.dw(x)


#
# Deep Equilibrium solver for MOL and MnM-MoDL
#
class DEQ(nn.Module):
    def __init__(self,f,K,A_init,lam_init,tol=0.05,verbose=True):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init = lam_init
        self.tol = tol
        self.verbose = verbose

    def forward(self, b, csm, mask):

        Atb = self.f.A.adjoint(b, csm).to(b.device)
        zero = torch.zeros_like(Atb).to(Atb.device)
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        x = sense_out
        with torch.no_grad():         
            
            for blk in range(self.K):
                xold = x
                x=self.f(x, Atb, csm, mask)
                errforward = torch.norm(x-xold)/torch.norm(xold)
                if(self.verbose):
                    print(errforward)
                    print("diff", torch.norm(x-xold).cpu().numpy()," xnew ",torch.norm(x).cpu().numpy()," xold ",torch.norm(xold).cpu().numpy())
                if(errforward < self.tol and blk>2):
                    if(self.verbose):
                        print("exiting front prop after ",blk," iterations with ",errforward )
                    break
                    
        
        z = self.f(x, Atb, csm, mask)  # forward layer seen by pytorch
            
        # For computation of Jacobian vector product
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, Atb, csm, mask)
        
       # Backward propagation of gradients
        def backward_hook(grad):
            g = grad
            for i in range(self.K):
                gold = g
                g = autograd.grad(f0,z0,gold,retain_graph=True)[0] + grad
                errback = torch.norm(g-gold)/torch.norm(gold)
                if(errback < self.tol):
                    if(self.verbose):
                        print("exiting back prop after ",blk," iterations with ",errback )
                    break
            g = autograd.grad(f0,z0,gold)[0] + grad
            #g = torch.clamp(g,min=-1,max=1)
            return(g)

       # Adding the hook to modify the gradients
        z.register_hook(backward_hook)
    
        return z, sense_out, errforward, blk

#
# DEQ solver for inference
#
class DEQ_inf(nn.Module):
    def __init__(self,f,K,A_init,lam_init,tol=0.05,verbose=True):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init = lam_init
        self.tol = tol
        self.verbose = verbose

    def forward(self, b, csm, mask):

        Atb = self.f.A.adjoint(b, csm)
        zero = torch.zeros_like(Atb).to(Atb.device)
        import time
        start = time.time()
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        end=time.time()
        if(self.verbose): print('sense time is:', end-start)
        x = sense_out
        
        
        with torch.no_grad():         
            
            for blk in range(self.K):
                xold = x
                x=self.f(x, Atb, csm, mask)
                errforward = torch.norm(x-xold)/torch.norm(xold)
                if(self.verbose):
                    print(errforward)
                    print("diff", torch.norm(x-xold).cpu().numpy()," xnew ",torch.norm(x).cpu().numpy()," xold ",torch.norm(xold).cpu().numpy())
                if(errforward < self.tol and blk>2):
                    if(self.verbose):
                        print("exiting front prop after ",blk," iterations with ",errforward )
                    break
                    
        
        z = self.f(x, Atb, csm, mask)  # forward layer seen by pytorch
            
    
        return z, sense_out, errforward, blk


#
# MoDL block
#
class modlBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(modlBlock, self).__init__()
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        self.A = A
        self.alpha = torch.tensor(1.0,dtype=torch.float32)
    
        
    def forward(self, x, Atb, csm, mask):
        z = x - self.dw(x)
        rhs = z + self.lam*Atb
        x = self.A.inv(x, rhs, self.lam, csm, mask)
        return x
    

#
# Unrolled solver for MoDL
#
class Unroll(nn.Module):
    def __init__(self,f,K,A_init,lam_init):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init=lam_init
        
    def forward(self, b, csm, mask):

        Atb = self.f.A.adjoint(b, csm)
        
        zero = torch.zeros_like(Atb).to(Atb.device)
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        x = sense_out
        
        for blk in range(self.K):
            xold = x
            x = self.f(x, Atb, csm, mask)
        
        errforward = torch.norm(x-xold)/torch.norm(xold)
        return x, sense_out, errforward, self.K