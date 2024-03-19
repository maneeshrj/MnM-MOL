"""
Contains cg block and sense.
"""
import torch
import torch.nn as nn

class cg_block(nn.Module):
    def __init__(self, cgIter, cgTol):
        super(cg_block, self).__init__()
        self.cgIter = cgIter
        self.cgTol = cgTol
        
    def forward(self, lhs, rhs, x0):
        fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b,axis=[-1,-2,-3]))
        x = x0
        r = rhs-lhs(x0)
        p = r
        rTr = fn(r,r)
        eps=torch.tensor(1e-10)
        for i in range(self.cgIter):
            Ap = lhs(p)
            alpha=rTr/(fn(p,Ap)+eps)
            x = x +  alpha[:,None,None,None] * p
            r = r -  alpha[:,None,None,None] * Ap
            rTrNew = fn(r,r)
            if torch.sum(torch.sqrt(rTrNew+eps)) < self.cgTol:
                break
            beta = rTrNew / (rTr+eps)
            p = r + beta[:,None,None,None] * p
            rTr=rTrNew
           
        return x

class sense(nn.Module):
    def __init__(self, cgIter):
        super().__init__()
        
        self.cgIter = cgIter
        self.cg = cg_block(self.cgIter, 1e-9)
        
    def forward(self, img, csm, mask):
        cimg = img*csm
        mcksp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(cimg, dim=[-1,-2]), dim=[-1,-2], norm="ortho"), dim=[-1,-2])
        usksp = mcksp * mask
        return usksp
        
    def adjoint(self, ksp, csm):
        img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
        cs_weighted_img = torch.sum(img*torch.conj(csm),1,True)
        return cs_weighted_img
    
    def ATA(self, img, csm, mask):
        cimg = img*csm
        mcksp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(cimg, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
        usksp = mcksp * mask
        usimg = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(usksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
        cs_weighted_img = torch.sum(usimg*torch.conj(csm),1,True)
        return cs_weighted_img
    
    def inv(self, x0, rhs, lam, csm, mask):
        
        lhs = lambda x: lam*self.ATA(x, csm, mask) + 1.001*x
        out = self.cg(lhs, rhs, x0)
        
        return out