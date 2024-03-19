"""
Contains Lipschitz constant estimation code.
"""
import torch
import torch.nn as nn


def l2_norm(x):
    return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)
   
class LocalLipschitzMasked(nn.Module):
    def __init__(self, x, fg_mask, delta, shape, model,lr=1e8):
        super().__init__()
        self.shape = shape
        self.model = model
        self.delta = delta
        self.lr = lr
        self.gpu=torch.device('cuda')
        
        self.x = x.to(self.gpu)
        self.fg_mask = fg_mask.to(self.gpu)
        self.eps = (self.delta*torch.norm(self.x)).to(self.gpu)
        
        self.v = torch.complex(torch.rand(self.shape,dtype=torch.float32),torch.rand(self.shape,dtype=torch.float32)) 
        self.v = self.v/torch.norm(self.v)
        self.v = self.v.to(self.gpu)*self.fg_mask
        
        self.u = torch.complex(torch.rand(self.shape,dtype=torch.float32),torch.rand(self.shape,dtype=torch.float32)) 
        self.u = self.u/torch.norm(self.u)
        self.u = self.u.to(self.gpu)*self.fg_mask
        
        self.u = self.u.requires_grad_(True)  
        self.v = self.v.requires_grad_(True)  
        
    def compute_ratio(self, model=None):
        if model != None:
            u_out = model(self.eps*self.u*self.fg_mask + self.x)
            v_out = model(self.eps*self.v*self.fg_mask + self.x)
        else:
            u_out = self.model(self.eps*self.u*self.fg_mask + self.x)
            v_out = self.model(self.eps*self.v*self.fg_mask + self.x)
            
        # loss = l2_norm(u_out - v_out) # should be norm(eps*u-eps*v) maybe?
        loss = l2_norm(u_out - v_out)
        loss = loss/(l2_norm(self.u*self.fg_mask - self.v*self.fg_mask)*self.eps)
        return loss

    def adverserial_update(self, iters=1,reinit=False):
        
        if(reinit):
            self.v = torch.complex(torch.rand(self.shape,dtype=torch.float32),torch.rand(self.shape,dtype=torch.float32)) 
            self.v = self.v/torch.norm(self.v)
            self.v = self.v.to(self.gpu)
        
            self.u = torch.complex(torch.rand(self.shape,dtype=torch.float32),torch.rand(self.shape,dtype=torch.float32)) 
            self.u = self.u/torch.norm(self.u)
            self.u = self.u.to(self.gpu)
        
            self.u = self.u.requires_grad_(True)  
            self.v = self.v.requires_grad_(True)  
            
        for i in range(iters):
            loss = self.compute_ratio()
            loss_sum = torch.sum(loss)
            loss_sum.backward()
            
            v_grad = self.v.grad.detach()
            v_tmp = self.v.data + self.lr * v_grad
            v_tmp = v_tmp*self.fg_mask
            if(torch.norm(v_tmp)>1):
                v_tmp = (v_tmp/torch.norm(v_tmp))
            self.v.grad.zero_()
            self.v.data = v_tmp

            u_grad = self.u.grad.detach()
            u_tmp = self.u.data + self.lr * u_grad
            u_tmp = u_tmp*self.fg_mask
            if(torch.norm(u_tmp)>1):
                u_tmp = (u_tmp/torch.norm(u_tmp))
            self.u.grad.zero_()
            self.u.data = u_tmp

            
        self.v = self.v.requires_grad_(False)  
        self.u = self.u.requires_grad_(False)  
    
        loss_sum = self.compute_ratio()
        return loss_sum 

class LipschitzEstimator(nn.Module):
    def __init__(self, shape, eps, lr=1e0):
        super().__init__()
        self.shape = shape
        self.eps = eps
        self.gpu = torch.device('cuda')
        self.lr = lr
        self.noises = self.eps*torch.complex(torch.rand(
            self.shape, dtype=torch.float32), torch.rand(self.shape, dtype=torch.float32))

    def compute_ratio(self, model, u, index):
        noise = torch.clone(self.noises[index:index+1]).to(self.gpu)
        v = u + noise
        u_out = model(u)
        v_out = model(v)
        loss = l2_norm(u_out - v_out)
        loss = loss/l2_norm(u - v)
        return loss

    def adverserial_update(self, model, u, index, iters=1, reinit=False):
        noise = self.noises[index:index+1].to(self.gpu)

        v = u + noise
        v = v.requires_grad_(True)
        for i in range(iters):
            u_out = model(u)
            v_out = model(v)
            loss = l2_norm(u_out - v_out)
            loss = loss/l2_norm(u - v)

            loss_sum = torch.sum(loss)
            loss_sum.backward()

            # noise_grad = noise.grad.detach()
            # noise_tmp = noise.data + self.lr * noise_grad
            # noise_tmp = (noise_tmp/torch.norm(noise_tmp))*torch.norm(noise)

            # noise.grad.zero_()
            v_grad = v.grad.detach()
            v_tmp = v.data + self.lr * v_grad
            v_tmp = (v_tmp/torch.norm(v_tmp))*torch.norm(u)
            v.grad.zero_()

        noise.data = v_tmp - u
        v = v.requires_grad_(False)

        self.noises[index:index+1] = noise.cpu().detach()
        loss = self.compute_ratio(model, u, index)
        return loss


class Lipschitz(nn.Module):
    def __init__(self, u, eps, shape, model, lr=1e8):
        super().__init__()
        self.shape = shape
        self.model = model
        self.lr = lr
        self.eps = eps
        self.u = u
        self.gpu = torch.device('cuda')

        self.u = self.u.to(self.gpu)
        self.eps = self.eps.to(self.gpu)
        self.v = torch.complex(torch.rand(self.shape, dtype=torch.float32), torch.rand(
            self.shape, dtype=torch.float32))
        self.v = self.v.to(self.gpu)
        self.v = self.u + self.eps*self.v

        self.v = self.v.requires_grad_(True)

    def compute_ratio(self, model=None):
        if model != None:
            u_out = model(self.u)
            v_out = model(self.v)
        else:
            u_out = self.model(self.u)
            v_out = self.model(self.v)

        loss = l2_norm(u_out - v_out)
        loss = loss/l2_norm(self.u - self.v)
        return loss

    def adverserial_update(self, iters=1, reinit=False):

        if(reinit):
            self.v = torch.complex(torch.rand(self.shape, dtype=torch.float32), torch.rand(
                self.shape, dtype=torch.float32))
            self.v = self.v.to(self.gpu)
            self.v = self.u + self.eps*self.v

        self.v = self.v.requires_grad_(True)

        for i in range(iters):
            loss = self.compute_ratio()
            loss_sum = torch.sum(loss)
            loss_sum.backward()

            v_grad = self.v.grad.detach()
            v_tmp = self.v.data + self.lr * v_grad
            v_tmp = (v_tmp/torch.norm(v_tmp))*torch.norm(self.u)

            self.v.grad.zero_()
            self.v.data = v_tmp

        self.v = self.v.requires_grad_(False)

        loss_sum = self.compute_ratio()
        return loss_sum


class Lipschitz_model(nn.Module):
    def __init__(self, u, csm, mask, eps, shap, model, lr=1e8):
        super().__init__()
        self.shap = shap
        self.model = model
        self.lr = lr
        self.eps = eps
        self.u = u
        self.csm = csm
        self.mask = mask
        self.gpu = torch.device('cuda')

        self.u = self.u.to(self.gpu)
        self.csm = self.csm.to(self.gpu)
        self.mask = self.mask.to(self.gpu)
        self.eps = self.eps.to(self.gpu)
        self.v = torch.complex(torch.rand(self.shap, dtype=torch.float32), torch.rand(
            self.shap, dtype=torch.float32))
        self.v = self.mask*self.v.to(self.gpu)
        self.v = (self.v/torch.norm(self.v))*torch.norm(self.u)
        self.v = self.v.requires_grad_(True)

    def compute_ratio(self):
        u_out, _, _, _ = self.model(self.u, self.csm, self.mask)
        v_out, _, _, _ = self.model(
            self.u + self.eps*self.v, self.csm, self.mask)
        loss = l2_norm(u_out - v_out)
        loss = loss/l2_norm(self.eps*self.v)
        return loss

    def adverserial_update(self, iters=1, reinit=False):

        if(reinit):
            self.v = torch.complex(torch.rand(self.shap, dtype=torch.float32), torch.rand(
                self.shap, dtype=torch.float32))
            self.v = self.mask*self.v.to(self.gpu)
            self.v = (self.v/torch.norm(self.v))*torch.norm(self.u)

        self.v = self.v.requires_grad_(True)

        for i in range(iters):
            loss = self.compute_ratio()
            loss_sum = torch.sum(loss)
            loss_sum.backward()

            v_grad = (self.mask*self.v.grad).detach()
            v_tmp = self.v.data + self.lr * v_grad
            v_tmp = (v_tmp/torch.norm(v_tmp))*torch.norm(self.u)

            self.v.grad.zero_()
            self.v.data = v_tmp

        self.v = self.v.requires_grad_(False)

        loss_sum = self.compute_ratio()
        return loss_sum
