import torch    
import torch.nn as nn
import torch.nn.functional as F
from rk4 import rk4
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from torch.optim import Adam,SGD,RMSprop
from dmd import dmd_torch
# from torchviz import make_dot   

class PassiveControlAutodif(nn.Module):
    def __init__(self, model: Model):
        super(PassiveControlAutodif, self).__init__()
        self.model = model
        self.X0=self.model.set_initial_conditions()
    def update_matrices_with_control_parameters(self, Kbeta: torch.Tensor, Dbeta: torch.Tensor, xh: torch.Tensor):
        self.model.update_control_parameters(Kbeta=Kbeta, xh=xh,Dbeta=Dbeta)
        self.Q=self.model.make_first_order_matrix()
    
    def compute_energy(self,h,alpha,beta,h_dot,alpha_dot,beta_dot):
        Ec=1/2*self.model.m*h_dot**2 + 1/2*self.model.Ialpha*alpha_dot**2+self.model.salpha*h_dot*alpha_dot*torch.cos(alpha)+1/2*self.model.Ibeta*beta_dot**2+self.model.sbeta*beta_dot*h_dot*torch.cos(beta)+self.model.Ialphabeta*beta_dot*alpha_dot*torch.cos(alpha-beta)
        Ep=1/2*self.model.kh*h**2+1/2*self.model.kalpha*alpha**2+1/2*self.model.kbeta*beta**2
        return Ec+Ep
    @staticmethod
    def make_trapz_integrator(N_step):
        W= torch.ones(N_step)
        W[0] = 0.5
        W[-1] = 0.5
        return W
    def objective_function(self,targeted_frequency :float =None):
        traj = rk4(self.model.ODE, self.model.t_span[0], self.X0, self.model.dt_computation, int((self.model.t_span[1] - self.model.t_span[0]) / self.model.dt_computation))
        
        h= torch.stack([y[3] for t, y in traj])  # Extract height from trajectory
        alpha = torch.stack([y[4] for t, y in traj])  # Extract alpha from trajectory
        beta = torch.stack([y[5] for t, y in traj])  # Extract beta from trajectory
        h_dot = torch.stack([y[0] for t, y in traj])  # Extract h_dot from trajectory
        alpha_dot = torch.stack([y[1] for t, y in traj])  # Extract alpha_dot from trajectory
        beta_dot = torch.stack([y[2] for t, y in traj])  # Extract beta_dot from trajectory
        # print(f"Height: {h}, Alpha: {alpha}, Beta: {beta}")
        energy_integrand = self.compute_energy(
            h=h,
            alpha=alpha,
            beta=beta,
            h_dot=h_dot,  # Extract h_dot from trajectory
            alpha_dot=alpha_dot,  # Extract alpha_dot from trajectory
            beta_dot=beta_dot   # Extract beta_dot from trajectory
        )
        ff_transform=torch.fft.rfft(h,n=len(h))
        # print(f"FFT Transform: {ff_transform}")
        #h_reconstructed = torch.fft.irfft(ff_transform, n=len(h)).real
        
        amplitudes= torch.abs(ff_transform)/len(h)*2
        fe=1/model.dt_computation
        frequencies  = torch.fft.rfftfreq(len(h), d=model.dt_computation)
        mask = (frequencies >= targeted_frequency/2) & (frequencies <= targeted_frequency*2)
        
        frequency_overlap=torch.trapz(amplitudes[mask], frequencies[mask])/ torch.trapz(amplitudes, frequencies)
        
        # print(f"Max Amplitude: {max_amplitude}, Max Frequency: {max_frequency}")
        # print(f"natural frequency {np.sqrt(model.kh/model.m)/2/np.pi}")
        
        
        # plt.figure(figsize=(4, 3),tight_layout=True)
        # plt.plot(frequencies.detach().numpy(), amplitudes.detach().numpy(), label='FFT Amplitudes', color='blue')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude')
        # plt.title('FFT of Height')
        # plt.legend()
        # plt.grid()
        # plt.savefig('height_fft_amplitudes.png')
        # print(max(h),min(h))
        
        
        # print(f"FFT {ff_transform}")
        # print(f"size FFT {ff_transform.shape}")
        # plt.figure(figsize=(4, 3),tight_layout=True)
        # plt.plot(h.detach().numpy(), label='Original h', color='blue')
        # plt.plot(h_reconstructed.detach().numpy(), label='Reconstructed h', color='orange', linestyle='--')
        # plt.xlabel('Time Step')
        # plt.ylabel('Height (m)')
        # plt.title('Height FFT comparison')
        # plt.legend()
        # plt.grid()
        # plt.savefig('height_fft.png')
        # d
        if False:
            W= self.make_trapz_integrator(len(energy_integrand))
            energy=torch.matmul(W.reshape(1,-1), energy_integrand.reshape(-1,1)).squeeze()* self.model.dt_computation 
        return frequency_overlap,traj 

if __name__=='__main__':
    config_env='config_env.yaml'
    model = Model(config_env)
    passive_control = PassiveControlAutodif(model)
    

    
    
    
    targeted_frequency = np.sqrt(model.kh/model.m)/2/np.pi*0.5  # Targeted frequency in Hz
    
    K_beta_min=0.01
    K_meta_max=model.kalpha
    
    xh_min=model.xf
    xh_max=model.c
    
    # Generate a list of parameter combinations for Kbeta and xh
    Kbeta_values = Kbeta_values = K_beta_min + (K_meta_max - K_beta_min) * torch.rand(10)
    xh_values =  xh_min + (xh_max - xh_max) * torch.rand(10)
    X_ini = [(Kbeta.item(), xh.item()) for Kbeta in Kbeta_values for xh in xh_values]
    print(len(X_ini))
    
    data_save=np.zeros((len(X_ini), 3,1000))  # Initialize data_save to store results
    j=0
    if True:
        
        for Kbeta, xh in X_ini:
            print(f"Initial Kbeta: {Kbeta}, xh: {xh}")
            
            
            
            
            Kbeta = torch.tensor(Kbeta, requires_grad=True)
            Dbeta = torch.tensor(Kbeta/1000, requires_grad=False)
            xh = torch.tensor(xh, requires_grad=True)
            optimizer = RMSprop([Kbeta, xh], lr=0.01)
            for epoch in range(10000):
                optimizer.zero_grad()
        
    
    
    
                passive_control.update_matrices_with_control_parameters(Kbeta, Dbeta, xh)
    
                # Compute the objective function
                max_amplitude,traj = passive_control.objective_function(targeted_frequency=targeted_frequency)
    
                loss = -max_amplitude
                # Perform backpropagation
                loss.backward()
                if epoch == 0:
                    first_grad_abs_K = Kbeta.grad.abs().item()
                    first_grad_abs_xh = xh.grad.abs().item()
    
            # Clip all gradients to be within [-first_grad_abs, first_grad_abs]
                with torch.no_grad():
                    Kbeta.grad.clamp_(-first_grad_abs_K, first_grad_abs_K)
                    xh.grad.clamp_(-first_grad_abs_xh, first_grad_abs_xh)
                optimizer.step()
                with torch.no_grad():
                    Kbeta.clamp_(min=0.001, max=model.kalpha)
                    xh.clamp_(min=model.xf, max=model.c)
                print(f'Epoch {epoch+1}, Max amplitudes: {loss.item()}, Kbeta: {Kbeta.item()}, xh: {xh.item()}')
                
                data_save[j, 0, epoch] = Kbeta.detach().numpy()
                data_save[j, 1, epoch] = xh.detach().numpy()
                data_save[j, 2, epoch] = loss.item()
            np.save('data_save.npy', data_save)
        j+=1
        #print(f"Gradients: Kbeta: {Kbeta.grad.item()}, xh: {xh.grad.item()}")
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    fig,ax=plt.subplots(1,1,tight_layout=True)
    ax.plot([t.item() for t, y in traj], [y[3].item() for t, y in traj], label='Trajectory',color='blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Trajectory of Height over Time')
    fig.savefig('h_trajectory.png')
    fig,ax=plt.subplots(1,1,tight_layout=True)
    ax.plot([t.item() for t, y in traj], [y[4].item() for t, y in traj], label='alpha',color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('alpha (rad)')
    ax.set_title('Trajectory of alpha over Time')
    fig.savefig('alpha_trajectory.png')
    fig,ax=plt.subplots(1,1,tight_layout=True)
    ax.plot([t.item() for t, y in traj], [y[5].item() for t, y in traj], label='Trajectory',color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Trajectory of beta over Time')
    fig.savefig('beta_trajectory.png')