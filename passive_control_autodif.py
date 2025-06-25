import torch    
import torch.nn as nn
import torch.nn.functional as F
from rk4 import rk4
from model import Model
import matplotlib.pyplot as plt

class PassiveControlAutodif(nn.Module):
    def __init__(self, model: Model):
        super(PassiveControlAutodif, self).__init__()
        self.model = model
        self.X0=self.model.set_initial_conditions()
    def update_matrices_with_control_parameters(self, Kbeta: torch.Tensor, Dbeta: torch.Tensor, xh: torch.Tensor):
        self.model.update_control_parameters(Kbeta=Kbeta, xh=xh,Dbeta=Dbeta)
        self.Q=self.model.make_first_order_matrix()
    
    
    
    def objective_function(self):
        traj = rk4(self.model.ODE, self.model.t_span[0], self.X0, self.model.dt_computation, int((self.model.t_span[1] - self.model.t_span[0]) / self.model.dt_computation))
        return traj

if __name__=='__main__':
    config_env='config_env.yaml'
    model = Model(config_env)
    passive_control = PassiveControlAutodif(model)
    
    # Example usage
    Kbeta = torch.tensor(0.4, requires_grad=True)
    Dbeta = torch.tensor(0.0, requires_grad=True)
    xh = torch.tensor(0.1125, requires_grad=True)
    
    passive_control.update_matrices_with_control_parameters(Kbeta, Dbeta, xh)
    
    # Compute the objective function
    traj = passive_control.objective_function()
    print(traj)  # This will print the trajectory computed by rk4
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