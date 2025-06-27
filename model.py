import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import logging


class Model:
    def __init__(self, config_file =None):
        if config_file is not None:
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            for key, value in config.items():
                
                if type(value) is float or type(value) is int:
                    setattr(self, key, torch.tensor(value))
                else:
                    setattr(self,key,value)
                
            self.t_span=(0,self.t_max)
            
        else:
            print('Initializing base parameters with default config')
            self.m=13.5
            self.xf=0.1
            self.c=0.25
            self.kh=2131.8346
            self.kalpha=198.9712
            self.rho=1.225
            self.damph=0.0
            self.dampalpha=0
            
            self.U=41.

        ###Initialize time integration variables 
            self.t_span=(0,10)
            self.f_actuation=50
            self.dt_computation=1/1000
            
        ## Initial conditions
            self.initial_condition_type='h' ##h or alpha or beta
            self.h0=[0.01,0.01]
            self.alpha0=[0.00,0.00]
            self.beta0=[0.00,0.00]
        self.salpha=self.m*(self.c/2-self.xf)
        self.Ialpha=self.m/3*(self.c**2-3*self.c*self.xf+3*self.xf**2)
        self.sbeta=self.m*(self.c-self.xh)**2/2/self.c
        self.Ibeta=self.m*(self.c-self.xh)**3/3/self.c
        self.Ialphabeta=self.Ibeta+(self.xh-self.xf)*self.sbeta
        self.t_actuation=1/self.f_actuation
        self.b=self.c/2
        self.a=self.xf/self.b-1

        self.psi1 = 0.165
        self.psi2 = 0.335  
        self.eps1=0.0455
        self.eps2=0.3     
    def update_control_parameters(self, Kbeta,xh,Dbeta):
        self.kbeta=Kbeta
        self.dampbeta=Dbeta
        self.xh=xh
        self.ch=self.xh/self.b-1     
        self.nu=torch.arccos(self.ch) 
        self.make_T_variables()


    def make_mass_matrix(self):
        s=self.m*(self.c/2-self.xf)
        Ialpha=self.m/3*(self.c**2-3*self.c*self.xf+3*self.xf**2)
        sbeta=self.m*(self.c-self.xh)**2/2/self.c
        Ibeta=self.m*(self.c-self.xh)**3/3/self.c
        Ialphabeta=Ibeta+(self.xh-self.xf)*sbeta
        A = torch.tensor([[self.m, s, sbeta],
                          [s, Ialpha, Ialphabeta],
                          [sbeta, Ialphabeta, Ibeta]])/self.span
                
        # A = torch.tensor([[self.m, s],
        #             [s, Ialpha]])/self.span
        return A
    def make_stiffness_matrix(self):
        E=torch.tensor([[self.kh,0,0],
                   [0,self.kalpha,0],
                   [0,0,self.kbeta]])/self.span
        # E=torch.tensor([[self.kh,0],
        #            [0,self.kalpha]])/self.span
        return E    
    def make_T_variables(self):
        self.T1=-1/3*torch.sqrt(1-self.ch**2)*(2+self.ch**2)+self.ch*self.nu
        self.T2=self.ch*(1-self.ch**2)-torch.sqrt(1-self.ch**2)*(1+self.ch**2)*self.nu+self.ch*self.nu**2
        self.T3=-(1/8+self.ch**2)*self.nu**2+1/4*self.ch*torch.sqrt(1-self.ch**2)*self.nu*(7+2*self.ch**2)-1/8*(1-self.ch**2)*(5*self.ch**2+4)
        self.T4=-self.nu+self.ch*torch.sqrt(1-self.ch**2)
        self.T5=-(1-self.ch**2)-self.nu**2+2*self.ch*torch.sqrt(1-self.ch**2)*self.nu
        self.T6=self.T2
        self.T7=-(1/8+self.ch**2)*self.nu+1/8*self.ch*torch.sqrt(1-self.ch**2)*(7+2*self.ch**2)
        self.T8=-1/3*torch.sqrt(1-self.ch**2)*(2*self.ch**2+1)+self.ch*self.nu
        self.T9=1/2*(1/3*(1-self.ch**2)**(1/3)+self.a*self.T4)
        self.T10=torch.sqrt(1-self.ch**2)+self.nu
        self.T11=self.nu*(1-2*self.ch)+torch.sqrt(1-self.ch**2)*(2-self.ch)
        self.T12=torch.sqrt(1-self.ch**2)*(2+self.ch)-self.nu*(2*self.ch+1)
        self.T13=1/2*(-self.T7-(self.ch-self.a)*self.T1)
        self.T14=1/16+1/2*self.a*self.ch
        
    def make_aero_mass_matrix(self):
        B=self.b**2*torch.tensor([[torch.pi,-torch.pi*self.a*self.b,-self.T1*self.b],
                   [-torch.pi*self.a*self.b,torch.pi*self.b**2*(1/8+self.a**2),-(self.T7+(self.ch-self.a)*self.T1)*self.b**2],
                   [-self.T1*self.b,2*self.T13*self.b**2,-self.T3*self.b**2/torch.pi]])
        # B=self.b**2*torch.tensor([[torch.pi,-torch.pi*self.a*self.b],
        #            [-torch.pi*self.a*self.b,torch.pi*self.b**2*(1/8+self.a**2)]])
        return B
            
    def make_aerodynamic_damping_matrix(self):
        phi0=1-self.psi1-self.psi2
        D1=self.b**2*torch.tensor([[0,torch.pi,-self.T4],
                   [0,torch.pi*(1/2-self.a)*self.b,(self.T1-self.T8-(self.ch-self.a)*self.T4+self.T11/2)*self.b],
                   [0,(-2*self.T9-self.T1+self.T4*(self.a-1/2))*self.b,-self.T4*self.T11*self.b/2/torch.pi]])##Waring /(2pi) ou /2*pi
        D2=torch.tensor([[2*torch.pi*self.b,2*torch.pi*self.b**2*(1/2-self.a),self.b**2*self.T11],
                   [-2*torch.pi*self.b**2*(self.a+1/2),-2*torch.pi*self.b**3*(self.a+1/2)*(1/2-self.a),-self.b**3*(self.a+1/2)*self.T11],
                   [self.b**2*self.T12,self.b**3*self.T12*(1/2-self.a),self.b**3*self.T12*self.T11/2/torch.pi]])##Waring /(2pi) ou /2*pi

        # D1=self.b**2*torch.tensor([[0,torch.pi],
        #            [0,torch.pi*(1/2-self.a)*self.b,]])
        # D2=torch.tensor([[2*torch.pi*self.b,2*torch.pi*self.b**2*(1/2-self.a)],
        #            [-2*torch.pi*self.b**2*(self.a+1/2),-2*torch.pi*self.b**3*(self.a+1/2)*(1/2-self.a)]])
        

        return D1+phi0*D2
    def make_aerodynamic_stiffness_matrix(self):
        epsilon=self.psi1*self.eps1/self.b+self.psi2*self.eps2/self.b
        phi0=1-self.psi1-self.psi2


        F1=self.b**2*torch.tensor([[0,0,0],
                   [0,0,(self.T4+self.T10)],
                   [0,0,(self.T5-self.T4*self.T10)/torch.pi]])
        F2=torch.tensor([[0,2*torch.pi*self.b,2*self.b*self.T10],
                   [0,-2*torch.pi*self.b**2*(self.a+1/2),-2*self.b**2*(self.a+1/2)*self.T10],
                   [0,self.b**2*self.T12,self.b**2*self.T12*self.T10/torch.pi]])
        F3=torch.tensor([[2*torch.pi*self.b,2*torch.pi*self.b**2*(1/2-self.a),self.b**2*self.T11],
                   [-2*torch.pi*self.b**2*(self.a+1/2),-2*torch.pi*self.b**3*(self.a+1/2)*(1/2-self.a),-self.b**3*(self.a+1/2)*self.T11],
                   [self.b**2*self.T12,self.b**3*self.T12*(1/2-self.a),self.b**3*self.T12*self.T11/2/torch.pi]])##Waring /(2pi) ou /2*pi

        # F1=self.b**2*torch.tensor([[0,0],
        #            [0,0]])
        # F2=torch.tensor([[0,2*torch.pi*self.b],
        #            [0,-2*torch.pi*self.b**2*(self.a+1/2)]])
        # F3=torch.tensor([[2*torch.pi*self.b,2*torch.pi*self.b**2*(1/2-self.a)],
        #            [-2*torch.pi*self.b**2*(self.a+1/2),-2*torch.pi*self.b**3*(self.a+1/2)*(1/2-self.a)]])
        return F1+phi0*F2+epsilon*F3
    def make_aerodynamic_influence_matrices(self):
        W0=torch.tensor([[-self.psi1*(self.eps1/self.b)**2],
                   [-self.psi2*(self.eps2/self.b)**2],
                   [self.psi1*self.eps1*(1-self.eps1*(1/2-self.a))/self.b],
                   [self.psi2*self.eps2*(1-self.eps2*(1/2-self.a))/self.b],
                   [self.psi1*self.eps1*(self.T10-self.eps1*self.T11/2)/torch.pi/self.b],##Warning /(pib) ou /pi+b
                   [self.psi2*self.eps2*(self.T10-self.eps2*self.T11/2)/torch.pi/self.b]])
        # W0=torch.tensor([[-self.psi1*(self.eps1/self.b)**2],
        #            [-self.psi2*(self.eps2/self.b)**2],
        #            [self.psi1*self.eps1*(1-self.eps1*(1/2-self.a))/self.b],
        #            [self.psi2*self.eps2*(1-self.eps2*(1/2-self.a))/self.b]])
        W=torch.hstack((2*torch.pi*self.b*W0,-2*torch.pi*self.b**2*(self.a+1/2)*W0,self.b**2*self.T12*W0)).T
        # W=torch.hstack((2*torch.pi*self.b*W0,-2*torch.pi*self.b**2*(self.a+1/2)*W0)).T
        W1=torch.tensor([[1,0,0],
                     [1,0,0],
                     [0,1,0],
                     [0,1,0],
                     [0,0,1],
                     [0,0,1]])
        # W1=torch.tensor([[1,0],
        #              [1,0],
        #              [0,1],
        #              [0,1]])
        W2=torch.tensor([[-self.eps1/self.b,0,0,0,0,0],
                     [0,-self.eps2/self.b,0,0,0,0],
                     [0,0,-self.eps1/self.b,0,0,0],
                     [0,0,0,-self.eps2/self.b,0,0],
                     [0,0,0,0,-self.eps1/self.b,0],
                     [0,0,0,0,0,-self.eps2/self.b]])
        # W2=torch.tensor([[-self.eps1/self.b,0,0,0],
        #              [0,-self.eps2/self.b,0,0,],
        #              [0,0,-self.eps1/self.b,0],
        #              [0,0,0,-self.eps2/self.b]])
        return W,W1,W2

    def make_first_order_matrix(self):
        A=self.make_mass_matrix()
        B=self.make_aero_mass_matrix()
        M=A+self.rho*B
        M_inv=torch.linalg.inv(M)
        
        E=self.make_stiffness_matrix()
       
        F=self.make_aerodynamic_stiffness_matrix()
        D=self.make_aerodynamic_damping_matrix()
        C=E/1000
        W,W1,W2=self.make_aerodynamic_influence_matrices()
        Intermediate=-self.rho*self.U**3*torch.matmul(M_inv,W)

        l1=torch.hstack((-torch.matmul(M_inv,C+self.rho*self.U*D),-torch.matmul(M_inv,E+self.rho*self.U**2*F),Intermediate[:,:3],Intermediate[:,3:]))
        l2=torch.hstack((torch.eye(3),torch.zeros((3,3)),torch.zeros((3,3)),torch.zeros((3,3))))
        l3=torch.hstack((torch.zeros((3,3)),W1[:3,:],self.U*W2[:3,:3],self.U*W2[:3,3:]))
        l4=torch.hstack((torch.zeros((3,3)),W1[3:,:],self.U*W2[3:,:3],self.U*W2[3:,3:]))
        
        # l1=torch.hstack((-torch.matmul(M_inv,C+self.rho*self.U*D),-torch.matmul(M_inv,E+self.rho*self.U**2*F),Intermediate[:,:2],Intermediate[:,2:]))
        # l2=torch.hstack((torch.eye(2),torch.zeros((2,2)),torch.zeros((2,2)),torch.zeros((2,2))))
        # l3=torch.hstack((torch.zeros((2,2)),W1[:2,:],self.U*W2[:2,:2],self.U*W2[:2,2:]))
        # l4=torch.hstack((torch.zeros((2,2)),W1[2:,:],self.U*W2[2:,:2],self.U*W2[2:,2:]))
        Q=torch.vstack((l1,l2,l3,l4))
        self.Q=Q

        return Q

    def set_initial_conditions(self):
        if self.initial_condition_type=='h':
            h0=torch.empty(1).uniform_(self.h0[0],self.h0[1])
            alpha0=0
            beta0=0
        elif self.initial_condition_type=='alpha':
            alpha0=torch.empty(1).uniform_(self.alpha0[0],self.alpha0[1])
            h0=0
            beta0=0
        elif self.initial_condition_type=='beta':
            beta0=torch.empty(1).uniform_(self.beta0[0],self.beta0[1])
            h0=0
            alpha0=0
        else:
            raise ValueError('Initial condition type not recognized')
        
        X0 = torch.tensor([0,0,0,h0,alpha0,beta0,0,0,0,0,0,0])
        #X0 = torch.tensor([0,0,h0,alpha0,0,0,0,0])
        # self.X_values=torch.te
        # nsor([X0])
        # self.t_values=[0]
        return X0
   
    def ODE(self,t,X):
        """Make the ODE that will be solved. Warning the Q matrix need to be computed before, the unsteady source term and the control function need to be initialized.

        Args:
            t (float): time variable
            X (array): State Variable

        Returns:
            fun: dynamical function
        """
        Nl_vector=torch.zeros(3)
        # print('Nl_vector',Nl_vector[2], 'X[5]', X[5])
        Nl_vector[2]= self.kbeta/self.span*X[5]**3
        
        A=self.make_mass_matrix()
        B=self.make_aero_mass_matrix()
        M=A+self.rho*B
        M_inv=torch.linalg.inv(M)
        Nl_vector=torch.matmul(-M_inv, Nl_vector)
        
        Nl_term=torch.zeros(12)
        Nl_term[:3]=Nl_vector
        

        return torch.matmul(self.Q, X)+ Nl_term
            
        
   