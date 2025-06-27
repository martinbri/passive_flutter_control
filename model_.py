import numpy as np
import matplotlib.pyplot as plt
import yaml
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig




class Model:
    def __init__(self, config_file =None):
        if config_file is not None:

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(self,key,value)
            self.t_span=(0,self.t_max)

        else:
            print('Initializing base parameters with default config')
            self.m=3
            self.xf=0.0375
            self.c=0.15
            self.xh=0.1125
            self.kh=200
            self.kalpha=4
            self.kbeta=0.4
            self.rho=1.225
            self.damph=0
            self.dampalpha=0
            self.dampbeta=0
            self.U=12.
            self.span=0.45

        ###Initialize time integration variables
            self.t_span=(0,10)
            self.f_actuation=50
            self.dt_computation=0.005

        ## Initial conditions
            self.initial_condition_type='h' ##h or alpha or beta
            self.h0=[0.03,0.03]
            self.alpha0=[0.00,0.00]
            self.beta0=[0.00,0.00]


        self.t_actuation=1/self.f_actuation
        self.b=self.c/2
        self.a=self.xf/self.b-1
        self.ch=self.xh/self.b-1
        self.nu=np.arccos(self.ch)
        self.psi1 = 0.165
        self.psi2 = 0.335
        self.eps1=0.0455
        self.eps2=0.3

        self.make_T_variables()

    def make_mass_matrix(self):
        s=self.m*(self.c/2-self.xf)
        Ialpha=self.m/3*(self.c**2-3*self.c*self.xf+3*self.xf**2)
        sbeta=self.m*(self.c-self.xh)**2/2/self.c
        Ibeta=self.m*(self.c-self.xh)**3/3/self.c
        Ialphabeta=Ibeta+(self.xh-self.xf)*sbeta

        A=np.array([[self.m,s,sbeta],
                   [s,Ialpha,Ialphabeta],
                   [sbeta,Ialphabeta,Ibeta]])/self.span
        return A
    def make_stiffness_matrix(self):
        E=np.array([[self.kh,0,0],
                   [0,self.kalpha,0],
                   [0,0,self.kbeta]])/self.span
        return E
    def make_T_variables(self):
        self.T1=-1/3*np.sqrt(1-self.ch**2)*(2+self.ch**2)+self.ch*self.nu
        self.T2=self.ch*(1-self.ch**2)-np.sqrt(1-self.ch**2)*(1+self.ch**2)*self.nu+self.ch*self.nu**2
        self.T3=-(1/8+self.ch**2)*self.nu**2+1/4*self.ch*np.sqrt(1-self.ch**2)*self.nu*(7+2*self.ch**2)-1/8*(1-self.ch**2)*(5*self.ch**2+4)
        self.T4=-self.nu+self.ch*np.sqrt(1-self.ch**2)
        self.T5=-(1-self.ch**2)-self.nu**2+2*self.ch*np.sqrt(1-self.ch**2)*self.nu
        self.T6=self.T2
        self.T7=-(1/8+self.ch**2)*self.nu+1/8*self.ch*np.sqrt(1-self.ch**2)*(7+2*self.ch**2)
        self.T8=-1/3*np.sqrt(1-self.ch**2)*(2*self.ch**2+1)+self.ch*self.nu
        self.T9=1/2*(1/3*(1-self.ch**2)**(1/3)+self.a*self.T4)
        self.T10=np.sqrt(1-self.ch**2)+self.nu
        self.T11=self.nu*(1-2*self.ch)+np.sqrt(1-self.ch**2)*(2-self.ch)
        self.T12=np.sqrt(1-self.ch**2)*(2+self.ch)-self.nu*(2*self.ch+1)
        self.T13=1/2*(-self.T7-(self.ch-self.a)*self.T1)
        self.T14=1/16+1/2*self.a*self.ch

    def make_aero_mass_matrix(self):
        B=self.b**2*np.array([[np.pi,-np.pi*self.a*self.b,-self.T1*self.b],
                   [-np.pi*self.a*self.b,np.pi*self.b**2*(1/8+self.a**2),-(self.T7+(self.ch-self.a)*self.T1)*self.b**2],
                   [-self.T1*self.b,2*self.T13*self.b**2,-self.T3*self.b**2/np.pi]])
        return B

    def make_aerodynamic_damping_matrix(self):
        phi0=1-self.psi1-self.psi2
        D1=self.b**2*np.array([[0,np.pi,-self.T4],
                   [0,np.pi*(1/2-self.a)*self.b,(self.T1-self.T8-(self.ch-self.a)*self.T4+self.T11/2)*self.b],
                   [0,(-2*self.T9-self.T1+self.T4*(self.a-1/2))*self.b,-self.T4*self.T11*self.b/2/np.pi]])##Waring /(2pi) ou /2*pi
        D2=np.array([[2*np.pi*self.b,2*np.pi*self.b**2*(1/2-self.a),self.b**2*self.T11],
                   [-2*np.pi*self.b**2*(self.a+1/2),-2*np.pi*self.b**3*(self.a+1/2)*(1/2-self.a),-self.b**3*(self.a+1/2)*self.T11],
                   [self.b**2*self.T12,self.b**3*self.T12*(1/2-self.a),self.b**3*self.T12*self.T11/2/np.pi]])##Waring /(2pi) ou /2*pi
        return D1+phi0*D2
    def make_aerodynamic_stiffness_matrix(self):
        epsilon=self.psi1*self.eps1/self.b+self.psi2*self.eps2/self.b
        phi0=1-self.psi1-self.psi2


        F1=self.b**2*np.array([[0,0,0],
                   [0,0,(self.T4+self.T10)],
                   [0,0,(self.T5-self.T4*self.T10)/np.pi]])
        F2=np.array([[0,2*np.pi*self.b,2*self.b*self.T10],
                   [0,-2*np.pi*self.b**2*(self.a+1/2),-2*self.b**2*(self.a+1/2)*self.T10],
                   [0,self.b**2*self.T12,self.b**2*self.T12*self.T10/np.pi]])
        F3=np.array([[2*np.pi*self.b,2*np.pi*self.b**2*(1/2-self.a),self.b**2*self.T11],
                   [-2*np.pi*self.b**2*(self.a+1/2),-2*np.pi*self.b**3*(self.a+1/2)*(1/2-self.a),-self.b**3*(self.a+1/2)*self.T11],
                   [self.b**2*self.T12,self.b**3*self.T12*(1/2-self.a),self.b**3*self.T12*self.T11/2/np.pi]])##Waring /(2pi) ou /2*pi

        return F1+phi0*F2+epsilon*F3
    def make_aerodynamic_influence_matrices(self):
        W0=np.array([[-self.psi1*(self.eps1/self.b)**2],
                   [-self.psi2*(self.eps2/self.b)**2],
                   [self.psi1*self.eps1*(1-self.eps1*(1/2-self.a))/self.b],
                   [self.psi2*self.eps2*(1-self.eps2*(1/2-self.a))/self.b],
                   [self.psi1*self.eps1*(self.T10-self.eps1*self.T11/2)/np.pi/self.b],##Warning /(pib) ou /pi+b
                   [self.psi2*self.eps2*(self.T10-self.eps2*self.T11/2)/np.pi/self.b]])
        W=np.hstack((2*np.pi*self.b*W0,-2*np.pi*self.b**2*(self.a+1/2)*W0,self.b**2*self.T12*W0)).T
        W1=np.array([[1,0,0],
                     [1,0,0],
                     [0,1,0],
                     [0,1,0],
                     [0,0,1],
                     [0,0,1]])
        W2=np.array([[-self.eps1/self.b,0,0,0,0,0],
                     [0,-self.eps2/self.b,0,0,0,0],
                     [0,0,-self.eps1/self.b,0,0,0],
                     [0,0,0,-self.eps2/self.b,0,0],
                     [0,0,0,0,-self.eps1/self.b,0],
                     [0,0,0,0,0,-self.eps2/self.b]])
        return W,W1,W2
    def compute_initial_excitation(self,alpha0,h0,beta0):
        g=self.b*(h0+self.b*(1/2-self.a)*alpha0+self.b*self.T11/2/np.pi*beta0)*np.array([[2*np.pi],
                                                                                         [-2*np.pi*self.b*(self.a+1/2)],
                                                                                         [self.b*self.T12]])

        return g
    def Wagner_function_derivative(self,t):
        return self.psi1*self.eps1*self.U/self.b*np.exp(-self.eps1*self.U*t/self.b)+self.psi2*self.eps2*self.U/self.b*np.exp(-self.eps2*self.U*t/self.b)

    #def make_structural_damping_matrix(self):
    #    C=np.array([[self.damph,0,0],
    #                [0,self.dampalpha,0],
    #                [0,0,self.dampbeta]])
    #    return C
    def make_first_order_matrix(self):
        A=self.make_mass_matrix()
        B=self.make_aero_mass_matrix()
        M=A+self.rho*B
        M_inv=np.linalg.inv(M)

        E=self.make_stiffness_matrix()

        F=self.make_aerodynamic_stiffness_matrix()
        D=self.make_aerodynamic_damping_matrix()
        C=E/1000
        W,W1,W2=self.make_aerodynamic_influence_matrices()
        Intermediate=-self.rho*self.U**3*np.dot(M_inv,W)

        l1=np.hstack((-np.dot(M_inv,C+self.rho*self.U*D),-np.dot(M_inv,E+self.rho*self.U**2*F),Intermediate[:,:3],Intermediate[:,3:]))
        l2=np.hstack((np.eye(3),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))))
        l3=np.hstack((np.zeros((3,3)),W1[:3,:],self.U*W2[:3,:3],self.U*W2[:3,3:]))
        l4=np.hstack((np.zeros((3,3)),W1[3:,:],self.U*W2[3:,:3],self.U*W2[3:,3:]))
        Q=np.vstack((l1,l2,l3,l4))
        self.Q=Q
        return Q
    def make_unsteady_source(self, unsteady,alpha0,h0,beta0):
        """Compute the unsteady Source. zero if unsteady False or accordoing to the book id unsteady true

        Args:
            unsteady (Bool): Enable/disable the unsteady source
            alpha0 (float): Initial pitch
            h0 (float): initial plunge
            beta0 (float): initial beta

        Out: Ndarray
        """
        if not unsteady:
            self.source_t = lambda t: np.zeros((12))
        else:
            A=self.make_mass_matrix()
            B=self.make_aero_mass_matrix()
            M=A+self.rho*B
            M_inv=np.linalg.inv(M)
            g=self.compute_initial_excitation(alpha0,h0,beta0)
            source = self.rho*self.U*np.dot(M_inv,g)


            source=np.vstack((source,np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1)))).reshape(12)
            self.source_t=lambda t: source*self.Wagner_function_derivative(t)

    def set_initial_conditions(self):
        if self.initial_condition_type=='h':
            h0=np.random.uniform(self.h0[0],self.h0[1])
            alpha0=0
            beta0=0
        elif self.initial_condition_type=='alpha':
            alpha0=np.random.uniform(self.alpha0[0],self.alpha0[1])
            h0=0
            beta0=0
        elif self.initial_condition_type=='beta':
            beta0=np.random.uniform(self.beta0[0],self.beta0[1])
            h0=0
            alpha0=0
        else:
            raise ValueError('Initial condition type not recognized')

        X0 = np.array([0,0,0,h0,alpha0,beta0,0,0,0,0,0,0])
        self.X_values=np.array([X0])
        self.t_values=[0]
        return X0

    def ODE(self,t,X,action):
        """Make the ODE that will be solved. Warning the Q matrix need to be computed before, the unsteady source term and the control function need to be initialized.

        Args:
            t (float): time variable
            X (array): State Variable

        Returns:
            fun: dynamical function
        """

        return np.dot(self.Q, X)+self.source_t(t)+np.array([0,0,action,0,0,0,0,0,0,0,0,0])
    def one_ODE_resolution_step(self,t_span,X,action):
        """Solve the system over the wanted time lapse tspan.

        Args:
            t_span (tupple): min and max time required for the solution
            X (array): current state

        Returns:
            float,array: updated time and state
        """
        t_span=tuple(round(x, 4) for x in t_span) ##round t span to avoid conflict with t_eval

        t_eval=np.arange(t_span[0],t_span[1],self.dt_computation)



        solution = solve_ivp(self.ODE, (t_span), X, args=(action,), method='RK45',t_eval=t_eval[1:-1])

        self.t_values.append(solution.t[-1])
        self.X_values=np.vstack((self.X_values,solution.y[:, -1]))  # Stocke la dernière valeur obtenue
        return solution.t[-1],solution.y[:, -1]




if __name__ == "__main__":



    model=Model()
    Q=model.make_first_order_matrix()
    X0=model.set_initial_conditions()
    print(X0[4])

    model.make_unsteady_source(unsteady=True,h0=X0[3],alpha0=X0[4],beta0=X0[5])

    t_span = (0, 10)  # De t=0 à t=10
    t_eval = np.linspace(0, 10, 10000)  # Points de calcul

    # Résolution du système

    # Boucle de résolution de l'EDO par sous-intervalles
    X=X0
    # for (i,t) in enumerate(t_eval[:-1]):
    #     time,X=model.one_ODE_resolution_step(t_span=(t,t_eval[i+1]),X=X,action=0)

    # plt.plot(model.t_values, model.X_values[:,3], 'r-.',label="h(t)")
    # plt.plot(model.t_values, model.X_values[:,4], 'b-.',label="alpha(t)")
    # plt.plot(model.t_values, model.X_values[:,5], 'g-.',label="beta(t)")


    # Fonction qui définit le système d'EDO
    def systeme(t, X):
        return np.dot(Q, X)  # Multiplication matricielle

    eigenvalues = np.linalg.eigvals(Q)
    print("Eigenvalues:", eigenvalues)





    # Conditions initiales X(0)
    X0 = np.array([0,0,0,0.03,0,0,0,0,0,0,0,0])

    # Intervalle de temps
    t_span = (0, 10)  # De t=0 à t=10
    t_eval = np.linspace(0, 10, 10000)  # Points de calcul

    # Résolution du système
    t_values = [0]  # Stockage du temps
    X_values = np.array([X0])  # Stockage des états

    # Boucle de résolution de l'EDO par sous-intervalles
    X=X0
    for (i,t) in enumerate(t_eval[:-1]):

    # Résolution de l'EDO sur [t_current, t_next]
        solution = solve_ivp(systeme, (t, t_eval[i+1]), X, method='RK45')

        # Mise à jour des valeurs
        t_values.append(solution.t[0])
        X_values=np.vstack((X_values,solution.y[:, -1]))  # Stocke la dernière valeur obtenue

        # Mise à jour des conditions initiales pour le prochain intervalle
        X = solution.y[:, -1]
    fig,ax=plt.subplots(1,1,tight_layout=True)
    ax.plot(t_values,X_values[:,3], label='Trajectory',color='blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Trajectory of Height over Time')
    fig.savefig('h_trajectory_dimitriadis.png')
    fig,ax=plt.subplots(1,1,tight_layout=True)
    ax.plot(t_values,X_values[:,4], label='alpha',color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('alpha (rad)')
    ax.set_title('Trajectory of alpha over Time')
    fig.savefig('alpha_trajectory_dimitriadis.png')
    fig,ax=plt.subplots(1,1,tight_layout=True)
    ax.plot(t_values,X_values[:,5], label='Trajectory',color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Trajectory of beta over Time')
    fig.savefig('beta_trajectory_dimitriadis.png')
    #sol = solve_ivp(systeme, t_span, X0, t_eval=t_eval, method='RK45')

    # Affichage des solutions
    import matplotlib.pyplot as plt

    # plt.plot(sol.t, sol.y[3], 'r--',label="h(t)")
    # plt.plot(sol.t, sol.y[4], 'b--',label="alpha(t)")
    # plt.plot(sol.t, sol.y[5], 'g--',label="beta(t)")

    # plt.xlabel("Temps t")
    # plt.ylabel("Valeurs de X(t)")
    plt.legend()
    plt.grid()
    plt.show()
