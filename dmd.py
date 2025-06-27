import torch

def dmd_torch(signal,dt,device='cpu',window_size=30):
    """
    X: tensor of shape (n, m) — n = dimension, m = time steps
    dt: time step between snapshots
    r: truncation rank (optional)
    """
    X=torch.zeros((window_size, signal.shape[1]-window_size), device=device)
    for i in range(window_size):
        X[i,:] = signal[:, i:signal.shape[1]-window_size+i]
    X = X.to(device)
    print(f"X shape: {X.shape}")
    r=100
    
    X1 = X[:, :-1]  # X
    X2 = X[:, 1:]   # X'
    print(X1.shape, X2.shape)
    
    
    print(X1.shape, X2.shape)

    # SVD
    U, S, Vh= torch.linalg.svd(X1, full_matrices=True)
    print(S)
    
    
    V=Vh.T
    r = min(r, U.shape[1])  # Ensure r does not exceed the rank of U

    U_r = U[:, :r]          # Truncate U to rank-r
    S_r = torch.diag(S[:r]) # Truncate S to rank-r and convert to diagonal matrix
    V_r = V[:, :r]          # Truncate V to rank-r
    print(f"SVD shapes: U={U_r.shape}, S={S_r.shape}, V={V_r.shape},X2={X2.shape}")
    S_inv = torch.linalg.inv(S_r) 
    Atilde = U_r.T @ X2 @ V_r@ S_inv 
    
    
    print(f"Atilde shape: {Atilde.shape}")
    eigvals, eigvecs = torch.linalg.eig(Atilde)
    
    print(f"Eigenvalues shape: {eigvals.shape}, Eigenvectors shape: {eigvecs.shape}",S_inv.shape)

    print(eigvecs)
    print(eigvals)

    X2 = X2.to(torch.complex64)
    V_r = V_r.to(torch.complex64)
    S_inv = S_inv.to(torch.complex64)
    Phi = X2 @ V_r @ S_inv @ eigvecs
    

    
    lambda_ = torch.diag(eigvals)
    
    omega = torch.log(lambda_) / dt
    
    x1 = X1[:, 0]
    # Solve for b (amplitudes of the DMD modes)
    x1 = x1.to(torch.complex64)
    print(Phi.shape, x1.shape)
    b, _ ,_,_= torch.linalg.lstsq(Phi, x1)
    mm1 = X1.shape[1]
    r = S_r.shape[0]
    t = torch.arange(0, mm1, device=device) * dt
    print(b.shape, omega.shape, t.shape)


    time_dynamics = torch.zeros((r, mm1), dtype=torch.complex64, device=device)  # Initialize time_dynamics

    for iter in range(mm1):
        print(b * torch.exp(omega * t[iter]))
        time_dynamics[:, iter] = b * torch.exp(omega * t[iter])
        
    
    Xdmd = Phi @ time_dynamics
    Xdmd = Xdmd.to(device)
    return Xdmd, Phi, omega, b