import numpy as np
from scipy.spatial import KDTree

def gaussian(sigma, x):
    return np.exp(-x**2 / (2 * sigma**2))

def bilateral_interpolation(P, D_P, s_f, sigma_s, sigma_d):

    B = []
    D_B = []
    tree = KDTree(P)
    
    for i, p in enumerate(P):
        for k in np.arange(1/s_f, 1, 1/s_f):
            dist, idx = tree.query(p)
            q = P[idx]
            
            r_p, theta_p, phi_p = p
            _, _, phi_q = q
            b = (r_p, theta_p, k * phi_p + (1 - k) * phi_q)
            
            W_s = gaussian(sigma_s, np.linalg.norm(np.array(p) - np.array(q)))
            
            D_b_prime = W_s * D_P[i] + (1 - W_s) * D_P[idx]
            
            W_b_p = gaussian(sigma_s, np.linalg.norm(np.array(b) - np.array(p))) * gaussian(sigma_d, D_b_prime - D_P[i])
            
            D_b_val = W_b_p * D_P[i] + (1 - W_b_p) * D_P[idx]
            
            B.append(b)
            D_B.append(D_b_val)
    
    return B, D_B


