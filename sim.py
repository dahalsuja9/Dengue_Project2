import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PARAMETERS ---
N_h, N_v = 10000.0, 30000.0
sigma_h, sigma_v = 1.0/5.0, 1.0/10.0
gamma, mu_v = 1.0/7.0, 1.0/14.0
birth_v = mu_v * N_v
beta_h, beta_v = 0.00005, 0.00003

# --- CONTROL FUNCTION ---
def get_controls(t):
    if t < 50: return 0.0, 0.0 
    else: return 0.1, 0.1 # Drugs + Spray

# --- EQUATIONS ---
def deriv(y, t, N_h, beta_h, beta_v, sigma_h, sigma_v, gamma, mu_v, birth_v):
    Sh, Eh, Ih, Rh, Sv, Ev, Iv = y
    drug, spray = get_controls(t)
    eff_gamma = gamma + drug
    eff_mu_v = mu_v + spray
    
    dSh = -beta_h * Sh * Iv
    dEh = (beta_h * Sh * Iv) - (sigma_h * Eh)
    dIh = (sigma_h * Eh) - (eff_gamma * Ih)
    dRh = (eff_gamma * Ih)
    dSv = birth_v - (beta_v * Sv * Ih) - (eff_mu_v * Sv)
    dEv = (beta_v * Sv * Ih) - (sigma_v * Ev) - (eff_mu_v * Ev)
    dIv = (sigma_v * Ev) - (eff_mu_v * Iv)
    return dSh, dEh, dIh, dRh, dSv, dEv, dIv

# --- RUN ---
y0 = (N_h-1, 1, 0, 0, N_v, 0, 0)
t = np.linspace(0, 150, 150)
ret = odeint(deriv, y0, t, args=(N_h, beta_h, beta_v, sigma_h, sigma_v, gamma, mu_v, birth_v))
Ih = ret.T[2] # Infected Humans

# --- PLOT & SAVE ---
plt.figure(figsize=(10,6))
plt.plot(t, Ih, 'r-', linewidth=2, label='Infected Humans')
plt.axvline(x=50, color='b', linestyle='--', label='Intervention Starts')
plt.title('Impact of Control Measures on Dengue Dynamics')
plt.xlabel('Time (Days)')
plt.ylabel('Infected Population')
plt.legend()
plt.grid(True)

# THIS IS THE IMPORTANT PART:
plt.savefig('dengue_graph.png') 
print("Graph saved as 'dengue_graph.png'!")
