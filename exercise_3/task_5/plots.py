from functions import *
from scipy.integrate import solve_ivp

def plotSIR(SIM0,t_0, t_end, NT, mu0,mu1,beta,A,d,b,nu,rtol,atol):
    time = np.linspace(t_0,t_end,NT)
    sol = solve_ivp(model, t_span=[time[0],time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='LSODA', rtol=rtol, atol=atol)

    fig,ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].plot(sol.t, sol.y[0]-0*sol.y[0][0], label='1E0*susceptible');
    ax[0].plot(sol.t, 1e3*sol.y[1]-0*sol.y[1][0], label='1E3*infective');
    ax[0].plot(sol.t, 1e1*sol.y[2]-0*sol.y[2][0], label='1E1*removed');
    ax[0].set_xlim([0, 500])
    ax[0].legend();
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")

    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label='recovery rate')
    ax[1].plot(sol.t, 1e2*sol.y[1], label='1E2*infective');
    ax[1].set_xlim([0, 500])
    ax[1].legend();
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    I_h = np.linspace(-0.,0.05,100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b));
    ax[2].plot(I_h, 0*I_h, 'r:')
    #ax[2].set_ylim([-0.1,0.05])
    ax[2].set_title("Indicator function h(I)")
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")
    fig.savefig('figures/SIR_plots_b='+str(b)+'.png')
    fig.tight_layout()
    

def plot3D(SIM0):
    t_0 = 0
    t_end = 1000
    NT = t_end-t_0
    # if these error tolerances are set too high, the solution will be qualitatively (!) wrong
    rtol=1e-8
    atol=1e-8

    # SIR model parameters3
    beta=11.5
    A=20
    d=0.1
    nu=1
    b=0.01 # try to set this to 0.01, 0.020, ..., 0.022, ..., 0.03
    mu0 = 10   # minimum recovery rate
    mu1 = 10.45  # maximum recovery rate

    fig=plt.figure(figsize=(20,30))
    time = np.linspace(t_0,1500,NT)
    axs = []

    cmap = ["BuPu", "Purples", "bwr"][1]
    for i in range(9):
        axs.append(fig.add_subplot(7,3,i+1,projection="3d"))
        sol = solve_ivp(model, t_span=[time[0],time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='DOP853', rtol=rtol, atol=atol)
        axs[i].plot(sol.y[0], sol.y[1], sol.y[2], 'r-');
        axs[i].scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap='bwr');
        axs[i].set_xlabel("S")
        axs[i].set_ylabel("I")
        axs[i].set_zlabel("R")
        axs[i].set_title("SIR trajectory with b :"+str(round(b,3)),color="Red")
        b += 0.002  
    fig.tight_layout()