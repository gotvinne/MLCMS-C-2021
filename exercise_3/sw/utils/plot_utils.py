
import matplotlib.pyplot as plt

def generate_field_vectors(NI,NJ,X1,X2,u,v,A,func):
    """Generate vector fields u and v for positions X1 and X2. 
    Args:
        NI (int): x dimension of grid
        NJ (int): y dimension of grid
        X1 (list): list of x positions
        X2 (list): list of y positions
        u (list): empty list of x field components
        v (list): empty list of y field components
        A (np.array): Linear vector field
        func (np.array): differential equation
    """
    for i in range(NI):
        for j in range(NJ):
            x, y = X1[i, j], X2[i, j]
            vec = func(None, [x, y],A)
            u[i, j], v[i, j] = vec[0], vec[1]

def configuration(title,xlabel,ylabel,xlim,ylim,X1=None,X2=None,u=None,v=None):
    """Defines the spesifics of a plotting
    Args:
        title (string): [description]
        xlabel (string): [description]
        ylabel (string): [description]
        xlim (int): [description]
        ylim (int): [description]

        For stream plots: 
        X1 (list): list of x positions
        X2 (list): list of y positions
        u (list): empty list of x field components
        v (list): empty list of y field components
    """
    if X1 is not None and X2 is not None and u is not None and v is not None:
        plt.quiver(X1, X2, u, v)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])

def add_orbit(flow,plt_str):
    """Adds a flow to a matplot_lib plot

    Args:
        flow (Bunch object, scipy.ivp_integrate()): Object holding the solution of an ODE
        plt_str (string): Defines the visualization of the orbit
    """
    plt.plot(flow.y[0], flow.y[1], plt_str) 
    plt.plot(flow.y[0][0], flow.y[1][0], 'o') 

def orbit_plot(x, t, r, x0):
    """Visualize an orbit for a first order system using matplot_lib
    Args:
        x (list): List of orbit values
        t (list): List of corresponding time stamps
        r (int): Bifurcation variable
        x0 (int): Intitial point
    """
    plt.figure()
    plt.title("r = " + str(r) + ", x0 = " + str(x0))
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.plot(t, x, "-b") 

def bifurcation_diagram(r, x, x0):
    """Plots a first order bifurcation diagram given parameters
    Args:
        r (list): list of bifurcation variables 
        x (list): list of system values
        x0 (int): initial value
    """
    plt.figure(figsize=(5, 3), dpi=200)
    plt.title("Bifurcation diagram, x0 = " + str(x0))
    plt.xlabel('$r$')
    plt.ylabel('$x$')
    plt.plot(r, x, ",", color="k") 
    plt.show()

def three_d_orbit(x,y,z,lw=0.1):
    """Plot an orbit defined in R^3
    Args:
        x (list): List of x values
        y (list): List of y values
        z (list): List of z values
        lw (float): line width
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, 'g-', lw=lw)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
