import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack, vstack, identity
from scipy.sparse.linalg import spsolve, eigsh

class Schrodinger:

    def __init__(self, amplitude=0.05, frequency=1.1, delta=0.8, Nodes=10000, xmax=250, xmin=-250, tmax=10, dt=0.1):
        ### Domain parameters ###
        self.Nodes = Nodes # Number of nodes
        self.xmin = xmin
        self.xmax = xmax

        ### Time parameters ###
        self.tmax = tmax
        self.dt = dt # Timestep

        ### Forcing parameters ###
        self.A = amplitude # Forcing amplitude
        self.w = frequency # Forcing angular frequency

        ### Delta ###
        self.delta = delta

        self.init_domain()
        self.init_vectors()
        self.init_params()
        self.finite_diff_matrix()


    ### Helper Functions ###
    def norm(self, v):
        return np.sqrt(abs(np.vdot(v,v)))


    ### Problem-Specific Functions ###
    def domain_wall(self, single_cell=False):
        x = self.x
        if single_cell == True:
            x = np.arange(0, 1, self.dx)
        return np.tanh(self.delta * x)

    def edge_state(self, single_cell=False):
        x = self.x
        if single_cell == True:
            x = np.arange(0, 1, self.dx)
            return 1/np.cosh(self.delta * x)
        n, m = 0, self.lenx
        if self.xmax > 600:
            n, m = np.where(self.x >= -600)[0][0], np.where(self.x >= 600)[0][0]
        vec = np.zeros(self.lenx)
        vec[n:m] = 1/np.cosh(self.x[n:m])**(self.theta_sharp/self.lambda_sharp)
        return vec

    def forcing(self, t):
        return self.A * np.cos(self.w * self.delta * t)

    def V(self, single_cell=False):
        x = self.x
        if single_cell == True:
            x = np.arange(0, 1, self.dx)
        return np.cos(4 * np.pi * x)

    def W(self, single_cell=False):
        x = self.x
        if single_cell == True:
            x = np.arange(0, 1, self.dx)
        return np.cos(2 * np.pi * x)


    ### Main Setup ###
    def init_domain(self):
        self.dx = (self.xmax - self.xmin) / self.Nodes
        #self.x = np.arange(self.xmin, self.xmax, self.dx)
        self.x = np.arange(self.xmin - self.dx, self.xmax + 2*self.dx, self.dx)
        self.lenx = len(self.x)


    def init_vectors(self):
        eigvals, eigvecs = self.edge_mode()
        Psi0 = eigvecs[:, 0]
        self.Psi = Psi0.copy() / np.amax(np.abs(Psi0))
        self.Psitp1 = self.Psi.copy()


    def init_params(self):
        self.nsteps = round(self.tmax / self.dt)
        self.id = identity(self.lenx, format="csr")


    def finite_diff_matrix(self):
        # Periodic BCs
        nums_ii = self.V() + self.delta * self.domain_wall() * self.W() + (2/self.dx**2) * np.ones(self.lenx)
        nums_ij = (-1/self.dx**2) * np.ones(self.lenx-1)
        data = np.concatenate((nums_ii, nums_ij, nums_ij, [-1/self.dx**2, -1/self.dx**2]))
        row = np.concatenate((np.arange(0, self.lenx, 1), np.arange(0, self.lenx-1, 1), np.arange(1, self.lenx, 1), [0, self.lenx-1]))
        col = np.concatenate((np.arange(0, self.lenx, 1), np.arange(1, self.lenx, 1), np.arange(0, self.lenx-1, 1), [self.lenx-1, 0]))
        self.D = coo_matrix((data, (row, col)), shape=(self.lenx, self.lenx)).tocsr()


    def forcing_matrix(self, t):
        alpha = 1j * self.delta * self.forcing(t) / self.dx
        nums = alpha * np.ones(self.lenx-1)
        data = np.concatenate((nums, -1*nums, [-1*alpha, alpha]))
        row = np.concatenate((np.arange(0, self.lenx-1, 1), np.arange(1, self.lenx, 1), [0, self.lenx-1]))
        col = np.concatenate((np.arange(1, self.lenx, 1), np.arange(0, self.lenx-1, 1), [self.lenx-1, 0]))
        return coo_matrix((data, (row, col)), shape=(self.lenx, self.lenx)).tocsr()


    def construct_matrix(self, k, single_cell=False, edge=False):
        if edge == True:
            nums_ii = k**2 + self.V(single_cell) + self.delta * self.domain_wall(single_cell) * self.W(single_cell) + (2/self.dx**2) * np.ones(self.lenx)
        else:
            nums_ii = k**2 + self.V(single_cell) + (2/self.dx**2) * np.ones(n)
        n = len(nums_ii)
        nums_ij_up = ((-1 / self.dx**2) - (1j * k / self.dx)) * np.ones(n-1)
        nums_ij_lo = ((-1 / self.dx**2) + (1j * k / self.dx)) * np.ones(n-1)
        data = np.concatenate((nums_ii, nums_ij_up, nums_ij_lo, [1j*k/self.dx - 1/self.dx**2, -1j*k/self.dx - 1/self.dx**2]))
        row = np.concatenate((np.arange(0, n, 1), np.arange(0, n-1, 1), np.arange(1, n, 1), [0, n-1]))
        col = np.concatenate((np.arange(0, n, 1), np.arange(1, n, 1), np.arange(0, n-1, 1), [n-1, 0]))
        D = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
        return D


    def edge_mode(self):
        D = self.construct_matrix(0, single_cell=False, edge=True)
        eigvals, eigvecs = eigsh(D, k=1, sigma=np.pi**2)
        return eigvals, eigvecs


    def solve_and_plot(self, plot=False):
        t = 0
        time = np.zeros(self.nsteps)
        Psi = np.zeros((self.lenx, self.nsteps), dtype=complex)

        for i in range(self.nsteps):
            plt.clf()

            ### Crank-Nicolson step ###
            F = self.forcing_matrix(t)
            mat = (1j * self.dt / 2) * (self.D + F)
            self.Psitp1 = spsolve(self.id + mat, (self.id - mat) * self.Psi)

            ### Update solution ###
            self.Psi = self.Psitp1.copy()
            #Psi[:,i] = np.real(self.Psi)
            Psi[:,i] = np.abs(self.Psi)

            ### Time ###
            time[i] = t

            if plot == True:
            ### Plot solution ###
                plt.plot(self.x, np.abs(self.Psi), "k-", lw=0.5, label=r"$|\psi(x)|$")
                plt.axis((self.xmin, self.xmax, 0, 1))
                plt.xlabel(r"$x$", fontsize=15)
                plt.ylabel(r"$|\psi(x)|$", fontsize=15)
                plt.legend(loc=1, fontsize=15)
                plt.title("Time = %1.3f" % (t + self.dt))
                plt.pause(0.01)

            t += self.dt

        return time, self.x, Psi
