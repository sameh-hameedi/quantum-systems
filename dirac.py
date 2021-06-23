import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve


class Dirac:

    def __init__(self, amplitude=0.05, frequency=1.1, delta=0.8, theta_sharp=0.5, lambda_sharp=2*np.pi, Nodes=10000, xmax=250, xmin=-250, tmax=10, dt=0.1):
        ### Domain parameters ###
        self.Nodes = Nodes # Number of nodes
        self.xmin = delta * xmax
        self.xmax = delta * xmin

        ### Time parameters ###
        self.tmax = delta * tmax #* (2*np.pi)
        self.dt = delta * dt #* (2*np.pi)

        ### Forcing parameters ###
        self.A = amplitude # Forcing amplitude
        self.w = frequency #/ (2*np.pi) # Forcing angular frequency

        ### Parameters ###
        self.delta = delta
        self.theta_sharp = theta_sharp
        self.lambda_sharp = lambda_sharp

        self.init_domain()
        self.init_vectors()
        self.init_params()
        self.init_null_matrix()
        self.init_gradient_matrices()
        self.init_domain_wall_matrix()
        self.init_crank_nicolson_matrices()


    ### Problem-Specific Functions ###
    def domain_wall(self):
        return self.theta_sharp * np.tanh(self.x)

    def edge_state(self):
        n, m = 0, self.lenx
        if self.xmax > 600:
            n, m = np.where(self.x >= -600)[0][0], np.where(self.x >= 600)[0][0]
        vec = np.zeros(self.lenx)
        vec[n:m] = 1/np.cosh(self.x[n:m])**(self.theta_sharp/self.lambda_sharp)
        return vec

    def forcing(self, t):
        return self.A * np.cos(self.w*t)


    ### Main Setup ###
    def init_domain(self):
        self.dx = (self.xmax - self.xmin) / self.Nodes
        self.x = np.arange(self.xmin - self.dx, self.xmax + 2*self.dx, self.dx)
        self.lenx = len(self.x)


    def init_vectors(self):
        u10 = self.edge_state()
        u20 = 1j * self.edge_state()
        U0 = np.append(u10, u20)
        self.U = U0.copy()
        self.Utp1 = self.U.copy()
        self.scaling = np.amax(np.abs(self.U[:self.lenx] + self.U[self.lenx:]))


    def init_params(self):
        self.nsteps = round(self.tmax / self.dt)
        self.alpha = self.lambda_sharp * self.dt / (4 * self.dx)
        self.beta = (1j/2) * self.dt


    def init_null_matrix(self):
        self.null = coo_matrix(np.zeros((self.lenx, self.lenx)))


    def init_gradient_matrices(self):
        # Periodic BCs
        nums = self.alpha * np.ones(self.lenx - 1)
        plus_data = np.concatenate((nums, -1*nums, np.ones(self.lenx), [-1*self.alpha, self.alpha]))
        minus_data = np.concatenate((-1*nums, nums, np.ones(self.lenx), [self.alpha, -1*self.alpha]))
        row = np.concatenate((np.arange(0, self.lenx-1, 1), np.arange(1, self.lenx, 1), np.arange(0, self.lenx, 1), [0, self.lenx-1]))
        col = np.concatenate((np.arange(1, self.lenx, 1), np.arange(0, self.lenx-1, 1), np.arange(0, self.lenx, 1), [self.lenx-1, 0]))
        self.Dx_plus = coo_matrix((plus_data, (row, col)), shape=(self.lenx, self.lenx))
        self.Dx_minus = coo_matrix((minus_data, (row, col)), shape=(self.lenx, self.lenx))


    def init_domain_wall_matrix(self):
        nums = self.beta * self.domain_wall()
        self.K = coo_matrix((nums, (np.arange(0, self.lenx, 1), np.arange(0, self.lenx, 1))), shape=(self.lenx, self.lenx))


    def init_crank_nicolson_matrices(self):
        self.M = vstack([hstack([self.Dx_minus, self.K]), hstack([self.K, self.Dx_plus])], format="csr")
        self.N = vstack([hstack([self.Dx_plus, -1*self.K]), hstack([-1*self.K, self.Dx_minus])], format="csr")


    def forcing_matrix(self, t):
        nums = self.beta * self.lambda_sharp * self.forcing(t) * np.ones(self.lenx)
        F_submatrix = coo_matrix((nums, (np.arange(0,self.lenx,1), np.arange(0,self.lenx,1))), shape=(self.lenx, self.lenx))
        return vstack([hstack([F_submatrix, self.null]), hstack([self.null, -1*F_submatrix])], format="csr")


    def solve_and_plot(self, plot=False):
        t = 0
        time = np.zeros(self.nsteps)
        U = np.zeros((self.lenx, self.nsteps), dtype=complex)

        for i in range(self.nsteps):
            plt.clf()

            ### Crank-Nicolson step ###
            F = self.forcing_matrix(t)
            self.Utp1 = spsolve(self.M + F, (self.N - F) * self.U)

            ### Update solution ###
            self.U = self.Utp1.copy()
            U[:,i] = (np.abs(self.U[:self.lenx]) + np.abs(self.U[self.lenx:])) / 2
            #U[:,i] = np.abs(self.U[:self.lenx]*np.exp(2j*np.pi*self.x+1j*np.pi/4) + self.U[self.lenx:]*np.exp(-2j*np.pi*self.x+1j*np.pi/4)) / 2
            ### Time ###
            time[i] = t

            if plot == True:
                ### Plot solution ###
                plt.plot(self.x/self.delta, np.abs(self.U[:self.lenx]), "b-", lw=0.5, label=r"$|\alpha_1(X)|$")
                plt.plot(self.x/self.delta, np.abs(self.U[self.lenx:]), "r-", lw=0.5, label=r"$|\alpha_2(X)|$")
                plt.axis((self.xmin/delta, self.xmax/delta, 0, 1))
                plt.xlabel(r"$X$", fontsize=15)
                plt.ylabel(r"$|\alpha(X)|$", fontsize=15)
                plt.legend(loc=1, fontsize=15)
                plt.title("Time = %1.3f" % ((t + self.dt)/self.delta))
                plt.pause(0.01)

            t += self.dt



        return time, self.x, U
