import numpy as np
import matplotlib.pyplot as plt
from dirac import Dirac
from schrodinger import Schrodinger

AMPLITUDE = 0.01
FREQUENCY = 0.6
DELTA = 0.5
THETA = 0.5
LAMBDA = 2 * np.pi

NODES = 26000
XMAX = 1000
XMIN = -1000
TMAX = 0.01
dt = 0.01

D = Dirac(amplitude=AMPLITUDE, frequency=FREQUENCY, delta=DELTA,
theta_sharp=THETA, lambda_sharp=LAMBDA, Nodes=NODES, xmax=XMAX, xmin=XMIN,
tmax=TMAX, dt=dt)

S = Schrodinger(amplitude=AMPLITUDE, frequency=FREQUENCY, delta=DELTA,
Nodes=NODES, xmax=XMAX, xmin=XMIN, tmax=TMAX, dt=dt)

T, X, D_sol = D.solve_and_plot()
t, x, S_sol = S.solve_and_plot()

plt.plot(x, S_sol[:,-1], "b-", lw=0.2)
plt.plot(x, D_sol[:,-1], "r--", lw=2)
plt.ylim(0, 1)
plt.xlim(-400, 400)
plt.show()
