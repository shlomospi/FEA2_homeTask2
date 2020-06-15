import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import eigh


def external_f(t):
    return np.zeros(5)*t


def time_to_step(time, dt):
    return int(time/dt)


def round_up(n, decimals=0):
    #  needs to import math
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def get_analytical(space_time=0.6, precision=50, x_coord=np.arange(0, 1, 0.001), mode="space"):
    
    temp_analytical = np.zeros(1000)
    if mode == "space":
        for n in range(1, precision):
            lambda_n = (n - 0.5) * math.pi  # L=1
            a_n = (2 / lambda_n) * math.sin(lambda_n)
            for i in range(0, len(x_coord)):
                temp_analytical[i] += a_n * math.exp(-lambda_n ** 2 * space_time) * math.cos(lambda_n * x_coord[i])
    elif mode == "time":
        for n in range(1, precision):
            lambda_n = (n - 0.5) * math.pi  # L=1
            a_n = (2 / lambda_n) * math.sin(lambda_n)
            const_in_time = math.cos(lambda_n * space_time)
            for t in range(1000):
                temp_analytical[t] += a_n * math.exp(-lambda_n ** 2 * t/1000) * const_in_time
    elif mode == "time_dot":
        for n in range(1, precision):
            lambda_n = (n - 0.5) * math.pi  # L=1
            a_n = (2 / lambda_n) * math.sin(lambda_n)
            const_in_time = math.cos(lambda_n * space_time)
            for t in range(1000):
                temp_analytical[t] += -lambda_n ** 2 * a_n * math.exp(-lambda_n ** 2 * t/1000) * const_in_time
    else:
        print("wrong mode")

    return temp_analytical


def solve_heat_rod(alpha=1, dt=0.1):
    print(dt)
    # Geometry and Mesh
    L = 1                           # rod length
    x = np.arange(0., 1.2, 0.2)     # Node locations
    number_elements = int(len(x) - 1)   # num of elements
    number_nodes = number_elements + 1
    Neq = number_nodes
    direchlet_bc = [6]
    h = 0.2

    # initial conditions
    u0 = 1
    d_0 = np.array([u0, u0, u0, u0, u0])
    F_0 = np.array([0, 0, 0, 0, 0])

    # Time
    NumSteps = int(1//dt+1)

    #  Physical Properties
    k = 1  # Conductivity
    c = 1  # Heat Capacity
    A = 1  # Cross section
    #  Build M & K Matrices
    Mfull = np.zeros((Neq, Neq))
    Kfull = np.zeros((Neq, Neq))

    for element in range(number_elements):

        M_ele = c * A * h / 6 * np.array([[2, 1], [1, 2]])
        K_ele = A * k / h * np.array([[1, -1], [-1, 1]])
        Mfull[element:element+M_ele.shape[0], element:element+M_ele.shape[0]] += M_ele
        Kfull[element:element+K_ele.shape[0], element:element+K_ele.shape[0]] += K_ele
    M = Mfull[:5, :5]
    K = Kfull[:5, :5]

    if alpha == 0:
        Mlumped = np.zeros((Neq-1, Neq-1))
        for n in range(Neq-1):
            Mlumped[n, n] = np.sum(M[n, :])
        M = Mlumped
        print("For Lumped M, the eigen values are:")
        print(eigh(K, M, eigvals_only=True))
    else:
        print("For regular M, the eigen values are:")
        print(eigh(K, M, eigvals_only=True))

    #   "Initial Vn"
    v_0 = np.linalg.solve(M, np.subtract(F_0, np.matmul(K, d_0)))
    d_n1 = d_0
    v_n1 = v_0

    # Keep history
    temp_history = np.zeros((Neq, NumSteps+1))
    temp_history[:5, 0] = d_0
    for DOF in direchlet_bc:
        temp_history[DOF-1, 0] = 1

    temp_rate_history = np.zeros((Neq, NumSteps+1))
    temp_rate_history[:5, 0] = np.zeros(5)  # v_0

    #   "Stepping in Time"
    for step in range(NumSteps):
        Time = step*dt
        d_n = d_n1
        v_n = v_n1
        d_n1_gal = d_n + (1-alpha)*dt*v_n   # prediction

        Mstar = M + alpha*dt*K
        Fstar = np.subtract(external_f(Time), np.matmul(K, d_n1_gal))
        v_n1 = np.linalg.solve(Mstar, Fstar)
        d_n1 = d_n1_gal + alpha * dt * v_n1  # Corrected
        temp_history[:5, step+1] = d_n1
        temp_rate_history[:5, step+1] = v_n1

    #       Plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle('\u03B1 = {}  \u0394t = {}h\u00b2/k'.format(alpha, round_up(dt/h**2, 5)))

    #   axs[0, 0]
    TimeSamples = [0., 0.01, 0.04, 0.1, 0.4, 1]
    x_coord = np.arange(0, 1, 0.001)
    for sample in TimeSamples:
        currentstep = int(time_to_step(sample, dt))
        currentTemp = temp_history[:, currentstep]
        axs[0, 0].plot(x, currentTemp, label='time={}'.format(sample))
        if sample > 0:
            tempo_analytical = get_analytical(sample, 100)
            axs[0, 0].plot(x_coord, tempo_analytical, "k-", alpha=0.5)
    axs[0, 0].set(xlabel='x/L', ylabel='u^h/u_0')
    axs[0, 0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                     ncol=6, mode="expand", borderaxespad=0.)
    
    #   axs[0, 1]
    axs[0, 1].plot(np.arange(0, k/L**2*((0.1//dt)*dt+dt), k/L**2*dt),
                   temp_history[3, :int(time_to_step(0.1, dt) + 1)], 'tab:orange')
    tempo_analytical = get_analytical(0.6, 100, x_coord, mode="time")
    axs[0, 1].plot(x_coord, tempo_analytical, "k-", alpha=0.5)
    axs[0, 1].set(xlabel='kt/L\u00b2', ylabel='u^h/u_0|x=0.6L')
    axs[0, 1].set_xlim([0., 0.1])
    axs[0, 1].set_ylim([0.6, 1.1])

    #   axs[1, 0]
    for sample in TimeSamples:
        currentstep = int(time_to_step(sample, dt))
        currentTemp = temp_history[:, currentstep]
        axs[1, 0].plot(x, currentTemp)
        if sample > 0:
            tempo_analytical = get_analytical(sample, 100)
            axs[1, 0].plot(x_coord, tempo_analytical, "k-", alpha=0.5)
    axs[1, 0].set(xlabel='x/L', ylabel='u^h/u_0')
    axs[1, 0].set_ylim([0.995, 1.005])

    #   axs[1, 1]
    axs[1, 1].plot(np.arange(0, k/L**2*((0.1//dt)*dt+dt), k/L**2*dt),
                   temp_rate_history[3, :int(time_to_step(0.1, dt) + 1)], 'tab:orange')
    tempo_analytical = get_analytical(0.6, 50, x_coord, mode="time_dot")
    tempo_analytical[0] = 0
    axs[1, 1].plot(x_coord, tempo_analytical, "k-", alpha=0.5)
    axs[1, 1].set(xlabel='kt/L\u00b2', ylabel='L\u00b2*udot^h/ku_0|x=0.6L')
    axs[1, 1].set_xlim([0., 0.1])
    axs[1, 1].set_ylim([-11, 11])

    plt.savefig('graphs/dt{}alpha{}.png'.format(dt, alpha))
