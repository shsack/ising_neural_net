import matplotlib.pyplot as plt
import numpy as np
import pickle
from numba import njit, prange

# Gives random spin configuration
def random_state(N):
    return 2 * np.random.randint(2, size=(N, N)) - 1

# Performs one Metropolis step
@njit(parallel=True) # for parallel for loop
def metropolis_step(lattice, T):
    for _ in prange(N * N):
        x, y = np.random.randint(0, N), np.random.randint(0, N)
        deltaE = 2 * lattice[x, y] * (lattice[(x + 1) % N, y] + lattice[x, (y + 1) % N] +
                       lattice[(x - 1) % N, y] + lattice[x, (y - 1) % N])
        if deltaE < 0:
            lattice[x, y] *= -1
        elif np.random.rand() < np.exp(-deltaE / T):
            lattice[x, y] *= -1

# Saves data
def save(object, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(object, f)

#--------------------------------------MAIN---------------------------------------------

# Setting up simulation parameters
N = 16
Tc = 2.269
MCsteps = 5000
EQsteps = 2000
ntemp = 100

T = np.linspace(1., 3.5, ntemp)
# magnetization = np.zeros(ntemp)
magnetization = []

# Initializing spins and labels
spins, labels = np.zeros((0, N * N)), np.zeros((0, 2))
high, low = np.array([1, 0]), np.array([0, 1])

# Loop over the temperatures
for index, temp in enumerate(T):
    tmp = []
    lattice = random_state(N)

    # Equilibrate spin lattice
    for eq in range(EQsteps):
        metropolis_step(lattice, temp)

    # Loop over spin configurations to collect data
    for mc in range(MCsteps):
        if mc % 200 == 0:
            tmp.append(np.sum(lattice))
        metropolis_step(lattice, temp)
    spins = np.vstack((spins, lattice.ravel()))

    # Append correct label corresponding to current spin configuration
    if temp < Tc:
        labels = np.vstack((labels, low))
    else:
        labels = np.vstack((labels, high))

    magnetization.append(np.mean(tmp) / (N * N))
    print('{} out of {} temperature steps'.format(index, len(T)))

# Save data
save(0.5 * (spins + 1), 'train_spins'), save(labels, 'train_labels'), save(T, 'temperature')
print("saved data!")

# Plot Monte Carlo simulation of magnetization
plt.plot(T, abs(np.array(magnetization)), 'o', color="green")
plt.xlabel("Temperature", fontsize=15)
plt.ylabel("Magnetization ", fontsize=15)
plt.grid()
plt.savefig('magnetization_MC.pdf')


