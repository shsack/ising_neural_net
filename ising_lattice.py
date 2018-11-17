import matplotlib.pyplot as plt
import numpy as np
import pickle

# Gives random spin configuration
def randomstate(N):
    return 2 * np.random.randint(2, size=(N, N)) - 1

# Performs one Metropolis step
def metropolis_step(lattice, T):
    for _ in range(N * N):
        x, y = np.random.randint(0, N), np.random.randint(0, N)
        spin = lattice[x, y]
        deltaE = 2 * spin * (lattice[(x + 1) % N, y] + lattice[x, (y + 1) % N] +
                       lattice[(x - 1) % N, y] + lattice[x, (y - 1) % N])
        if deltaE < 0:
            spin *= -1
        elif np.random.rand() < np.exp(-deltaE / T):
            spin *= -1
        lattice[x, y] = spin
    return lattice

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
Magnetization = np.zeros(ntemp)

# Initializing spins and labels
spins, labels = np.zeros((0, N * N)), np.zeros((0, 2))
high, low = np.array([1, 0]), np.array([0, 1])

# Perform simulation
for t in range(len(T)):
    tmp = []
    lattice = randomstate(N)

    # Equilibrate spin lattice
    for eq in range(EQsteps):
        metropolis_step(lattice, T[t])

    # Loop over spin configurations to collect data
    for mc in range(MCsteps):
        if mc % 200 == 0:
            metropolis_step(lattice, T[t])
            tmp.append(np.sum(lattice))
    spins = np.vstack((spins, lattice.ravel()))

    # Append correct label corresponding to current spin configuration
    if T[t] < Tc:
        labels = np.vstack((labels, low))
    else:
        labels = np.vstack((labels, high))
    Magnetization[t] = np.mean(tmp) / (N * N)
    print(t, "out of", len(T))

# Save data
save(0.5 * (spins + 1), 'train_spins'), save(labels, 'train_labels'), save(T, 'temperature')
#save(0.5 * (spins + 1), 'train_spins_two'), save(labels, 'train_labels_two'), save(T, 'temperature')
print("saved data!")

# Plot Monte Carlo simulation of magnetization
plt.plot(T, abs(Magnetization), 'o', color="green")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Magnetization ", fontsize=20)
plt.grid()
plt.show()


