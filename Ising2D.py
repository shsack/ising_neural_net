import numpy as np
import pickle

Tc = 2.269
N = 10
np.random.seed(42)


def MonteCarloSpins(T):
    lattice = np.ones((N, N))

    def deltaE(i, j):

        inext = 0 if i == N - 1 else i + 1

        iprev = N - 1 if i == 0 else i - 1

        jnext = 0 if j == N - 1 else j + 1

        jprev = N - 1 if j == 0 else j - 1

        return lattice[iprev, j] + lattice[inext, j] + lattice[i, jprev] + lattice[i, jnext]

    def output(i, j):
        if lattice[i, j] == 1:
            return 1
        else:
            return 0

    for MCS in range(100):
        for i in range(N):
            for j in range(N):
                if deltaE(i, j) <= 0:
                    lattice[i, j] *= -1
                elif np.random.uniform(0., 1.) < np.exp(-deltaE(i, j) / T):
                    lattice[i, j] *= -1
    for i in range(N):
        for j in range(N):
            lattice[i, j] = output(i, j)
    return lattice.ravel()


def data(size):
    spins, label = np.zeros((size, N*N)), np.zeros((size, 2))

    for j in range(size):
        T_rand = np.random.uniform(0., 2. * Tc)
        if T_rand > Tc:
            label[j, 0] = 1
        else:
            label[j, 1] = 1
        for i in range(N*N):
            spins[j, i] = MonteCarloSpins(T_rand)[i]
        print(j, "out of", size)
    return spins, label


def data_bad(size):
    high, low = np.array([1, 0]), np.array([0, 1])
    spins, label = np.zeros((0, N*N)), np.zeros((0, 2))

    for j in range(size):
        T_rand = np.random.uniform(0., 2. * Tc)
        if T_rand > Tc:
            label = np.vstack((label, high))
        else:
            label = np.vstack((label, low))
        spins = np.vstack((spins, MonteCarloSpins(T_rand)))
        print(j, "out of", size)
    return spins, label

#print(data(3))

#print(data_bad(3))

def save(object, filename):
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(object, f)


#train_spins, train_labels = data(10000)
#test_spins, test_labels = data(2000)


#save(train_spins_new, 'train_spins_new'), save(train_labels_new, 'train_labels_new')
#save(test_spins_new, 'test_spins_new'), save(test_labels_new, 'test_labels_new')

print("set up the data!")