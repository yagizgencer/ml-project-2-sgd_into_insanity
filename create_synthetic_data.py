"""
This file creates the synthetic data for the experiments. It generates 180 pairs of angles (theta, phi) and
places them evenly on a hemisphere.
Then it creates base matrix H including single fiber response of corresponding angle pairs. It creates 6 different
H with different SNR values.
Then it generates Nf, the number of fibers in each voxel, from a multinomial distribution with probabilities [0.55, 0.30, 0.15].
For each voxel, it randomly chooses Nf number of angle pairs from the 180 pairs and assigns a random fraction to each of them.
The angle pairs and their fractions are saved in the matrix F.
The script then finally creates S, the signal matrix, by summing up the base matrices H with the corresponding fractions in F,
and stacking them together.
The script saves S, F, H, Nf, thetas and phis in the synthetic_data folder
"""

import matplotlib.pyplot as plt
import pickle
import random
from dipy.core.sphere import disperse_charges, HemiSphere, Sphere
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import multi_tensor
from helpers import *

# Load real bvecs and bvals from the Hardi Data
with open('real_data/hardi-scheme.bvec.txt', 'r') as file:
    lines = file.readlines()

real_bvecs = []
for line in lines:
    values = line.split()
    real_bvecs.append([float(value) for value in values])
real_bvecs = np.array(real_bvecs).T


with open('real_data/hardi-scheme.bval.txt', 'r') as file:
    lines = file.readlines()

real_bvals = []
for line in lines:
    values = line.split()
    real_bvals.append([float(value) for value in values])

real_bvals = np.array(real_bvals).reshape(-1)

gtab = gradient_table(real_bvals, real_bvecs)
d_parallel = 0.0015
d_perp = 0.00039
eigenvals = [d_parallel, d_perp, d_perp]

# Generate n pairs of angles 
n = 180
H_snr_10 = np.zeros((n, len(real_bvals)))
H_snr_15 = np.zeros((n, len(real_bvals)))
H_snr_20 = np.zeros((n, len(real_bvals)))
H_snr_25 = np.zeros((n, len(real_bvals)))
H_snr_30 = np.zeros((n, len(real_bvals)))
H_noiseless = np.zeros((n, len(real_bvals)))
thetas = []
phis = []
np.random.seed(2)

for i in range(n):
    thetas.append(np.random.uniform(0,90))
    phis.append(np.random.uniform(0,360))


hemisphere = HemiSphere(theta=thetas, phi=phis)
hemisphere, _ = disperse_charges(hemisphere,int(50000))
sphere = Sphere(xyz=np.vstack((hemisphere.vertices, -hemisphere.vertices)))

np.save("synthetic_data/thetas.npy", np.array(hemisphere.theta))
np.save("synthetic_data/phis.npy", np.array(hemisphere.phi))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hemisphere.vertices[:,0], hemisphere.vertices[:,1], hemisphere.vertices[:,2], c='r', marker='o')

# Set labels for each axis
ax.set_xlabel('x Label')
ax.set_ylabel('y Label')
ax.set_zlabel('z Label')

fig.savefig("angle_pairs_visualization_" + str(n) + "_pairs.png")

angle_pairs = []
for i in range(len(hemisphere.phi)):
    angle_pairs.append((hemisphere.theta[i] * (180/np.pi), hemisphere.phi[i] * (180/np.pi)))
    H_snr_10[i] = multi_tensor(gtab, mevals = [eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100],
                               snr=10)[0]
    H_snr_15[i] = multi_tensor(gtab, mevals=[eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100],
                               snr=15)[0]
    H_snr_20[i] = multi_tensor(gtab, mevals=[eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100],
                               snr=20)[0]
    H_snr_25[i] = multi_tensor(gtab, mevals=[eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100],
                               snr=25)[0]
    H_snr_30[i] = multi_tensor(gtab, mevals=[eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100],
                               snr=30)[0]
    H_noiseless[i] = multi_tensor(gtab, mevals=[eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100],snr=None)[0]

N = int(2e5)
# Generating Nf
probabilities = [0.55, 0.30, 0.15]
Nf = np.random.choice([1,2,3], size=N, p=probabilities)


# Creating S and our ground-truth
S = np.zeros((6*N, len(real_bvals)))
S_noiseless = np.zeros((N, len(real_bvals)))
F = np.zeros((6*N,n))
for i, k in enumerate(Nf):
    indices = np.random.choice(n, size=k, replace=False)
    random_nums = np.random.uniform(size=k)
    random_nums /= np.sum(random_nums)
    random_nums[random_nums < 0.10] = 0
    random_nums /= np.sum(random_nums)
    Nf[i] = np.sum(random_nums > 0)
    for j in range(k):
        S[i] += random_nums[j] * H_noiseless[indices[j]]
        S[N + i] += random_nums[j] * H_snr_30[indices[j]]
        S[2*N + i] += random_nums[j] * H_snr_25[indices[j]]
        S[3*N + i] += random_nums[j] * H_snr_20[indices[j]]
        S[4*N + i] += random_nums[j] * H_snr_15[indices[j]]
        S[5*N + i] += random_nums[j] * H_snr_10[indices[j]]
        F[i][indices[j]] = random_nums[j]
        F[N + i][indices[j]] = random_nums[j]
        F[2*N + i][indices[j]] = random_nums[j]
        F[3*N + i][indices[j]] = random_nums[j]
        F[4*N + i][indices[j]] = random_nums[j]
        F[5*N + i][indices[j]] = random_nums[j]


combined_data = list(zip(S, F))
random.seed(42)
random.shuffle(combined_data)
S, F = zip(*combined_data)

np.save("synthetic_data/S.npy", S)
np.save("synthetic_data/F.npy", F)
np.save("synthetic_data/H.npy", H_noiseless)

with open('synthetic_data/angle_pairs.pkl', 'wb') as file:
    pickle.dump(angle_pairs, file)

with open('synthetic_data/Nf.pkl', 'wb') as file:
    pickle.dump(Nf, file)
