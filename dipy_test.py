import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
#from dipy.viz import window, actor

n_pts = 64
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)

hsph_updated, potential = disperse_charges(hsph_initial, 5000)

vertices = hsph_updated.vertices
values = np.ones(vertices.shape[0])

bvecs = np.vstack((vertices, vertices))
bvals = np.hstack((1000 * values, 2500 * values))

bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
bvals = np.insert(bvals, (0, bvals.shape[0]), 0)

gtab = gradient_table(bvals, bvecs)

# Fixed diffusivity parameters 
d_parallel = 0.0015#1.7
d_perp = 0.0003#0.2
eigenvals = [d_parallel, d_perp, d_perp]

# Generate n pairs of angles 
n = 180
H = np.zeros((n, len(bvals)))
H_noisy = np.zeros((n, len(bvals)))
thetas = []
phis = []
np.random.seed(2)
for i in range(n):
    thetas.append(np.random.uniform(0,90))
    phis.append(np.random.uniform(0,360))

hemisphere = HemiSphere(theta=thetas, phi=phis)
hemisphere, _ = disperse_charges(hemisphere,50000)

angle_pairs = []
for i in range(len(hemisphere.phi)):
    angle_pairs.append((hemisphere.theta[i] * (180/np.pi), hemisphere.phi[i] * (180/np.pi)))
    H[i] = multi_tensor(gtab, mevals = [eigenvals], S0=100, angles = [angle_pairs[i]], fractions = [100], snr = None)[0]
    H_noisy[i] = multi_tensor(gtab, mevals = [eigenvals], S0=100, angles = [angle_pairs[i]], fractions = [100], snr = 20)[0]


N = int(1e5)
# Generating Nf
Nf = np.random.randint(3, size=N) + 1

# Creating S and our ground-truth
S = np.zeros((N, len(bvals)))
F = np.zeros((N,n))
for i, k in enumerate(Nf):
    indices = np.random.randint(n, size=k)
    random_nums = np.random.uniform(size=k)
    random_nums /= np.sum(random_nums)
    random_nums[random_nums < 0.10] = 0
    random_nums /= np.sum(random_nums)
    Nf[i] = np.sum(random_nums >= 0)
    for j in range(k):
        S[i] += random_nums[j] * H[indices[j]]
        F[i][indices[j]] = random_nums[j]

print(np.shape(S))
print(np.shape(H))
print(np.shape(F))

np.save("S.npy",S)
np.save("F.npy",F)
np.save("H.npy",H)
np.save("H_noisy.npy",H_noisy)

with open('angle_pairs.pkl', 'wb') as file:
    pickle.dump(angle_pairs, file)

with open('Nf.pkl', 'wb') as file:
    pickle.dump(Nf, file)