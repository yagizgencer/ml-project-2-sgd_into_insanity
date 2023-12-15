import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
from dipy.core.sphere import disperse_charges, HemiSphere, Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.direction import peak_directions
#from dipy.viz import window, actor

def detect_peaks(F, relative_peak_threshold, min_separation_angle):
    peak_format = np.zeros((len(F), 15))
    for i, sample in enumerate(F):
        # Duplicate the sample for both hemispheres
        F_sphere = np.hstack((sample, sample)) / 2

        # Find peak directions
        directions, values, indices = peak_directions(F_sphere, sphere, relative_peak_threshold, min_separation_angle)
        directions = sample[indices][:, np.newaxis] * directions # multiplying with fractions
        directions_flattened = directions.flatten()
        peak_format[i][0:len(directions_flattened)] = directions_flattened
    return peak_format

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
H_noisy_input = np.zeros((n, len(real_bvals)))
H_noisy_validation = np.zeros((n, len(real_bvals)))
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
    H_noisy_input[i] = multi_tensor(gtab, mevals = [eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100], snr=random.randint(10, 30))[0]
    H_noisy_validation[i] = multi_tensor(gtab, mevals = [eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100], snr=random.randint(10, 30))[0]
    H_noiseless[i] = multi_tensor(gtab, mevals=[eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100],snr=None)[0]

N = int(1e5)
# Generating Nf
Nf = np.random.randint(3, size=N) + 1

# Creating S and our ground-truth
S = np.zeros((N, len(real_bvals)))
S_noiseless = np.zeros((N, len(real_bvals)))
F = np.zeros((N,n))
for i, k in enumerate(Nf):
    indices = np.random.choice(n, size=k, replace=False)
    random_nums = np.random.uniform(size=k)
    random_nums /= np.sum(random_nums)
    random_nums[random_nums < 0.10] = 0
    random_nums /= np.sum(random_nums)
    Nf[i] = np.sum(random_nums > 0)
    if np.sum(random_nums) == 0:
        print(f"Iteration {i}: All zero case encountered for row {i}")
    for j in range(k):
        S[i] += random_nums[j] * H_noisy_input[indices[j]]
        S_noiseless[i] += random_nums[j] * H_noiseless[indices[j]]
        F[i][indices[j]] = random_nums[j]

# Parameters for peak detection
relative_peak_threshold = 0.1
min_separation_angle = 15

peak_format = detect_peaks(F, relative_peak_threshold, min_separation_angle)

pd.to_pickle(peak_format, "synthetic_data/peaks_synthetic_formatted.pkl")

np.save("synthetic_data/S.npy", S)
np.save("synthetic_data/S_noiseless.npy", S_noiseless)
np.save("synthetic_data/F.npy", F)
np.save("synthetic_data/H_noisy_input.npy", H_noisy_input)
np.save("synthetic_data/H_noisy_validation.npy", H_noisy_validation)
np.save("synthetic_data/H_noiseless.npy", H_noiseless)

with open('synthetic_data/angle_pairs.pkl', 'wb') as file:
    pickle.dump(angle_pairs, file)

with open('synthetic_data/Nf.pkl', 'wb') as file:
    pickle.dump(Nf, file)
