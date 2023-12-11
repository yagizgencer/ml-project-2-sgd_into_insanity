import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
#from dipy.viz import window, actor

with open('real_data/hardi-scheme.bvec.txt', 'r') as file:
    lines = file.readlines()

real_bvecs = []
for line in lines:
    values = line.split()
    real_bvecs.append([float(value) for value in values])
real_bvecs = np.array(real_bvecs).T
#print(real_bvecs)

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
    H_noisy_input[i] = multi_tensor(gtab, mevals = [eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100], snr=random.randint(10, 30))[0]
    H_noisy_validation[i] = multi_tensor(gtab, mevals = [eigenvals], S0=100, angles=[angle_pairs[i]], fractions=[100], snr=random.randint(10, 30))[0]

thetas = np.save("synthetic_data/thetas.npy", np.array(hemisphere.theta))
phis = np.save("synthetic_data/phis.npy", np.array(hemisphere.phi))

N = int(1e6)
# Generating Nf
Nf = np.random.randint(3, size=N) + 1
print(np.shape(np.nonzero(Nf)[0]))

# Creating S and our ground-truth
S = np.zeros((N, len(real_bvals)))
F = np.zeros((N,n))
for i, k in enumerate(Nf):
    #indices = np.random.randint(n, size=k)
    indices = np.random.choice(n, size=k, replace=False)
    random_nums = np.random.uniform(low = 0.1, size=k)
    random_nums /= np.sum(random_nums)
    # After thresholding
    # Define the threshold and the minimum number of non-zero elements to retain
    threshold = 0.10
    min_non_zero = 1  # At least one non-zero entry
    
    # Apply threshold, but ensure at least 'min_non_zero' largest values remain non-zero
    largest_indices = np.argpartition(random_nums, -min_non_zero)[-min_non_zero:]
    random_nums_below_threshold = random_nums < threshold
    random_nums_below_threshold[largest_indices] = False
    random_nums[random_nums_below_threshold] = 0
    
    # Check if all values are zero after thresholding
    if np.sum(random_nums) == 0:
        print(f"Iteration {i}: All zero case encountered for row {i}")
        random_nums[np.argmax(random_nums)] = 0.1
        random_nums /= np.sum(random_nums)
    
    # Update the number of fibers (Nf)
    Nf[i] = np.count_nonzero(random_nums)

    for j in range(k):
        S[i] += random_nums[j] * H_noisy_input[indices[j]]
        #if(random_nums[j] != 0):
        F[i][indices[j]] = random_nums[j]

    # Diagnostic print to check if a row in F is all zeros
    if np.all(F[i] == 0):
        print(f"Row {i} in F is all zeros - Random_nums: {random_nums}, Indices: {indices}")

print(Nf)
print(np.shape(F))
# Check if any row contains only zeros
rows_with_only_zeros = np.all(F == 0, axis=1)

# Find the indices of such rows
zero_rows_indices = np.where(rows_with_only_zeros)[0]

# Print the indices of rows that contain only zeros
if zero_rows_indices.size > 0:
    print("Rows containing only zeros:", zero_rows_indices)
else:
    print("There are no rows containing only zeros.")


np.save("synthetic_data/S.npy",S)
np.save("synthetic_data/F.npy",F)
np.save("synthetic_data/H_noisy_input.npy",H_noisy_input)
np.save("synthetic_data/H_noisy_validation.npy",H_noisy_validation)