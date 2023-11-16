import matplotlib.pyplot as plt
import numpy as np

from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf

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
d_parallel = 1.7
d_perp = 0.2
eigenvals = [d_parallel, d_perp, d_perp]

# Generate n pairs of angles 
n = 300
angle_values = np.linspace(0,360,int(1e3))
angle_pairs = []
H = np.zeros((n, len(bvals)))
for i in range(n):
    #angle_pair = (st.uniform(scale=180).rv(size=1), st.uniform(scale=360).rv(size=1))
    angle_pair = (np.random.uniform(0,180), np.random.uniform(0,360))
    angle_pairs.append(angle_pair)
    H[i] = multi_tensor(gtab, eigenvals, S0=100, angle=[angle_pair], fractions=[100], snr=None)[0]


N = 650
# Generating Nf
Nf = np.random.randint(3, size=N) + 1

# Creating S and our ground-truth
S = np.zeros((N, len(bvals)))
for i, k in enumerate(Nf):
    indices = np.random.randint(n, size=k)
    #random_nums = st.uniform().rv(size=k)
    random_nums = np.random.uniform(size=k)
    random_nums /= np.sum(random_nums)
    S[i] = np.sum(random_nums * H[indices])





