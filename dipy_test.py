import matplotlib.pyplot as plt
import numpy as np

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
n = 300
angle_values = np.linspace(0,360,int(1e3))
angle_pairs = []
H = np.zeros((n, len(bvals)))
for i in range(n):
    #angle_pair = (st.uniform(scale=180).rv(size=1), st.uniform(scale=360).rv(size=1))
    angle_pair = (np.random.uniform(0,180), np.random.uniform(0,360))
    angle_pairs.append(angle_pair)
    H[i] = multi_tensor(gtab, [eigenvals], S0=100, angles=[angle_pair], fractions=[100], snr=None)[0]


N = int(1e3)
# Generating Nf
Nf = np.random.randint(3, size=N) + 1

# Creating S and our ground-truth
S = np.zeros((N, len(bvals)))
F = np.zeros((N,n))
for i, k in enumerate(Nf):
    indices = np.random.randint(n, size=k)
    #random_nums = st.uniform().rv(size=k)
    random_nums = np.random.uniform(size=k)
    random_nums /= np.sum(random_nums)
    for j in range(k):
        S[i] += random_nums[j] * H[indices[j]]
        # SNR 20-30
        F[i][indices[j]] = random_nums[j]
    #S[i] = np.sum(random_nums * H[indices])

"""sphere = get_sphere('repulsion724')
sphere = sphere.subdivide(2)

#Creating the orientation distribution function:
non_zero_indices = np.nonzero(F[0])
angles = []
fractions = []
for index in non_zero_indices:
    angles.append(angle_pairs[index])
    fractions.append(F[0][index])
odf = multi_tensor_odf(sphere.vertices, eigenvals, angles, fractions)

# Enables/disables interactive visualization
interactive = False

scene = window.Scene()

odf_actor = actor.odf_slicer(odf[None, None, None, :], sphere=sphere,
                             colormap='plasma')
odf_actor.RotateX(90)

scene.add(odf_actor)

print('Saving illustration as multi_tensor_simulation')
window.record(scene, out_path='multi_tensor_simulation.png', size=(300, 300))
if interactive:
    window.show(scene)"""

print(np.shape(S))
print(np.shape(H))
print(np.shape(F))

np.save("S.npy",S)
np.save("F.npy",F)
np.save("H.npy",H)





