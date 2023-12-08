import numpy as np
from dipy.core.sphere import disperse_charges, HemiSphere, Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.direction import peak_directions

thetas = list(np.load("synthetic_data/thetas.npy"))
phis = list(np.load("synthetic_data/phis.npy"))

hemisphere = HemiSphere(theta=thetas, phi=phis) # We already dispersed charges when building the hemisphere in dipy_test
sphere = Sphere(xyz=np.vstack((hemisphere.vertices, -hemisphere.vertices)))

F = np.load("synthetic_data/F.npy")

# Parameters for peak detection
relative_peak_threshold = 0.1
min_separation_angle = 15

peak_format = np.zeros((len(F), 15))

for i, sample in enumerate(F):
    # Duplicate the sample for both hemispheres
    F_sphere = np.hstack((sample, sample)) / 2

    # Find peak directions
    directions, values, indices = peak_directions(F_sphere, sphere, relative_peak_threshold, min_separation_angle)
    directions_flattened = directions.flatten()
    peak_format[i][0:len(directions_flattened)] = directions_flattened

print(peak_format)