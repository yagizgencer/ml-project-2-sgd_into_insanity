import numpy as np
from dipy.core.sphere import disperse_charges, HemiSphere, Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.direction import peak_directions
import nibabel as nib

thetas = list(np.load("synthetic_data/thetas.npy"))
phis = list(np.load("synthetic_data/phis.npy"))

hemisphere = HemiSphere(theta=thetas, phi=phis) # We already dispersed charges when building the hemisphere in dipy_test
sphere = Sphere(xyz=np.vstack((hemisphere.vertices, -hemisphere.vertices)))

F = np.load("synthetic_data/F.npy")

def detect_peaks(F, relative_peak_threshold, min_separation_angle):
    peak_format = np.zeros((len(F), 15))
    for i, sample in enumerate(F):
        # Duplicate the sample for both hemispheres
        F_sphere = np.hstack((sample, sample)) / 2

        # Find peak directions
        directions, values, indices = peak_directions(F_sphere, sphere, relative_peak_threshold, min_separation_angle)
        directions_flattened = directions.flatten()
        peak_format[i][0:len(directions_flattened)] = directions_flattened
    return peak_format

def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def generate_peak_gt(F, thetas, phis):
    peaks = np.zeros((len(F), 15))
    cartesians = np.zeros((len(thetas), 3))
    for i in range(len(thetas)):
        cartesians[i][0], cartesians[i][1], cartesians[i][2] = spherical_to_cartesian(thetas[i], phis[i]) 
    for i, sample in enumerate(F):
        nnz_indices = np.nonzero(sample)[0]
        for j in range(len(nnz_indices)):
            index = nnz_indices[j]
            peaks[i][3*j] = cartesians[index][0]
            peaks[i][3*j + 1] = cartesians[index][1]
            peaks[i][3*j + 2] = cartesians[index][2]
    return peaks


# Parameters for peak detection
relative_peak_threshold = 0.1
min_separation_angle = 15

#peak_format = np.zeros((len(F), 15))
#
#for i, sample in enumerate(F):
#    # Duplicate the sample for both hemispheres
#    F_sphere = np.hstack((sample, sample)) / 2
#
#    # Find peak directions
#    directions, values, indices = peak_directions(F_sphere, sphere, relative_peak_threshold, min_separation_angle)
#    directions_flattened = directions.flatten()
#    peak_format[i][0:len(directions_flattened)] = directions_flattened
peak_format = detect_peaks(F, relative_peak_threshold, min_separation_angle)
# Check if any row contains only zeros
rows_with_only_zeros = np.all(peak_format == 0, axis=1)

# Find the indices of such rows
zero_rows_indices = np.where(rows_with_only_zeros)[0]

# Print the indices of rows that contain only zeros
if zero_rows_indices.size > 0:
    print("Rows containing only zeros:", zero_rows_indices)
else:
    print("There are no rows containing only zeros.")

reshaped_data = peak_format.reshape((100, 100, 100, 15))

# Create a NIfTI image (assuming affine as identity, you might need to adjust this)
affine = np.eye(4)
nifti_img = nib.Nifti1Image(reshaped_data, affine)

# Save the NIfTI image
nib.save(nifti_img, 'your_image.nii.gz')

#print(np.shape(peak_format))
# Saving the ground truth to a .ni.gzz file 
#peaks = nib.load("your_image.nii.gz").get_fdata()
#print(peaks)
peaks_gt = generate_peak_gt(F, thetas, phis)
print(peaks_gt)
print(peak_format)

# Check if any row contains only zeros
rows_with_only_zeros = np.all(peaks_gt == 0, axis=1)

# Find the indices of such rows
zero_rows_indices = np.where(rows_with_only_zeros)[0]

# Print the indices of rows that contain only zeros
if zero_rows_indices.size > 0:
    print("Rows containing only zeros:", zero_rows_indices)
else:
    print("There are no rows containing only zeros.")
