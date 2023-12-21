import numpy as np
from dipy.direction import peak_directions
from dipy.core.sphere import Sphere
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math


def detect_peaks(F, hemisphere, relative_peak_threshold, min_separation_angle):
    """
    :param F & hemisphere: F is the odf defined on the directions of the hemisphere
    :param relative_peak_threshold: only peaks greater than this fraction of the largest peak are returned
    :param min_separation_angle: the minimum difference between directions of two peaks
    :return: x, y, z coordinates of the peaks
    """
    peak_format = np.zeros((len(F), 42))
    max_peak_count = 0
    min_peak_count = 15
    #Create a sphere by concataneting the hemisphere and the same hemisphere in the opposite direction
    sphere = Sphere(xyz=np.vstack((hemisphere.vertices, -hemisphere.vertices)))
    for i, sample in enumerate(F):
        #Duplicate the odf sample for both hemispheres
        F_sphere = np.hstack((sample, sample)) / 2
        # Find peak directions
        directions, values, indices = peak_directions(F_sphere, sphere, relative_peak_threshold, min_separation_angle)
        directions = sample[indices][:, np.newaxis] * directions  # multiplying with fractions
        directions_flattened = directions.flatten()
        peak_format[i][0:len(directions_flattened)] = directions_flattened
        if len(directions_flattened) / 3 > max_peak_count:
            max_peak_count = len(directions_flattened) / 3
        if len(directions_flattened) / 3 < min_peak_count:
            min_peak_count = len(directions_flattened) / 3
    return peak_format, max_peak_count, min_peak_count


class MatrixFactorizationBatchNormalizationNet_4_layer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MatrixFactorizationBatchNormalizationNet_4_layer, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.softplus(self.fc4(x))  # Softplus activation for the output layer
        return x

class MatrixFactorizationBatchNormalizationNet_3_layer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MatrixFactorizationBatchNormalizationNet_3_layer, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.softplus(self.fc3(x))  # Softplus activation for the output layer
        return x


class MatrixFactorizationBatchNormalizationNet_5_layer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MatrixFactorizationBatchNormalizationNet_5_layer, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.softplus(self.fc5(x))  # Softplus activation for the output layer
        return x


class MatrixFactorizationNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MatrixFactorizationNet, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.softplus(self.fc4(x))  # Softplus activation for the output layer
        return x


class MatrixDataset(Dataset):
    def __init__(self, S, F):
        self.S = S
        self.F = F

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.S[idx], self.F[idx]


def apply_sparsity(F, sparsity_threshold=0.1):
    """
    This function takes an odf matrix F and a sparsity threshold as input.
    It sets all values below the max(F)*sparsity_threshold to zero.
    It is applied to output of the network to get the final F.
    """
    max_value, _ = F.max(dim=1, keepdim=True)
    F[F < sparsity_threshold * max_value] = 0
    return F


def spherical_to_cartesian(theta, phi):
    """
    This function takes spherical theta and phi coordinates in degrees and returns x, y, z in cartesian coordinates.
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    This function takes cartesian x, y, z coordinates and returns spherical coordinates theta and phi in degrees.
    """
    phi = math.atan2(y, x)
    r_xy = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(r_xy, z)

    phi_deg = math.degrees(phi)
    theta_deg = math.degrees(theta)

    return phi_deg, theta_deg


def angles_fractions(peaks):
    """
    This function takes as input a list of length n = 3*k which includes
     peak directions (x, y, z in Cartesian Coordinates) in a voxel.
    It calculates the angle pairs (theta, phi) and fraction of that angle pair for each of the k peaks.
    """
    angles = []
    fractions = []
    for i in range(len(peaks)//3):
        if peaks[3*i] != 0.0:
            phi, theta = cartesian_to_spherical(peaks[3*i], peaks[3*i+1], peaks[3*i+2])
            angles.append((theta, phi))
            fractions.append(peaks[3*i]**2 + peaks[3*i+1]**2 + peaks[3*i+2]**2)
    fractions = np.array(fractions) / sum(fractions)
    return angles, fractions
