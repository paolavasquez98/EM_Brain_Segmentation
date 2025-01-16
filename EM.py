# Description: Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMMs).

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.cluster import KMeans
from functions import (
    multivariate_gaussian, e_step, m_step, log_likelihood, em_algorithm, assign_clusters
)

# Constants
BASE_FOLDER_PATH = '/Users/paolavasquez/Documents/MAIA_program/Girona/MISA/Labs/Lab1/P2_data'
N_CLUSTERS = 3
TOLERANCE = 1e-4
MAX_ITERATIONS = 100

# Validate base folder
if not os.path.isdir(BASE_FOLDER_PATH):
    raise FileNotFoundError(f"Base folder path '{BASE_FOLDER_PATH}' not found.")

# Load T1, T2, and Ground Truth (GT) images
t1_images, t2_images, gt_images = [], [], []

for folder_name in os.listdir(BASE_FOLDER_PATH):
    folder_path = os.path.join(BASE_FOLDER_PATH, folder_name)

    # Skip non-directory files
    if not os.path.isdir(folder_path):
        continue

    # Paths to required files
    t1_file_path = os.path.join(folder_path, 'T1.nii')
    t2_file_path = os.path.join(folder_path, 'T2_FLAIR.nii')
    gt_file_path = os.path.join(folder_path, 'LabelsForTesting.nii')

    # Validate file existence
    if os.path.isfile(t1_file_path) and os.path.isfile(t2_file_path) and os.path.isfile(gt_file_path):
        t1_images.append(nib.load(t1_file_path).get_fdata())
        t2_images.append(nib.load(t2_file_path).get_fdata())
        gt_images.append(nib.load(gt_file_path).get_fdata() > 0)  # Create mask
    else:
        print(f"Required files not found in: {folder_path}")

# Preprocess images
y1_images, y2_images = [], []
for t1_image, t2_image, gt_image in zip(t1_images, t2_images, gt_images):
    t1_image[gt_image == 0] = 0  # Mask out non-brain regions
    t2_image[gt_image == 0] = 0
    y1_images.append(t1_image[gt_image])
    y2_images.append(t2_image[gt_image])

# Perform KMeans clustering
means, covariances, labels_list = [], [], []

for y1, y2 in zip(y1_images, y2_images):
    data = np.stack((y1, y2), axis=1)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(data)

    # Store cluster centers, labels, and covariances
    means.append(kmeans.cluster_centers_)
    labels = kmeans.labels_
    labels_list.append(labels)

    # Compute cluster covariances
    image_covariances = []
    for i in range(N_CLUSTERS):
        cluster_data = data[labels == i]
        covariance_matrix = (
            np.cov(cluster_data, rowvar=False) if len(cluster_data) > 1 else np.eye(cluster_data.shape[1])
        )
        image_covariances.append(covariance_matrix)
    covariances.append(image_covariances)

# Initialize Gaussian Mixture Model for the first image
mean = means[0]
covariance = covariances[0]
x = np.stack((y1_images[0], y2_images[0]), axis=1)

# Compute mixing coefficients
n_samples = len(x)
mixing_coefficients = np.array([np.sum(labels_list[0] == k) / n_samples for k in range(N_CLUSTERS)])

# Run the EM algorithm
final_means, final_covariances, final_mixing_coefficients, log_likelihoods = em_algorithm(
    x, mean, covariance, mixing_coefficients, max_iters=MAX_ITERATIONS, tol=TOLERANCE
)

# Assign data points to clusters
cluster_assignments = assign_clusters(x, final_means, final_covariances, final_mixing_coefficients)
print("Cluster Assignments:", cluster_assignments)

# Visualize clustering results
y1, y2 = y1_images[0], y2_images[0]
data = np.stack((y1, y2), axis=1)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='viridis', marker='o', label='Data Points')
plt.scatter(final_means[:, 0], final_means[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
plt.title('EM Clustering Results')
plt.xlabel('T1 Intensities')
plt.ylabel('T2 Intensities')
plt.legend()
plt.show()