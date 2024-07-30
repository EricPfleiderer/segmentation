import random

import torch

from enum import Enum


class KMeansInit(Enum):
    Forgy = 'forgy'
    RandomPartition = 'random partition'
    Manual = 'manual'


def build_feature_matrix(torch_image):

    print('Building feature matrix from selected image')
    matrix = torch.zeros(size=(torch_image.size(1) * torch_image.size(2), 6), dtype=torch.int64)  # 6 features: X, Y, R, G, B, A

    n = 0
    for y in range(torch_image.size(1)):
        for x in range(torch_image.size(2)):
            xy = torch.tensor([x, y])
            rgba = torch_image[:, y, x]
            matrix[n] = torch.cat((xy, rgba))
            n += 1
    return matrix


def build_torch_image(feature_matrix, width, height):
    # build image
    cluster_img = torch.zeros(size=(4, height, width), dtype=torch.uint8)

    for n in range(len(feature_matrix)):
        x = feature_matrix[n, 0].to(torch.int16)
        y = feature_matrix[n, 1].to(torch.int16)
        r = feature_matrix[n, 2]
        g = feature_matrix[n, 3]
        b = feature_matrix[n, 4]
        a = feature_matrix[n, 5]
        cluster_img[0, y, x] = r
        cluster_img[1, y, x] = g
        cluster_img[2, y, x] = b
        cluster_img[3, y, x] = a

    return cluster_img


def kmeans(n_centroids, feature_matrix, threshold=5, init_type: KMeansInit = KMeansInit.Forgy):

    # Initialize labels randomly in preparation for Random Partitioning of clusters
    labels = torch.randint(low=1, high=n_centroids + 1, size=(len(feature_matrix),), dtype=torch.int64)

    # Assign class 0 to pixels who do not meet the alpha threshold to be considered for clustering
    labels[torch.where(feature_matrix[:, 5] < threshold)[0]] = 0

    # Centroids
    centroids = torch.zeros(size=(n_centroids, 2))

    print('Computing initial centroids')
    for n in range(n_centroids):
        if init_type == KMeansInit.Forgy:
            centroids[n] = feature_matrix[random.randint(0, len(feature_matrix))][0:2]

        elif init_type == KMeansInit.RandomPartition:
            centroids[n] = torch.mean(feature_matrix[torch.where(labels == n + 1)[0], 0:2].to(torch.float64), dim=0)

    # Start KMeans loop
    converged = False
    while not converged:
        print('Assigning labels to each pixel')

        old_labels = torch.clone(labels)
        positions = feature_matrix[torch.where(feature_matrix[:, 5] > threshold)[0], 0:2]
        distances = torch.empty((n_centroids, positions.size(0)))
        for n in range(n_centroids):
            distances[n] = torch.sum((centroids[n] - positions) ** 2, dim=1)

        labels[torch.where(feature_matrix[:, 5] > threshold)[0]] = 1 + torch.argmin(distances, dim=0)

        # Check for convergence
        if torch.all(old_labels.eq(labels)):
            print('No new assignments. K means has converged.')
            converged = True
            break

        print('Computing new centroids')
        for n in range(n_centroids):
            centroids[n] = torch.mean(feature_matrix[torch.where(labels == n + 1)[0], 0:2].to(torch.float64), dim=0)

        print(f'Centroids: {centroids}')

    return labels











