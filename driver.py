import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


if __name__ == '__main__':

    K = 8
    img = Image.open('test-img-5.png', 'r')

    torch_img = pil_to_tensor(img)
    # torch_img = torch_img[:, 0:2048, 0:1024]
    n_pixels = torch_img.size(1) * torch_img.size(2)
    print(f'Image shape: {torch_img.shape}')
    print(f'N pixels: {n_pixels}')
    print(f'Min: {torch.min(torch_img)}', f'Max: {torch.max(torch_img)}' )

    plt.figure()
    plt.imshow(torch_img.permute(1, 2, 0))
    plt.show()

    threshold = 5
    non_zero_alpha = torch.where(torch_img[3] > threshold)  # Ignore pixels with a small alpha
    print(f'Nonzero alpha: { round(len(non_zero_alpha[0])*100/(torch_img.size(1)*torch_img.size(2)), 2)} %')

    print('Building feature matrix from selected image')
    feature_matrix = torch.zeros(size=(n_pixels, 6), dtype=torch.int64)  # 6 features: X, Y, R, G, B, A
    n = 0
    for y in range(torch_img.size(1)):
        for x in range(torch_img.size(2)):

            if n % 100000 == 0:
                print(f'Building ... {round(n*100/n_pixels, 2)} %')

            XY = torch.tensor([x, y])
            RGBA = torch_img[:, y, x]
            feature_matrix[n] = torch.cat((XY, RGBA))

            n += 1

    # TODO: Initialize random labels and set centroids as mean of labels
    clusters_x = torch.randint(low=0, high=torch_img.size(2), size=(K,))
    clusters_y = torch.randint(low=0, high=torch_img.size(1), size=(K,))

    # K clusters of dimensions 2 (position based only for now)
    centroids = torch.stack((clusters_x, clusters_y)).permute(dims=(1, 0)).to(torch.float64)

    labels = torch.zeros(size=(n_pixels,), dtype=torch.int64)

    converged = False
    while not converged:
        print('Assigning labels to each pixel')

        old_labels = torch.clone(labels)

        # Previous implementation START --------------------------------------------------------------------------------
        # for n in range(n_pixels):
        #
        #     if n % 100000 == 0:
        #         print(f'Assigning ... {round(n * 100 / n_pixels, 2)} %')
        #
        #     # Ignore pixels with small alpha
        #     if feature_matrix[n, 5] > threshold:
        #         position = feature_matrix[n, 0:2]
        #         distances = torch.sum((centroids - position) ** 2, dim=1)
        #         labels[n] = 1 + torch.argmin(distances)
        # Previous implementation END --------------------------------------------------------------------------------

        # New implementation START ------------------------- -----------------------------------------------------------
        positions = feature_matrix[torch.where(feature_matrix[:, 5] > threshold)[0], 0:2]
        distances = torch.empty((K, positions.size(0)))  # k, n,
        for k in range(K):
            distances[k] = torch.sum((centroids[k] - positions)**2, dim=1)

        labels[torch.where(feature_matrix[:, 5] > threshold)[0]] = 1 + torch.argmin(distances, dim=0)
        # New implementation END ---------------------------------------------------------------------------------------

        if torch.all(old_labels.eq(labels)):
            print('No new assignments. K means has converged.')
            converged = True
            break

        print('Computing new centroids')
        for k in range(K):
            centroids[k] = torch.mean(feature_matrix[torch.where(labels == k+1)[0], 0:2].to(torch.float64), dim=0)

        print(f'Centroids: {centroids}')

    print('Building segmentated images')
    for k in range(K):
        # build image
        cluster_img = torch.zeros(size=(4, torch_img.size(1), torch_img.size(2)), dtype=torch.int16)

        cluster = feature_matrix[torch.where(labels == k+1)[0]]
        for n in range(len(cluster)):

            if n % 100000 == 0:
                print(f'Building image {k+1}... {round(n*100/len(cluster), 2)} %')

            X = cluster[n, 0].to(torch.int16)
            Y = cluster[n, 1].to(torch.int16)
            R = cluster[n, 2]
            G = cluster[n, 3]
            B = cluster[n, 4]
            A = cluster[n, 5]
            cluster_img[0, Y, X] = R
            cluster_img[1, Y, X] = G
            cluster_img[2, Y, X] = B
            cluster_img[3, Y, X] = A

        plt.figure()
        plt.imshow(cluster_img.permute(1, 2, 0))
        plt.savefig(f'cluster{k+1}')








