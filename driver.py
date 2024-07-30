import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from python.kmeans import kmeans, KMeansInit, build_torch_image, build_feature_matrix


if __name__ == '__main__':

    # Settings
    K = 9
    img = 'imgs/test-img-7.png'
    T = 5
    centroid_init = KMeansInit.RandomPartition

    # Load Image and convert to tensor
    torch_img = pil_to_tensor(Image.open(img, 'r'))
    # torch_img = torch_img[:, 0:1024, 0:1024]

    # Image description
    w = torch_img.size(2)
    h = torch_img.size(1)
    n_pixels = w * h
    print(f'Image shape: {torch_img.shape}')
    print(f'N pixels: {n_pixels}')
    print(f'Min: {torch.min(torch_img)}', f'Max: {torch.max(torch_img)}')
    non_zero_alpha = torch.where(torch_img[3] > T)  # Ignore pixels with a small alpha
    print(f'Nonzero alpha: { round(len(non_zero_alpha[0])*100/(torch_img.size(1)*torch_img.size(2)), 2)} %')

    # Image inspection
    plt.figure()
    plt.imshow(torch_img.permute(1, 2, 0))
    plt.show()

    # Build a feature matrix from our torch image
    X = build_feature_matrix(torch_img)
    Y = kmeans(K, X)

    print('Building segmentated images')
    for k in range(K):
        cluster_matrix = X[torch.where(Y == k + 1)[0]]
        cluster_image = build_torch_image(cluster_matrix, w, h)

        pil_img = to_pil_image(cluster_image)
        pil_img.save(f'outs/cluster{k}.png')
