import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import cv2


def open_image(name):
    img = cv2.imread(name, 0)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    return img


def kernel_gaussian(size, sigma=1):
    size = int(size) // 2
    x = torch.tensor(range(-size, size + 1))
    y = x.detach().clone()
    x_grid, y_grid = torch.meshgrid(x, y)
    g = torch.exp(-((x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))) * 1 / (2 * torch.pi * sigma ** 2)

    return g


def grad_cal(img):
    Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)[None, None, :]
    Ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)[None, None, :]

    Ix = F.conv2d(img[None, None, :], Kx, padding=1)
    Iy = F.conv2d(img[None, None, :], Ky, padding=1)

    G = torch.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = torch.atan2(Iy, Ix)

    return G, theta


def non_maximum(img, theta):
    x, y = img.shape
    result = torch.zeros((x, y), dtype=torch.float32)
    angle = theta * 180 / torch.pi
    angle[angle < 0] += 180

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            try:
                m = 255
                n = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    m = img[i, j + 1]
                    n = img[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    m = img[i + 1, j - 1]
                    n = img[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    m = img[i + 1, j]
                    n = img[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    m = img[i - 1, j - 1]
                    n = img[i + 1, j + 1]

                if (img[i, j] >= m) and (img[i, j] >= n):
                    result[i, j] = img[i, j]
                else:
                    result[i, j] = 0
            except IndexError as e:
                pass
    return result


def threshold(img, lowRatio, highRatio):
    highTreshold = img.max() * highRatio
    lowTreshold = highTreshold * lowRatio

    x, y = img.shape
    res = torch.zeros((x, y), dtype=torch.int32)

    strong_i, strong_j = torch.where(img >= highTreshold)
    zeros_i, zeros_j = torch.where(img < lowTreshold)

    weak_i, weak_j = torch.where((img <= highTreshold) & (img >= lowTreshold))

    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 25

    return res


def hysteresis(img, lowThreshold, highThreshold):
    x, y = img.shape
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            if img[i, j] == lowThreshold:
                try:
                    if (
                        (img[i + 1, j - 1] == highThreshold)
                        or (img[i + 1, j] == highThreshold)
                        or (img[i + 1, j + 1] == highThreshold)
                        or (img[i, j - 1] == highThreshold)
                        or (img[i, j + 1] == highThreshold)
                        or (img[i - 1, j - 1] == highThreshold)
                        or (img[i - 1, j] == highThreshold)
                        or (img[i - 1, j + 1] == highThreshold)
                    ):
                        img[i, j] = highThreshold
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def Canny_Edge_Detection(image_path, lowThreshold=25, highThreshold=250):
    img = open_image(image_path)
    ker = kernel_gaussian(5, 20)
    img = F.conv2d(img[None, :], ker[None, None, :])
    img, theta = grad_cal(torch.squeeze(img))
    img = non_maximum(torch.squeeze(img), torch.squeeze(theta))
    img = threshold(img, 0.045, 0.1)
    img = hysteresis(img, lowThreshold=lowThreshold, highThreshold=highThreshold)
    return img


if __name__ == "__main__":
    result = Canny_Edge_Detection("bowl-of-fruit.jpg", lowThreshold=25, highThreshold=250)
    plt.imshow(np.array(result), cmap="gray")
