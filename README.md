# Canny_Edge_Detection

This project implements the Canny edge detection algorithm to identify edges in an image. The Canny edge detection algorithm is a multi-step process that involves the following steps:

1. Noise reduction: The first step is to reduce noise in the image by applying a Gaussian blur.
2. Gradient calculation: Next, the gradient of the image is calculated using Sobel operators.
3. Non-maximum suppression: In this step, we suppress pixels that are not local maxima in the gradient direction.
4. Double threshold: In this step, we apply two thresholds (upper and lower) to the gradient magnitude image to identify strong and weak edges.
5. Edge tracking: Finally, we use hysteresis to track and connect strong edges to form a complete edge map.

## Example (Input-Output):
Input image:
![alt text](https://github.com/NimaVahdat/Canny_Edge_Detection/blob/main/Images/bowl-of-fruit.jpg?raw=true=250x250)
Output image:
![alt text](https://github.com/NimaVahdat/Canny_Edge_Detection/blob/main/Images/edges.png?raw=true=250x250)

To use this project, you will need to install the following dependencies:

- NumPy
- Pytorch
- OpenCV
- Matplotlib (optional, for visualization)
