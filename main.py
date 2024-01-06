import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

EPSILON = 1E-250

class gaussian2Dkernel:

    def __init__(self, regularization = 25., width=0, height=0):
        self.gamma = regularization
        self.width = width
        self.height = height
        self.n = width*height
        self.kernel1d = []
        for i in range(max(width, height)) :
            self.kernel1d.append(max(EPSILON, math.exp(-i*i / self.gamma)))
        self.kernel1d = np.array(self.kernel1d)


    def init_from_kernel(self, kernel):
        self.gamma = kernel.gamma_
        self.width = kernel.width_
        self.height = kernel.height_
        self.n = kernel.n_
        self.kernel1d = np.copy(kernel.kernel1d)

    def convolve(self, u):
        tmp = np.zeros((self.height, self.width))
        result = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                indices = np.abs(np.ones(self.width)*j - np.arange(self.width))
                # constructing the kernel from the "k" for loop from the template code
                arkernel = self.kernel1d[indices.astype(int)]
                # using numpy operations to make it a bit faster
                tmp[i, j] = np.sum(arkernel*u[i])

        for j in range(self.width):
            for i in range(self.height):
                # constructing the kernel from the "k" for loop from the template code
                indices = np.abs(np.ones(self.height)*i - np.arange(self.height))
                arkernel = self.kernel1d[indices.astype(int)]
                # using numpy operations to make it a bit faster
                result[i, j] = np.sum(arkernel * tmp[:, j])

        return result



nb_epochs = 30
image1 = cv2.imread('evol1.bmp', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('evol2.bmp', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('evol3.bmp', cv2.IMREAD_GRAYSCALE)
#vectorization
vimage1 = np.reshape(image1, (image1.shape[0]*image1.shape[1], 1))
vimage2 = np.reshape(image2, (image2.shape[0]*image2.shape[1], 1))
vimage3 = np.reshape(image3, (image3.shape[0]*image3.shape[1], 1))

#kernel init
kernel = gaussian2Dkernel(25., image1.shape[0], image1.shape[1]) #we suppose every image has the same size

#normalize
normimage1 = image1 / np.sum(vimage1)
normimage2 = image2 / np.sum(vimage2)
normimage3 = image3 / np.sum(vimage3)


#b init
b0 = np.ones(image1.shape)
b1 = np.ones(image1.shape)
b2 = np.ones(image1.shape)

#influence weights:
lambdas = [1, 1, 1]
lambdas = lambdas / np.sum(lambdas)

for i in range(nb_epochs):
    #a arrays (initially normalized images) divided by blurred b arrays (initially whites)
    a0 = normimage1 / np.maximum(kernel.convolve(b0), EPSILON)#making sure we don't divide by 0
    a1 = normimage2 / np.maximum(kernel.convolve(b1), EPSILON)
    a2 = normimage3 / np.maximum(kernel.convolve(b2), EPSILON)

    #convolutions of a arrays
    fa0 = np.array(kernel.convolve(a0))
    fa1 = np.array(kernel.convolve(a1))
    fa2 = np.array(kernel.convolve(a2))


    #fusing the 3 convolutions, with wheights, making sure result stays normalized
    normresult = np.power(fa0, lambdas[0]) * np.power(fa1, lambdas[1]) * np.power(fa2, lambdas[2])


    #updating b arrays (dividing the result by the blurred a arrays)
    b0 = np.divide(normresult, np.maximum(fa0, EPSILON))
    b1 = np.divide(normresult, np.maximum(fa1, EPSILON))
    b2 = np.divide(normresult, np.maximum(fa2, EPSILON))

    #denormalizing to make the image readable
    result = normresult * (lambdas[0] * np.sum(np.reshape(fa0, (fa0.shape[0]*fa0.shape[1], 1))) + lambdas[1] * np.sum(np.reshape(fa1, (fa1.shape[0]*fa1.shape[1], 1))) + lambdas[2] * np.sum(np.reshape(fa2, (fa2.shape[0]*fa2.shape[1], 1)) ))
    plt.imshow( result , cmap='gray')
    plt.show()









