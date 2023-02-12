import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

from PIL import Image
from matplotlib import pyplot as plt

from scipy.signal import wiener


# Open image file
image = Image.open("image.jpg")

#image.show()
blurred_image = Image.open("blurred_image.jpg")
blurred_image = blurred_image.save("blurred_image.jpg")
#blurred_image.show()
#plt.imshow(image)

# Convert image to numpy array
image_array = np.array(image)
gray_image = np.dot(blurred_image[...,:3], [0.2989, 0.5870, 0.1140]).round().astype(np.uint8)
# Print shape of array (height, width, number of channels)
print(gray_image.shape)
#print(image_array)







#general
def tik_fft(B, PSF, center, alpha=None):
    # Get the size of the image
    #np_array = np.array(B)
    m, n = np.shape(B)
    if alpha is None:
        # choose default value for alpha using generalized cross validation
        pass
    # Shift the PSF to the center
    PSF = np.roll(np.roll(PSF, -center[0], axis=0), -center[1], axis=1)
    # Compute the FFT of the PSF
    PSF_fft = fft2(PSF, (m, n))
    # Compute the FFT of the image
    B_fft = fft2(B)
    # Compute the FFT of the convolution
    conv_fft = B_fft * PSF_fft
    # Compute the regularization term
    regularization = alpha * PSF_fft
    # Compute the FFT of the deblurred image
    X_fft = np.conj(PSF_fft) / (np.abs(PSF_fft)**2 + alpha) * conv_fft
    # Compute the deblurred image
    X = np.real(ifft2(X_fft))
    return X, alpha





def get_psf(blurred_image, noise_level=0.01):
    """
    Function to estimate the PSF from a blurred image using Wiener Deconvolution
    
    Parameters:
    blurred_image (ndarray): The blurred image
    noise_level (float): The noise level of the blurred image (default: 0.01)
    
    Returns:
    ndarray: The estimated PSF
    """
    psf_estimate = wiener(blurred_image, noise=noise_level)
    return psf_estimate



def get_center_of_psf(psf):
    """
    Function to find the center of a PSF
    Parameters:
    psf (ndarray): The estimated PSF
    
    Returns:
    tuple: The indices of the center of the PSF (row, col)
    """
    center = np.array(psf.shape) / 2
    return tuple(center.round().astype(int))

PSF = get_psf(blurred_image, noise_level=0.01)
center = get_center_of_psf(PSF)
final_reult = tik_fft(blurred_image, PSF, center, alpha=None)
print(final_reult)

'''# Example usage
B = # The blurred image
PSF =get_psf(B, noise_level=0.01) # The point spread function
center = [row, col] # Indices of center of PSF
alpha = 0.2# The regularization parameter
X, alpha_used = tik_fft(B, PSF, center, alpha)


# with Periodic Boundary Conditions
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

def tik_fft(B, PSF, center, alpha=None):
    # Get the size of the image
    m, n = B.shape
    if alpha is None:
        # choose default value for alpha using generalized cross validation
        pass
    # Shift the PSF to the center
    PSF = np.roll(np.roll(PSF, -center[0], axis=0), -center[1], axis=1)
    # Pad the image and kernel with zeros
    B = np.pad(B, [(0, m), (0, n)], mode='wrap')
    PSF = np.pad(PSF, [(0, m), (0, n)], mode='wrap')
    # Compute the FFT of the PSF
    PSF_fft = fft2(PSF)
    # Compute the FFT of the image
    B_fft = fft2(B)
    # Compute the FFT of the convolution
    conv_fft = B_fft * PSF_fft
    # Compute the regularization term
    regularization = alpha * PSF_fft
    # Compute the FFT of the deblurred image
    X_fft = np.conj(PSF_fft) / (np.abs(PSF_fft)**2 + alpha) * conv_fft
    # Compute the deblurred image
    X = np.real(ifft2(X_fft))
    # Crop the image to remove the padding
    X = X[:m, :n]
    return X, alpha

# Example usage
B = # The blurred image
PSF = # The point spread function
center = [row, col] # Indices of center of PSF
alpha = # The regularization parameter
X, alpha_used = tik_fft(B, PSF, center, alpha)


 #with Reflexive Boundary Conditions

 import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

def tik_fft(B, PSF, center, alpha=None):
    # Get the size of the image
    m, n = B.shape
    if alpha is None:
        # choose default value for alpha using generalized cross validation
        pass
    # Shift the PSF to the center
    PSF = np.roll(np.roll(PSF, -center[0], axis=0), -center[1], axis=1)
    # Pad the image and kernel with a reflection of the image
    B = np.pad(B, [(0, m), (0, n)], mode='reflect')
    PSF = np.pad(PSF, [(0, m), (0, n)], mode='reflect')
    # Compute the FFT of the PSF
    PSF_fft = fft2(PSF)
    # Compute the FFT of the image
    B_fft = fft2(B)
    # Compute the FFT of the convolution
    conv_fft = B_fft * PSF_fft
    # Compute the regularization term
    regularization = alpha * PSF_fft
    # Compute the FFT of the deblurred image
    X_fft = np.conj(PSF_fft) / (np.abs(PSF_fft)**2 + alpha) * conv_fft
    # Compute the deblurred image
    X = np.real(ifft2(X_fft))
    # Crop the image to remove the padding
    X = X[:m, :n]
    return X, alpha

# Example usage
B = # The blurred image
PSF = # The point spread function
center = [row, col] # Indices of center of PSF
alpha = # The regularization parameter
X, alpha_used = tik_fft(B, PSF, center, alpha)
'''