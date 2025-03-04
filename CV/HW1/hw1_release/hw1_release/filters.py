import numpy as np
from scipy.signal import convolve2d

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    kernel = np.flip(np.flip(kernel, 0), 1)

    # looping over the pixels of the output image
    for m in range(Hi):
        for n in range(Wi):
            
            # aply the convolution operation
            for i in range(Hk):
                for j in range(Wk):

                    # calculate the position in the original image
                    img_x = m + i - Hk // 2
                    img_y = n + j - Wk // 2

                    # chek that we don't go out of the bounds of the input image
                    if 0 <= img_x < Hi and 0 <= img_y < Wi:
                        out[m, n] += image[img_x, img_y] * kernel[i, j]

                   
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    ### YOUR CODE HERE
    
    # create yhe output array with the padded shape and  fill it 3ith zeros
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))

    # copy the original image to the center of the padded array
    out[pad_height:H + pad_height, pad_width:W + pad_width] = image
  
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    ### YOUR CODE HERE
    kernel = np.flip(kernel)

    # calculate the amount of the needed padding 
    pad_h = Hk // 2
    pad_w = Wk // 2

    # add pading the image with zeros around the border
    padded_image = zero_pad(image, pad_h, pad_w)
    
    # initialize the empty output image
    out = np.empty((Hi, Wi)) 

    # loop over the pixeles in the original image
    for m in range(Hi):
        for n in range(Wi):
            # perform element wise multiplication and sum it for each pixel region
            #out[m, n] = np.sum(padded_image[m:m+Hk, n:n+Wk] * kernel)
            out[m, n] = np.dot(padded_image[m:m+Hk, n:n+Wk].ravel(), kernel.ravel())


    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    ### YOUR CODE HERE
    pad_h = Hk // 2
    pad_w = Wk // 2
    
    # flip the kernel for convolution (both horizontally and vertically)
    #kernel = np.flip(kernel)

    # add pading the image with zeros
    # padded_image = zero_pad(image, pad_h, pad_w)
    
    out = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    # initialize the output array
    # out = np.empty((Hi, Wi))

    # use np.tensordot for faster element wise convolution
    # for i in range(Hi):
    #     for j in range(Wi):
    #         out[i, j] = np.tensordot(padded_image[i:i + Hk, j:j + Wk], kernel)


    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    ### YOUR CODE HERE
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g_zero_mean = g - np.mean(g)
    # perform cross-correlation with the zero  mean kernel
    out = cross_correlation(f, g_zero_mean)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """



    ### YOUR CODE HERE
    out =None
    ### END YOUR CODE

    return out
