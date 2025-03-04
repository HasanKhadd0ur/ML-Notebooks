import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from utils import pad, unpad
from utils import get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    
    dxx = dx ** 2
    dyy = dy ** 2
    dxdy= dx * dy
    
    #  gaussian_window = cv2.getGaussianKernel(ksize=3, sigma=1)
    #  gaussian_window = gaussian_window * gaussian_window.T  # 2D kernel

    mxx = convolve(dxx, window, mode='constant')
    myy = convolve(dyy, window, mode='constant')
    mxy = convolve(dxdy, window, mode='constant')

    # the deteermenent of the matrix 
    det = (mxx * myy - mxy ** 2) 
    # the trace of the matrix 
    trace = mxx + myy

    # R which is the Det minus the k * (trace)^2
    response = det - k * trace** 2

    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    ### YOUR CODE HERE
    # Calculate the mean of the patch
    mean = patch.mean()

    # Calculate the standard deviation of the patch
    std = patch.std()
    # To aVoid dividing on zero     
    std = 1 if std == 0  else std
    
    # Normalizing the patch to standard (0 ,1 )
    feature = ((patch - mean) / std)

    # re shape to 1d array 
    feature=feature.reshape(-1)
    
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
#    print(desc[0].shape)
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

#    print(dists[0])
    ### YOUR CODE HERE

#    print(desc1[0].shape)

    # Iterate through each row in the distance matrix 
    for i, Di in enumerate(dists):
        
        # Find the index of ther closeest descriptor in desc2 to the current descriptor
        matchj = np.argmin(Di)
        
        # Sorting  the distances to find the two smallest distances
        sortedDist = np.sort(Di)
        
        # Checkung if the ratio of the smallest distance to the second smallest distance is below the threshold
        if sortedDist[0] / sortedDist[1] < threshold:
            matches.append((i, matchj))
    
    matches = np.array(matches)
       ### END YOUR CODE
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    for _ in range(n_iters):

        # Cohse randomn sample
        indices = np.random.choice(N, n_samples, replace=False)
        
        # extract its key points 
        s1, s2 = matched1[indices], matched2[indices]
        
        # exstimate the affine matrix  
        H = np.linalg.lstsq(s2, s1, rcond=None)[0]
        
        # Add colum to make the matrix homogenous  
        H[:,2] = np.array([0, 0, 1])
       
        # SSD of the estimation of P2 to move it to P! 
        # in other word : 
        #    first we do (P2 . H) which is est to P1  => so we do Sum of  [  P2 . H - P1  ] ^ 2 
        #    then we root them 
        #    then we threshold them 
        curr_inliers = np.sqrt(((matched2.dot(H) - matched1) ** 2).sum(axis=-1)) < threshold
        
        # count the inliers 
        curr_n_inliers = curr_inliers.sum()

        # checking if the current model has an inlier count more than the old found one 
        if curr_n_inliers > n_inliers:
            max_inliers, n_inliers = curr_inliers, curr_n_inliers

        
    H = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers], rcond=None)[0]
    H[:,2] = np.array([0, 0, 1])

    ### END YOUR CODE
    return H, matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector

    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)

    Returns:
        block: 1D array of shape ((h*w*n_bins)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
   
    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    
    # 
    bin_assignments = theta_cells // degrees_per_bin
    
    # mask indicating if each angle is closer to the lower bin boundary
    is_closer_to_lower_bin = (theta_cells % degrees_per_bin <= degrees_per_bin / 2)


    # identify the lower and upper bins for each angle in the cell
    
    lower_bins = ((bin_assignments - is_closer_to_lower_bin) % n_bins).astype(int)
    
    upper_bins = ((lower_bins + 1) % n_bins).astype(int)
    
    # xalculate interpolation weights for each angle 
    #     to split the gradient magnitude between two bins
    lower_weights = ((degrees_per_bin / 2 - theta_cells % degrees_per_bin) % degrees_per_bin) / degrees_per_bin
    upper_weights = 1 - lower_weights

    indices = np.indices((rows, cols, pixels_per_cell[0], pixels_per_cell[1]))

    #  accumulate gradient magnitudes into the appropriate bins

    i, j = indices[0].reshape(-1), indices[1].reshape(-1)
    np.add.at(cells, (i, j, lower_bins.reshape(-1)), (lower_weights * G_cells).reshape(-1))
    np.add.at(cells, (i, j, upper_bins.reshape(-1)), (upper_weights * G_cells).reshape(-1))
    
    # flaten te blcok
    block = cells.reshape(-1)

    # normalize the block to make it invariant to lighting
    block /= np.linalg.norm(block)
    ### YOUR CODE HERE
    
    return block

import numpy as np

def image_blend(img1, img2):
    
    #Blends two images with linear blending in the overlapping area.
    
    
    
    # create a mask to detect non-zeropixels in each image
    mask1 = img1 > 0
    mask2 = img2 > 0

    # find the overlap area where both masks are true
    overlap_mask = mask1 & mask2

    # create a distance map for linear blending in the overlapping region
    # We assign weights that increase linearly 
    # from one side to the other across the overlap
    weight1 = np.zeros_like(img1, dtype=float)
    weight2 = np.zeros_like(img2, dtype=float)
    
   
    # find the indices of non zero values in the overlapping area
    y_indices, x_indices = np.nonzero(overlap_mask)

    # set weights that linearly decrease from the left edge to the right edge of the overlap
    for x in np.unique(x_indices):

        #the y of  over lapping regions 
        col_indices = np.where(x_indices == x)[0]

        # ؤalculate alpha  a weight factor that change linearly from 0 to 1 across the overlap
       
        alpha = (x - x_indices.min()) / (x_indices.max() - x_indices.min())
        
        # aply linear weights across each overlapping column
        weight1[y_indices[col_indices], x] = 1 - alpha
        weight2[y_indices[col_indices], x] = alpha

    # use weights to blend images
    blended_image = img1 * weight1 + img2 * weight2

    # jandle non-overlapping regions
    blended_image[mask1 & ~overlap_mask] = img1[mask1 & ~overlap_mask]
    blended_image[mask2 & ~overlap_mask] = img2[mask2 & ~overlap_mask]




    return blended_image , img1 * weight1 , img2 * weight2