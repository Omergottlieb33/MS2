import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

def get_3d_bounding_box_corners(mask: np.ndarray) -> np.ndarray:
    """
    Calculates the 8 corners of the 3D bounding box for a given 3D mask.

    The mask is expected to be a 3D numpy array where non-zero values
    indicate the object's presence. The input array is assumed to have
    dimensions ordered as (z, y, x).

    Args:
        mask (np.ndarray): A 3D numpy array (z, y, x) representing the shape.

    Returns:
        np.ndarray: An array of shape (8, 3) containing the (x, y, z)
                    coordinates of the 8 corners of the bounding box.
                    Returns an empty array of shape (0, 3) if the mask is empty.
    """
    # Find the coordinates of all non-zero voxels.
    # np.where returns a tuple of arrays, one for each dimension (z, y, x).
    coords = np.where(mask > 0)
    if not coords[0].size:
        # No object found in the mask
        return np.empty((0, 3), dtype=int)

    z_coords, y_coords, x_coords = coords

    # Find the min and max for each dimension to define the bounding box
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    return z_min, y_min, x_min, z_max, y_max, x_max

def calculate_center_of_mass_3d(mask_3d):
    """
    Calculate the center of mass of a 3D binary mask.
    
    Args:
        mask_3d (np.ndarray): 3D binary mask where non-zero values represent the object
        
    Returns:
        tuple: (z_center, y_center, x_center) coordinates of the center of mass
        None: If the mask is empty
    """
    # Check if mask is empty
    if not np.any(mask_3d):
        return None
    
    # Get coordinates of all non-zero voxels
    coords = np.where(mask_3d > 0)
    z_coords, y_coords, x_coords = coords
    
    # Calculate center of mass
    z_center = np.mean(z_coords)
    y_center = np.mean(y_coords)
    x_center = np.mean(x_coords)
    
    return (x_center, y_center, z_center)

def filter_ransac_poly(points, degree=2, residual_threshold=2.0, max_trials=200, use_mad=True, mad_k=3):
    """
    The following function detects inliers and outliers in a set of 2D points using RANSAC with polynomial fitting.
    PolynomialFeatures lets a linear model fit a nonlinear (polynomial) trajectory x(t), y(t).
    * RANSACRegressor uses a linear estimator (LinearRegression). Without feature expansion it fits straight lines: x ≈ a0 + a1 t.
    * PolynomialFeatures builds [1, t, t^2, …, t^d], so the same linear estimator can fit curves: x ≈ a0 + a1 t + a2 t^2 + … (same for y). This captures smooth motion with acceleration/curvature.
    Args:
        points (np.ndarray): Array of shape (n, 2) containing 2D points.
        degree (int): Degree of the polynomial to fit (default: 2).
        residual_threshold (float): RANSAC residual threshold (default: 2.0).
        max_trials (int): Maximum number of RANSAC iterations (default: 200).
        use_mad (bool): Whether to use Median Absolute Deviation for outlier detection (default: True).
    Returns:
        inliers (np.ndarray): Array of inlier points.
        mask (np.ndarray): Boolean mask indicating inliers.
        info (dict): Dictionary with additional information (e.g., fitted models, residuals).
    """
    points = np.asarray(points)
    n = len(points)
    if n < (degree + 2):
        return points, np.ones(n, dtype=bool), {}

    t = np.arange(n).reshape(-1, 1)
    feats = PolynomialFeatures(degree=degree, include_bias=True)

    ransac_x = RANSACRegressor(
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=0
    )
    ransac_y = RANSACRegressor(
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=0
    )

    X = feats.fit_transform(t)
    ransac_x.fit(X, points[:, 0])
    ransac_y.fit(X, points[:, 1])

    x_pred = ransac_x.predict(X)
    y_pred = ransac_y.predict(X)

    residuals = np.sqrt((points[:,0] - x_pred)**2 + (points[:,1] - y_pred)**2)

    if use_mad:
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) + 1e-9
        thresh = med + mad_k * 1.4826 * mad
    else:
        thresh = residual_threshold

    mask = residuals <= thresh
    inliers = points[mask]
    return inliers, mask, {"x_model": ransac_x, "y_model": ransac_y, "residuals": residuals, "threshold": thresh}

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian function.

    Args:
        xy: Tuple of (x, y) meshgrid coordinates.
        amplitude: Amplitude of the Gaussian.
        x0: X-coordinate of the center.
        y0: Y-coordinate of the center.
        sigma_x: Standard deviation in the x-direction.
        sigma_y: Standard deviation in the y-direction.
        offset: Constant offset.

    Returns:
        Flattened 2D Gaussian values.
    """
    x, y = xy
    return (amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                                 ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset).ravel()


def estimate_emitter_2d_gaussian(image, initial_position, initial_sigma=1.0):
    """
    Estimates the parameters of a 2D Gaussian emitter in an image.

    Args:
        image (2D array): Input image containing the emitter.
        initial_position (tuple): Initial guess for the (x, y) position of the emitter.
        initial_sigma (float): Initial guess for the Gaussian sigma (default: 1.0).

    Returns:
        dict: Estimated parameters of the Gaussian (amplitude, x0, y0, sigma_x, sigma_y, offset).
    """
    # Create a meshgrid for the image
    y, x = np.indices(image.shape)
    if y.max() < initial_position[1]:
        initial_position = (initial_position[0], y.max())
    if x.max() < initial_position[0]:
        initial_position = (x.max(), initial_position[1])

    # Initial guesses for the parameters
    amplitude_guess = image.max() - image.min()
    x0_guess, y0_guess = initial_position
    offset_guess = image.min()
    initial_guess = (amplitude_guess, x0_guess, y0_guess,
                     initial_sigma, initial_sigma, offset_guess)

    # Define bounds for the parameters
    bounds = (
        (0, 0, 0, 0.5, 0.5, 0),  # Lower bounds
        (image.max(), image.shape[1], image.shape[0],
         3, 3, np.inf)  # Upper bounds
    )

    # Fit the 2D Gaussian model to the image
    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y),
                               image.ravel(), p0=initial_guess, bounds=bounds)
        params = {
            "amplitude": popt[0],
            "x0": popt[1],
            "y0": popt[2],
            "sigma_x": popt[3],
            "sigma_y": popt[4],
            "offset": popt[5]
        }
        return params, pcov
    except RuntimeError:
        print("Gaussian fitting failed.")
        return None, None
