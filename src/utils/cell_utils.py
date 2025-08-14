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

def estimate_background_offset_annulus(
    image: np.ndarray,
    center: tuple[float, float],
    r_inner: float = 3.0,
    r_outer: float | None = None,
    min_pixels: int = 200,
    drop_top_fraction: float | None = 0.20,
    sigma: float = 3.0,
    max_iters: int = 5,
) -> float:
    """
    Estimate a fixed background offset (b) using an annulus median around an emitter.

    Strategy:
      - Mask a central disk (radius r_inner) to avoid emitter contamination.
      - Use pixels in an outer annulus (r_inner..r_outer).
      - Optionally drop the top brightest fraction (e.g., 20%).
      - Apply iterative sigma-clipping around the median.
      - Return the median of the remaining pixels.

    Args:
        image: 2D array (patch containing the emitter).
        center: (x0, y0) center in image coordinates.
        r_inner: Inner radius of masked disk (px).
        r_outer: Outer radius of annulus (px). If None, chosen adaptively.
        min_pixels: Target minimum number of annulus pixels; r_outer expands up to fit.
        drop_top_fraction: If set, drop this fraction of brightest annulus pixels before clipping.
        sigma: Sigma threshold for clipping relative to robust MAD-based sigma.
        max_iters: Max iterations for sigma clipping.

    Returns:
        float: Estimated background offset (b).
    """
    if image.ndim != 2:
        raise ValueError("estimate_background_offset_annulus expects a 2D image.")
    h, w = image.shape
    x0, y0 = float(center[0]), float(center[1])

    # Build distance map
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    # Compute a reasonable maximum radius (stay within frame)
    to_left = x0
    to_right = w - 1 - x0
    to_top = y0
    to_bottom = h - 1 - y0
    r_max_edge = max(0.0, min(to_left, to_right, to_top, to_bottom))

    # If r_outer not provided, choose adaptively and expand until we have enough pixels
    if r_outer is None:
        r_outer = min(r_inner + 8.0, r_max_edge)

    # Annulus selection with adaptive expansion
    def annulus_vals(r_out: float) -> np.ndarray:
        mask = (dist >= r_inner) & (dist <= r_out)
        return image[mask].astype(np.float64, copy=False)

    vals = annulus_vals(r_outer)
    # Expand r_outer up to edge if not enough pixels
    while vals.size < min_pixels and r_outer < r_max_edge:
        r_outer = min(r_max_edge, r_outer * 1.25 + 1.0)
        vals = annulus_vals(r_outer)

    # Fallbacks if still empty
    if vals.size == 0:
        # As a last resort, use global median excluding center disk
        mask = dist >= r_inner
        fallback_vals = image[mask]
        if fallback_vals.size:
            return float(np.median(fallback_vals))
        return float(np.median(image))

    # Optionally drop the top brightest fraction to avoid residual hot pixels
    if drop_top_fraction is not None and 0.0 < drop_top_fraction < 1.0 and vals.size > 10:
        q = np.quantile(vals, 1.0 - drop_top_fraction)
        vals = vals[vals <= q]
        if vals.size == 0:
            # If everything got dropped, revert to pre-drop values
            vals = annulus_vals(r_outer)

    # Iterative sigma-clipping around the median using robust MAD
    clipped = vals
    for _ in range(max_iters):
        med = np.median(clipped)
        mad = np.median(np.abs(clipped - med))
        # Convert MAD to robust sigma estimate
        rob_sigma = 1.4826 * mad
        if rob_sigma <= 0:
            break
        keep = np.abs(clipped - med) <= (sigma * rob_sigma)
        if keep.all():
            break
        clipped = clipped[keep]
        if clipped.size == 0:
            # If over-clipped, use last median
            clipped = np.array([med], dtype=np.float64)
            break

    return float(np.median(clipped))

