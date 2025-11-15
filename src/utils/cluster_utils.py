
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from src.utils.plot_utils import plot_single_gaussian, plot_multiple_gaussians


class SpaitalClustering:
    def __init__(self, sigma_rms_threshold=6.0, random_state=42, nsig=3.0, bic_max_components=5, min_cluster_size=2):
        self.sigma_rms_threshold = sigma_rms_threshold
        self.random_state = random_state
        self.nsig = nsig
        self.bic_max_components = bic_max_components
        self.min_cluster_size = min_cluster_size
        self.gmm1 = GaussianMixture(
            n_components=1, random_state=self.random_state)

    def __call__(self, df, outpud_dir, cell_id):
        self.collect_data(df)
        labels = self.gmm1.fit_predict(self.pts)
        means = self.gmm1.means_
        covs = self.gmm1.covariances_
        if np.sqrt(covs[0][0][0] + covs[0][1][1]) < self.sigma_rms_threshold:
            keep_mask = self._inlier_mask_mahalanobis(
                self.pts, labels, 0, means, covs, nsig=self.nsig)
            df_filtered = df[keep_mask].copy()
            output_path = f"{outpud_dir}/cell_{cell_id}_spatial_clustering.png"
            plot_single_gaussian(self.pts, self.intensity, keep_mask, means, covs, np.sqrt(
                covs[0][0][0] + covs[0][1][1]), output_path)
            return df_filtered
        else:
            n_components, best_gmm = self.get_best_gmm_components()
            labels = best_gmm.predict(self.pts)
            means_n = best_gmm.means_
            covs_n = best_gmm.covariances_
            sigma_rms_n = [np.sqrt(covs_n[i][0, 0] + covs_n[i][1, 1])
                           for i in range(n_components)]
            # TODO: add 3SD filtering here as well
            keep_mask, best_idx = self._select_cluster(labels, means_n, covs_n)
            df_filtered = df[keep_mask].copy()
            output_path = f"{outpud_dir}/cell_{cell_id}_spatial_clustering.png"
            if best_idx == -1:
                plot_multiple_gaussians(n_components, self.pts, self.intensity, means_n, covs_n, sigma_rms_n,
                                    cell_id, output_path)
            else:
                plot_single_gaussian(self.pts, self.intensity, keep_mask, means_n[best_idx], covs_n[best_idx], np.min(
                    sigma_rms_n), output_path)
            return df_filtered

    def collect_data(self, df):
        self.dist = df["dist_to_center"].to_numpy().astype(float)
        self.angle = df["angle_to_center"].to_numpy().astype(float)
        x = self.dist * np.cos(self.angle)
        y = self.dist * np.sin(self.angle)
        self.pts = np.column_stack([x, y])
        self.intensity = df["ellipse_sum"].to_numpy()

    def _inlier_mask_mahalanobis(self, pts, labels, label, means, covs, nsig=3.0, eps=1e-9):
        """
        Returns a boolean mask marking points that are within nsig Mahalanobis
        stds of their assigned GMM component mean.
        """
        K = len(means)
        keep = np.zeros(len(pts), dtype=bool)
        thr2 = float(nsig) ** 2  # squared threshold, e.g. 9 for 3σ
        idx = (labels == label)
        mu = means[label]
        cov_k = covs[label]
        if cov_k.ndim == 1:
            cov_k = np.diag(cov_k)
        # Regularize in case of near-singular covariance
        cov_k = cov_k + eps * np.eye(cov_k.shape[0])
        inv_cov = np.linalg.inv(cov_k)
        dx = pts[idx] - mu
        # Squared Mahalanobis distance for each point
        d2 = np.sum((dx @ inv_cov) * dx, axis=1)
        keep[idx] = d2 <= thr2
        return keep

    def get_best_gmm_components(self):
        range_n = range(
            2, self.bic_max_components+1) if len(self.pts) >= self.bic_max_components else range(2, len(self.pts))
        bics, aics, models, min_counts = [], [], [], []
        for n in range_n:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',   # 'full' is flexible; 'diag' if you want axis-aligned
                n_init=10,
                random_state=42,
                reg_covar=1e-6            # avoids singular covariances
            )
            gmm.fit(self.pts)
            models.append(gmm)
            bics.append(gmm.bic(self.pts))
            aics.append(gmm.aic(self.pts))
            counts = np.bincount(gmm.predict(self.pts), minlength=n)
            min_counts.append(counts.min())

        sigma_rms_models = []
        for gmm in models:
            covs_g = gmm.covariances_
            if covs_g.ndim == 3:           # full covariance (k, d, d)
                sigmas = np.sqrt(np.trace(covs_g, axis1=1, axis2=2))
            else:                          # diag form (k, d)
                sigmas = np.sqrt(np.sum(covs_g, axis=1))
            sigma_rms_models.append(sigmas.astype(
                float))   # ensure numeric dtype

        bics = np.array(bics)
        aics = np.array(aics)
        min_counts = np.array(min_counts)

        # Prefer the model with smallest BIC satisfying:
        # 1. No singleton clusters (min_counts >= min_cluster_size)
        # 2. At least one component with sigma_rms > sigma_rms_threshold
        order = np.argsort(bics)

        def passes_all(i):
            return (min_counts[i] >= self.min_cluster_size and
                    np.any(sigma_rms_models[i] < self.sigma_rms_threshold))
        # First try full criteria
        filtered = [i for i in order if passes_all(i)]

        if filtered:
            best_idx = filtered[0]
        else:
            # Relax: only min cluster size criterion
            filtered_size = [
                i for i in order if min_counts[i] >= self.min_cluster_size]
            if filtered_size:
                best_idx = filtered_size[0]
            else:
                # Final fallback: absolute best BIC
                best_idx = order[0]
        best_n = int(list(range_n)[best_idx])
        best_gmm = models[best_idx]
        return best_n, best_gmm

    def _select_cluster(self, labels, means, covs):
        """
        Return a boolean mask selecting the unique cluster whose sigma_rms < th and has > 2 points.
        If zero or more than one clusters satisfy the condition, return an all-False mask.

        Parameters
        ----------
        labels : (N,) array-like of int
            Cluster labels (0..K-1) for each point.
        covs : array
            GMM covariances. Shape (K, d, d) for 'full' or (K, d) for 'diag'.
        th : float
            Sigma-RMS threshold.
        min_points : int, optional
            Minimum number of points required in the cluster (default 3 == “> 2 points”).

        Returns
        -------
        mask : (N,) np.ndarray of bool
            True for points in the selected cluster, else False.
        """
        labels = np.asarray(labels)
        covs = np.asarray(covs)

        # Compute sigma_rms per component
        if covs.ndim == 3:            # full covariance (K, d, d)
            sigma_rms = np.sqrt(np.trace(covs, axis1=1, axis2=2))
        elif covs.ndim == 2:          # diag covariance (K, d)
            sigma_rms = np.sqrt(np.sum(covs, axis=1))
        else:
            raise ValueError(
                f"Unexpected covs shape {covs.shape}; expected (K,d,d) or (K,d)")

        K = sigma_rms.shape[0]
        counts = np.bincount(labels, minlength=K)

        # Eligible clusters: sigma_rms below threshold and enough points
        eligible = (sigma_rms < float(self.sigma_rms_threshold)) & (
            counts > int(self.min_cluster_size))
        eligible_idxs = np.where(eligible)[0]

        if eligible_idxs.size == 0:
            return np.zeros_like(labels, dtype=bool), -1  # No eligible clusters

        # If more than one eligible, choose the one with the smallest sigma_rms
        best_idx = eligible_idxs[np.argmin(sigma_rms[eligible_idxs])]
        keep = self._inlier_mask_mahalanobis(
            self.pts, labels, best_idx, means=means, covs=covs)
        return keep, best_idx
