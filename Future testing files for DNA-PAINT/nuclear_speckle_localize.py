"""
Nuclear Speckle Localization Module
==================================
Enhanced DNA-PAINT localization optimized for clustered molecules in nuclear speckles.

This module replaces traditional single-emitter localization with multi-emitter fitting
to handle dense molecular clusters typical of nuclear speckles.

Key Features:
- Multi-emitter Gaussian fitting for overlapping PSFs
- Adaptive clustering-aware detection
- Sub-pixel precision for dense environments
- Quality metrics for cluster coherence
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from skimage.feature import peak_local_maxima
from skimage.filters import gaussian
import warnings
from typing import List, Tuple, Dict, Optional
import h5py

class ClusterAwareLocalizer:
    """
    Advanced localizer designed for nuclear speckle DNA-PAINT analysis.
    
    Unlike traditional single-emitter localizers, this class can detect and fit
    multiple overlapping emitters within dense molecular clusters.
    """
    
    def __init__(self, 
                 pixel_size: float = 10.0,  # nm/pixel
                 detection_threshold: float = 5.0,
                 min_distance: int = 2,  # Reduced for cluster detection
                 max_cluster_size: int = 10,
                 fitting_window: int = 7,
                 sigma_range: Tuple[float, float] = (0.8, 3.0)):
        """
        Initialize the cluster-aware localizer.
        
        Parameters:
        -----------
        pixel_size : float
            Pixel size in nanometers
        detection_threshold : float
            Minimum signal-to-noise ratio for detection
        min_distance : int
            Minimum distance between peaks (pixels) - reduced for clusters
        max_cluster_size : int
            Maximum number of emitters to fit in a single cluster
        fitting_window : int
            Size of fitting window around detected clusters
        sigma_range : tuple
            Valid range for PSF sigma values (pixels)
        """
        self.pixel_size = pixel_size
        self.detection_threshold = detection_threshold
        self.min_distance = min_distance
        self.max_cluster_size = max_cluster_size
        self.fitting_window = fitting_window
        self.sigma_range = sigma_range
        
        # Pre-compute fitting grids
        self._setup_fitting_grids()
    
    def _setup_fitting_grids(self):
        """Pre-compute coordinate grids for fitting."""
        half_window = self.fitting_window // 2
        x = np.arange(-half_window, half_window + 1)
        y = np.arange(-half_window, half_window + 1)
        self.xx, self.yy = np.meshgrid(x, y)
    
    def _multi_gaussian_2d(self, coords, *params):
        """
        Multi-emitter 2D Gaussian model for fitting overlapping PSFs.
        
        Parameters format: [background, amp1, x1, y1, sigma1, amp2, x2, y2, sigma2, ...]
        """
        x, y = coords
        background = params[0]
        n_emitters = (len(params) - 1) // 4
        
        result = np.full_like(x, background)
        
        for i in range(n_emitters):
            base_idx = 1 + i * 4
            amp = params[base_idx]
            x0 = params[base_idx + 1]
            y0 = params[base_idx + 2]
            sigma = params[base_idx + 3]
            
            # 2D Gaussian
            gaussian = amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
            result += gaussian
            
        return result.ravel()
    
    def _estimate_background(self, image: np.ndarray) -> float:
        """Estimate background level using robust statistics."""
        # Use median of bottom 25% of pixels as background estimate
        sorted_pixels = np.sort(image.ravel())
        n_pixels = len(sorted_pixels)
        background = np.median(sorted_pixels[:n_pixels//4])
        return max(background, 1.0)  # Avoid zero background
    
    def _find_initial_peaks(self, image: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
        """
        Find initial peak positions using local maxima detection.
        
        Returns list of (y, x) coordinates.
        """
        # Enhance cluster regions with multi-scale filtering
        enhanced = self._enhance_clusters(image)
        
        # Find local maxima
        peaks = peak_local_maxima(
            enhanced,
            min_distance=self.min_distance,
            threshold_abs=threshold,
            exclude_border=self.fitting_window//2
        )
        
        return [(y, x) for y, x in peaks]
    
    def _enhance_clusters(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-scale enhancement to improve cluster detection.
        
        Uses difference of Gaussians to enhance cluster-like structures.
        """
        # Small scale (individual molecules)
        small_scale = gaussian(image, sigma=0.5)
        
        # Medium scale (small clusters)
        medium_scale = gaussian(image, sigma=1.5)
        
        # Large scale (background)
        large_scale = gaussian(image, sigma=3.0)
        
        # Difference of Gaussians to enhance cluster regions
        enhanced = medium_scale - large_scale + 0.5 * (small_scale - medium_scale)
        
        return np.maximum(enhanced, 0)
    
    def _group_nearby_peaks(self, peaks: List[Tuple[int, int]], 
                           max_distance: float = 5.0) -> List[List[Tuple[int, int]]]:
        """
        Group nearby peaks into potential clusters.
        
        Parameters:
        -----------
        peaks : list
            List of (y, x) peak coordinates
        max_distance : float
            Maximum distance to consider peaks as part of same cluster
            
        Returns:
        --------
        List of peak groups (clusters)
        """
        if len(peaks) < 2:
            return [[peak] for peak in peaks]
        
        # Calculate distance matrix
        coords = np.array(peaks)
        distances = squareform(pdist(coords))
        
        # Group peaks using simple clustering
        clusters = []
        visited = set()
        
        for i, peak in enumerate(peaks):
            if i in visited:
                continue
                
            # Start new cluster
            cluster = [peak]
            visited.add(i)
            
            # Find nearby peaks
            nearby = np.where(distances[i] <= max_distance)[0]
            for j in nearby:
                if j != i and j not in visited:
                    cluster.append(peaks[j])
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _fit_cluster(self, image: np.ndarray, peak_group: List[Tuple[int, int]]) -> Dict:
        """
        Fit multi-emitter Gaussian model to a cluster of peaks.
        
        Returns dictionary with fit results and quality metrics.
        """
        if not peak_group:
            return None
        
        # Determine fitting region
        y_coords = [p[0] for p in peak_group]
        x_coords = [p[1] for p in peak_group]
        
        y_center = int(np.mean(y_coords))
        x_center = int(np.mean(x_coords))
        
        half_window = self.fitting_window // 2
        
        y_min = max(0, y_center - half_window)
        y_max = min(image.shape[0], y_center + half_window + 1)
        x_min = max(0, x_center - half_window)
        x_max = min(image.shape[1], x_center + half_window + 1)
        
        # Extract fitting region
        fit_region = image[y_min:y_max, x_min:x_max]
        
        if fit_region.size == 0:
            return None
        
        # Create coordinate grids for fitting
        y_fit = np.arange(fit_region.shape[0])
        x_fit = np.arange(fit_region.shape[1])
        xx_fit, yy_fit = np.meshgrid(x_fit, y_fit)
        
        # Estimate background
        background = self._estimate_background(fit_region)
        
        # Limit number of emitters to avoid overfitting
        n_emitters = min(len(peak_group), self.max_cluster_size)
        peak_group = peak_group[:n_emitters]
        
        # Initial parameter guess
        initial_params = [background]
        bounds_lower = [0]
        bounds_upper = [np.inf]
        
        for y_peak, x_peak in peak_group:
            # Convert to local coordinates
            y_local = y_peak - y_min
            x_local = x_peak - x_min
            
            # Ensure coordinates are within fitting region
            y_local = np.clip(y_local, 0, fit_region.shape[0] - 1)
            x_local = np.clip(x_local, 0, fit_region.shape[1] - 1)
            
            # Initial amplitude estimate
            amp_estimate = max(fit_region[y_local, x_local] - background, 10.0)
            
            initial_params.extend([
                amp_estimate,           # amplitude
                x_local,               # x position
                y_local,               # y position  
                1.2                    # sigma
            ])
            
            # Parameter bounds
            bounds_lower.extend([10.0, -2.0, -2.0, self.sigma_range[0]])
            bounds_upper.extend([np.inf, fit_region.shape[1] + 2.0, 
                               fit_region.shape[0] + 2.0, self.sigma_range[1]])
        
        try:
            # Perform fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                popt, pcov = curve_fit(
                    lambda coords, *params: self._multi_gaussian_2d(coords, *params),
                    (xx_fit, yy_fit),
                    fit_region.ravel(),
                    p0=initial_params,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=1000
                )
            
            # Extract fitted parameters
            background_fit = popt[0]
            n_emitters_fit = (len(popt) - 1) // 4
            
            localizations = []
            
            for i in range(n_emitters_fit):
                base_idx = 1 + i * 4
                amp = popt[base_idx]
                x_local = popt[base_idx + 1]
                y_local = popt[base_idx + 2]
                sigma = popt[base_idx + 3]
                
                # Convert back to global coordinates
                x_global = x_local + x_min
                y_global = y_local + y_min
                
                # Estimate errors from covariance matrix
                if pcov is not None and not np.any(np.isinf(pcov)):
                    param_errors = np.sqrt(np.diag(pcov))
                    x_error = param_errors[base_idx + 1] if base_idx + 1 < len(param_errors) else 0.1
                    y_error = param_errors[base_idx + 2] if base_idx + 2 < len(param_errors) else 0.1
                else:
                    x_error = y_error = 0.1
                
                # Estimate photon count (simplified)
                photons = max(amp * 2 * np.pi * sigma**2, 50.0)
                
                localizations.append({
                    'x': x_global,
                    'y': y_global,
                    'photons': photons,
                    'sigma_x': sigma,
                    'sigma_y': sigma,
                    'x_error': x_error,
                    'y_error': y_error,
                    'amplitude': amp,
                    'background': background_fit
                })
            
            # Calculate cluster quality metrics
            fitted_image = self._multi_gaussian_2d((xx_fit, yy_fit), *popt).reshape(fit_region.shape)
            
            # Chi-squared goodness of fit
            residuals = fit_region - fitted_image
            chi_squared = np.sum(residuals**2) / (fit_region.size - len(popt))
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((fit_region - np.mean(fit_region))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'localizations': localizations,
                'n_emitters': n_emitters_fit,
                'chi_squared': chi_squared,
                'r_squared': r_squared,
                'background': background_fit,
                'fit_region_size': fit_region.size
            }
            
        except Exception as e:
            # Fitting failed, return simple centroid-based localization
            localizations = []
            for y_peak, x_peak in peak_group:
                # Simple intensity-weighted centroid
                y_min_simple = max(0, y_peak - 2)
                y_max_simple = min(image.shape[0], y_peak + 3)
                x_min_simple = max(0, x_peak - 2)
                x_max_simple = min(image.shape[1], x_peak + 3)
                
                region = image[y_min_simple:y_max_simple, x_min_simple:x_max_simple]
                
                if region.size > 0:
                    # Weighted centroid
                    y_indices, x_indices = np.mgrid[0:region.shape[0], 0:region.shape[1]]
                    total_intensity = np.sum(region)
                    
                    if total_intensity > 0:
                        y_centroid = np.sum(y_indices * region) / total_intensity + y_min_simple
                        x_centroid = np.sum(x_indices * region) / total_intensity + x_min_simple
                        photons = max(total_intensity - background * region.size, 50.0)
                        
                        localizations.append({
                            'x': x_centroid,
                            'y': y_centroid,
                            'photons': photons,
                            'sigma_x': 1.2,
                            'sigma_y': 1.2,
                            'x_error': 0.2,
                            'y_error': 0.2,
                            'amplitude': region.max(),
                            'background': background
                        })
            
            return {
                'localizations': localizations,
                'n_emitters': len(localizations),
                'chi_squared': 999.0,  # High value indicates poor fit
                'r_squared': 0.0,
                'background': background,
                'fit_region_size': 25
            }
    
    def localize_frame(self, image: np.ndarray, frame_number: int = 0) -> List[Dict]:
        """
        Localize molecules in a single frame using cluster-aware detection.
        
        Parameters:
        -----------
        image : ndarray
            Input image frame
        frame_number : int
            Frame number for tracking
            
        Returns:
        --------
        List of localization dictionaries
        """
        # Estimate noise level for threshold
        background = self._estimate_background(image)
        noise_std = np.std(image[image < np.percentile(image, 25)])
        threshold = background + self.detection_threshold * noise_std
        
        # Find initial peaks
        peaks = self._find_initial_peaks(image, threshold)
        
        if not peaks:
            return []
        
        # Group nearby peaks into clusters
        peak_groups = self._group_nearby_peaks(peaks, max_distance=self.fitting_window/2)
        
        # Fit each cluster
        all_localizations = []
        
        for group_idx, peak_group in enumerate(peak_groups):
            fit_result = self._fit_cluster(image, peak_group)
            
            if fit_result and fit_result['localizations']:
                # Add cluster information to each localization
                for loc in fit_result['localizations']:
                    loc.update({
                        'frame': frame_number,
                        'cluster_id': -1,  # Will be assigned later by clustering module
                        'cluster_n_molecules': fit_result['n_emitters'],
                        'cluster_chi_squared': fit_result['chi_squared'],
                        'cluster_r_squared': fit_result['r_squared'],
                        'localization_precision': np.sqrt(loc['x_error']**2 + loc['y_error']**2)
                    })
                
                all_localizations.extend(fit_result['localizations'])
        
        return all_localizations
    
    def localize_movie(self, movie: np.ndarray, 
                      progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Localize molecules in a complete DNA-PAINT movie.
        
        Parameters:
        -----------
        movie : ndarray
            Input movie (frames, height, width)
        progress_callback : callable, optional
            Function called with progress updates
            
        Returns:
        --------
        List of all localizations across all frames
        """
        all_localizations = []
        n_frames = movie.shape[0]
        
        for frame_idx in range(n_frames):
            frame_locs = self.localize_frame(movie[frame_idx], frame_idx)
            all_localizations.extend(frame_locs)
            
            if progress_callback:
                progress_callback(frame_idx + 1, n_frames)
        
        return all_localizations
    
    def save_localizations_hdf5(self, localizations: List[Dict], filename: str):
        """
        Save localizations to HDF5 file in Picasso-compatible format.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
        filename : str
            Output filename
        """
        if not localizations:
            print("No localizations to save")
            return
        
        # Convert to structured array
        n_locs = len(localizations)
        
        # Define dtype for localization data
        loc_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('photons', 'f4'),
            ('sigma_x', 'f4'), ('sigma_y', 'f4'),
            ('x_error', 'f4'), ('y_error', 'f4'),
            ('frame', 'i4'), ('cluster_id', 'i4'),
            ('cluster_n_molecules', 'i4'),
            ('cluster_chi_squared', 'f4'),
            ('cluster_r_squared', 'f4'),
            ('localization_precision', 'f4'),
            ('amplitude', 'f4'), ('background', 'f4')
        ]
        
        # Create structured array
        loc_array = np.zeros(n_locs, dtype=loc_dtype)
        
        for i, loc in enumerate(localizations):
            for field in loc_dtype:
                field_name = field[0]
                if field_name in loc:
                    loc_array[i][field_name] = loc[field_name]
        
        # Save to HDF5
        with h5py.File(filename, 'w') as f:
            f.create_dataset('locs', data=loc_array)
            
            # Add metadata
            f.attrs['n_localizations'] = n_locs
            f.attrs['pixel_size'] = self.pixel_size
            f.attrs['localization_method'] = 'cluster_aware_multi_emitter'
            
            if n_locs > 0:
                f.attrs['n_frames'] = int(loc_array['frame'].max() + 1)
                f.attrs['photon_range'] = [float(loc_array['photons'].min()), 
                                         float(loc_array['photons'].max())]
        
        print(f"Saved {n_locs} localizations to {filename}")

def test_localizer():
    """Test function for the ClusterAwareLocalizer."""
    import matplotlib.pyplot as plt
    
    # Create synthetic test data with clusters
    np.random.seed(42)
    image = np.random.poisson(5, (100, 100)).astype(float)
    
    # Add some clusters
    cluster_centers = [(25, 25), (75, 75), (50, 30)]
    for center in cluster_centers:
        y, x = center
        # Add multiple molecules in cluster
        for _ in range(5):
            dy = np.random.normal(0, 2)
            dx = np.random.normal(0, 2)
            yy = int(y + dy)
            xx = int(x + dx)
            if 0 <= yy < 100 and 0 <= xx < 100:
                # Add Gaussian blob
                for dy2 in range(-3, 4):
                    for dx2 in range(-3, 4):
                        if 0 <= yy+dy2 < 100 and 0 <= xx+dx2 < 100:
                            dist = np.sqrt(dy2**2 + dx2**2)
                            image[yy+dy2, xx+dx2] += 100 * np.exp(-dist**2 / 2)
    
    # Test localizer
    localizer = ClusterAwareLocalizer()
    localizations = localizer.localize_frame(image)
    
    print(f"Detected {len(localizations)} molecules")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image, cmap='hot')
    ax1.set_title('Input Image')
    
    ax2.imshow(image, cmap='gray', alpha=0.7)
    if localizations:
        x_coords = [loc['x'] for loc in localizations]
        y_coords = [loc['y'] for loc in localizations]
        photons = [loc['photons'] for loc in localizations]
        scatter = ax2.scatter(x_coords, y_coords, c=photons, cmap='viridis', s=30)
        plt.colorbar(scatter, ax=ax2, label='Photons')
    ax2.set_title('Detected Localizations')
    
    plt.tight_layout()
    plt.savefig('/home/scrapybara/test_localizer.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return localizations

if __name__ == "__main__":
    # Run test
    test_localizations = test_localizer()
    print("Test completed. Check test_localizer.png for results.")