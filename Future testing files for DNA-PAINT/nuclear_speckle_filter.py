"""
Nuclear Speckle Filter Module
============================
Advanced filtering for DNA-PAINT localizations with cluster-aware quality control.

This module provides sophisticated filtering specifically designed for nuclear speckle
analysis, including quality metrics relevant to clustered molecules.

Key Features:
- Cluster-aware quality assessment
- Temporal consistency filtering for DNA-PAINT
- Photon count and precision filtering
- PSF shape analysis
- Spatial outlier detection
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
import h5py
from typing import List, Dict, Tuple, Optional
import warnings

class ClusterAwareFilter:
    """
    Advanced filter for nuclear speckle DNA-PAINT data.
    
    Performs multi-level quality control including individual localization
    quality and cluster coherence assessment.
    """
    
    def __init__(self,
                 min_photons: float = 50.0,
                 max_photons: float = 50000.0,
                 max_localization_precision: float = 20.0,  # nm
                 max_sigma: float = 3.0,  # pixels
                 min_sigma: float = 0.5,  # pixels
                 max_ellipticity: float = 2.0,
                 min_cluster_molecules: int = 2,
                 max_cluster_chi_squared: float = 10.0,
                 spatial_outlier_threshold: float = 3.0,
                 temporal_consistency_frames: int = 5,
                 pixel_size: float = 10.0):  # nm/pixel
        """
        Initialize the cluster-aware filter.
        
        Parameters:
        ----------- 
        min_photons : float
            Minimum photon count for valid localization
        max_photons : float
            Maximum photon count (to filter oversaturated spots)
        max_localization_precision : float
            Maximum localization precision in nanometers
        max_sigma : float
            Maximum PSF sigma in pixels
        min_sigma : float
            Minimum PSF sigma in pixels
        max_ellipticity : float
            Maximum PSF ellipticity (sigma_y/sigma_x)
        min_cluster_molecules : int
            Minimum molecules required for valid cluster
        max_cluster_chi_squared : float
            Maximum chi-squared for cluster fitting quality
        spatial_outlier_threshold : float
            Z-score threshold for spatial outlier detection
        temporal_consistency_frames : int
            Number of frames to check for temporal consistency
        pixel_size : float
            Pixel size in nanometers
        """
        self.min_photons = min_photons
        self.max_photons = max_photons
        self.max_localization_precision = max_localization_precision
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.max_ellipticity = max_ellipticity
        self.min_cluster_molecules = min_cluster_molecules
        self.max_cluster_chi_squared = max_cluster_chi_squared
        self.spatial_outlier_threshold = spatial_outlier_threshold
        self.temporal_consistency_frames = temporal_consistency_frames
        self.pixel_size = pixel_size
    
    def _calculate_quality_score(self, localization: Dict) -> float:
        """
        Calculate comprehensive quality score for a localization.
        
        Returns score between 0 (poor) and 1 (excellent).
        """
        score = 1.0
        
        # Photon count score
        photons = localization.get('photons', 0)
        if photons < self.min_photons:
            score *= 0.1
        elif photons > self.max_photons:
            score *= 0.5
        else:
            # Optimal photon range scoring
            normalized_photons = np.clip((photons - self.min_photons) / 
                                       (self.max_photons - self.min_photons), 0, 1)
            # Peak around 1000-5000 photons
            optimal_range = (1000 - self.min_photons) / (self.max_photons - self.min_photons)
            photon_score = 1.0 - abs(normalized_photons - optimal_range)
            score *= photon_score
        
        # Localization precision score
        precision = localization.get('localization_precision', 100.0)
        precision_score = max(0, 1.0 - precision / self.max_localization_precision)
        score *= precision_score
        
        # PSF shape score
        sigma_x = localization.get('sigma_x', 1.0)
        sigma_y = localization.get('sigma_y', 1.0)
        
        # Sigma range score
        avg_sigma = (sigma_x + sigma_y) / 2
        if self.min_sigma <= avg_sigma <= self.max_sigma:
            sigma_score = 1.0
        else:
            sigma_score = 0.3
        score *= sigma_score
        
        # Ellipticity score
        ellipticity = max(sigma_x, sigma_y) / min(sigma_x, sigma_y)
        if ellipticity <= self.max_ellipticity:
            ellipticity_score = 1.0 - (ellipticity - 1.0) / (self.max_ellipticity - 1.0)
        else:
            ellipticity_score = 0.1
        score *= ellipticity_score
        
        # Cluster quality score (if available)
        cluster_chi_squared = localization.get('cluster_chi_squared', 1.0)
        if cluster_chi_squared <= self.max_cluster_chi_squared:
            cluster_score = 1.0 - cluster_chi_squared / self.max_cluster_chi_squared
        else:
            cluster_score = 0.2
        score *= cluster_score
        
        return np.clip(score, 0.0, 1.0)
    
    def _filter_photons(self, localizations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Filter based on photon count."""
        passed = []
        stats = {'rejected_low_photons': 0, 'rejected_high_photons': 0}
        
        for loc in localizations:
            photons = loc.get('photons', 0)
            if photons < self.min_photons:
                stats['rejected_low_photons'] += 1
            elif photons > self.max_photons:
                stats['rejected_high_photons'] += 1
            else:
                passed.append(loc)
        
        return passed, stats
    
    def _filter_localization_precision(self, localizations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Filter based on localization precision."""
        passed = []
        stats = {'rejected_poor_precision': 0}
        
        for loc in localizations:
            precision = loc.get('localization_precision', 100.0)
            if precision <= self.max_localization_precision:
                passed.append(loc)
            else:
                stats['rejected_poor_precision'] += 1
        
        return passed, stats
    
    def _filter_psf_shape(self, localizations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Filter based on PSF shape parameters."""
        passed = []
        stats = {'rejected_psf_sigma': 0, 'rejected_ellipticity': 0}
        
        for loc in localizations:
            sigma_x = loc.get('sigma_x', 1.0)
            sigma_y = loc.get('sigma_y', 1.0)
            
            # Check sigma range
            avg_sigma = (sigma_x + sigma_y) / 2
            if not (self.min_sigma <= avg_sigma <= self.max_sigma):
                stats['rejected_psf_sigma'] += 1
                continue
            
            # Check ellipticity
            ellipticity = max(sigma_x, sigma_y) / min(sigma_x, sigma_y)
            if ellipticity > self.max_ellipticity:
                stats['rejected_ellipticity'] += 1
                continue
            
            passed.append(loc)
        
        return passed, stats
    
    def _filter_cluster_quality(self, localizations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Filter based on cluster fitting quality."""
        passed = []
        stats = {'rejected_cluster_fit': 0, 'rejected_small_cluster': 0}
        
        for loc in localizations:
            # Check cluster size
            cluster_n_molecules = loc.get('cluster_n_molecules', 1)
            if cluster_n_molecules < self.min_cluster_molecules:
                stats['rejected_small_cluster'] += 1
                continue
            
            # Check cluster fitting quality
            cluster_chi_squared = loc.get('cluster_chi_squared', 0.0)
            if cluster_chi_squared > self.max_cluster_chi_squared:
                stats['rejected_cluster_fit'] += 1
                continue
            
            passed.append(loc)
        
        return passed, stats
    
    def _detect_spatial_outliers(self, localizations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Detect and remove spatial outliers using local density analysis."""
        if len(localizations) < 10:
            return localizations, {'rejected_spatial_outliers': 0}
        
        # Extract coordinates
        coords = np.array([[loc['x'], loc['y']] for loc in localizations])
        
        # Calculate local density using k-nearest neighbors
        k = min(10, len(localizations) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Calculate mean distance to k neighbors (density metric)
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self (distance 0)
        
        # Identify outliers using z-score
        z_scores = np.abs(zscore(mean_distances))
        outlier_mask = z_scores > self.spatial_outlier_threshold
        
        # Filter out outliers
        passed = [loc for i, loc in enumerate(localizations) if not outlier_mask[i]]
        stats = {'rejected_spatial_outliers': np.sum(outlier_mask)}
        
        return passed, stats
    
    def _check_temporal_consistency(self, localizations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Check temporal consistency for DNA-PAINT blinking patterns.
        
        Removes localizations that appear in too many consecutive frames
        (likely artifacts) or have inconsistent blinking patterns.
        """
        if not localizations:
            return localizations, {'rejected_temporal_inconsistency': 0}
        
        # Group by spatial position (allowing for localization uncertainty)
        position_tolerance = 2.0  # pixels
        
        # Sort by frame
        sorted_locs = sorted(localizations, key=lambda x: x.get('frame', 0))
        
        passed = []
        rejected_count = 0
        
        i = 0
        while i < len(sorted_locs):
            current_loc = sorted_locs[i]
            current_frame = current_loc.get('frame', 0)
            current_pos = (current_loc['x'], current_loc['y'])
            
            # Find nearby localizations in subsequent frames
            consecutive_frames = [current_frame]
            j = i + 1
            
            while j < len(sorted_locs):
                next_loc = sorted_locs[j]
                next_frame = next_loc.get('frame', 0)
                next_pos = (next_loc['x'], next_loc['y'])
                
                # Check if frames are consecutive and positions are close
                if (next_frame == consecutive_frames[-1] + 1 and
                    np.sqrt((current_pos[0] - next_pos[0])**2 + 
                           (current_pos[1] - next_pos[1])**2) < position_tolerance):
                    consecutive_frames.append(next_frame)
                    j += 1
                else:
                    break
            
            # Apply temporal consistency rules
            if len(consecutive_frames) > self.temporal_consistency_frames:
                # Too many consecutive frames - likely artifact
                rejected_count += len(consecutive_frames)
                i = j
            else:
                # Accept this group of localizations
                for k in range(i, min(i + len(consecutive_frames), len(sorted_locs))):
                    passed.append(sorted_locs[k])
                i = j
        
        stats = {'rejected_temporal_inconsistency': rejected_count}
        return passed, stats
    
    def filter_localizations(self, localizations: List[Dict], 
                           verbose: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Apply comprehensive filtering to localization data.
        
        Parameters:
        -----------
        localizations : list
            Input localizations
        verbose : bool
            Print filtering statistics
            
        Returns:
        --------
        Tuple of (filtered_localizations, filter_statistics)
        """
        if not localizations:
            return [], {}
        
        initial_count = len(localizations)
        current_locs = localizations.copy()
        all_stats = {'initial_count': initial_count}
        
        # Apply filters sequentially
        filter_steps = [
            ('photons', self._filter_photons),
            ('precision', self._filter_localization_precision),
            ('psf_shape', self._filter_psf_shape),
            ('cluster_quality', self._filter_cluster_quality),
            ('spatial_outliers', self._detect_spatial_outliers),
            ('temporal_consistency', self._check_temporal_consistency)
        ]
        
        for step_name, filter_func in filter_steps:
            current_locs, step_stats = filter_func(current_locs)
            all_stats.update(step_stats)
            
            if verbose:
                remaining = len(current_locs)
                rejected = sum(step_stats.values()) if step_stats else 0
                print(f"After {step_name} filter: {remaining} localizations "
                      f"({rejected} rejected)")
        
        # Calculate quality scores for remaining localizations
        for loc in current_locs:
            loc['quality_score'] = self._calculate_quality_score(loc)
        
        # Final statistics
        all_stats.update({
            'final_count': len(current_locs),
            'total_rejected': initial_count - len(current_locs),
            'rejection_rate': (initial_count - len(current_locs)) / initial_count if initial_count > 0 else 0
        })
        
        if verbose:
            print(f"\nFiltering Summary:")
            print(f"Initial localizations: {initial_count}")
            print(f"Final localizations: {len(current_locs)}")
            print(f"Rejection rate: {all_stats['rejection_rate']:.1%}")
            
            if current_locs:
                quality_scores = [loc['quality_score'] for loc in current_locs]
                print(f"Mean quality score: {np.mean(quality_scores):.3f}")
                print(f"Quality score range: {np.min(quality_scores):.3f} - {np.max(quality_scores):.3f}")
        
        return current_locs, all_stats
    
    def save_filtered_data(self, localizations: List[Dict], filename: str, 
                          filter_stats: Dict = None):
        """
        Save filtered localizations to HDF5 file.
        
        Parameters:
        -----------
        localizations : list
            Filtered localizations
        filename : str
            Output filename
        filter_stats : dict, optional
            Filter statistics to include in metadata
        """
        if not localizations:
            print("No localizations to save")
            return
        
        n_locs = len(localizations)
        
        # Define extended dtype including quality information
        loc_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('photons', 'f4'),
            ('sigma_x', 'f4'), ('sigma_y', 'f4'),
            ('x_error', 'f4'), ('y_error', 'f4'),
            ('frame', 'i4'), ('cluster_id', 'i4'),
            ('cluster_n_molecules', 'i4'),
            ('cluster_chi_squared', 'f4'),
            ('cluster_r_squared', 'f4'),
            ('localization_precision', 'f4'),
            ('amplitude', 'f4'), ('background', 'f4'),
            ('quality_score', 'f4')
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
            f.attrs['filtering_method'] = 'cluster_aware_multi_criteria'
            
            # Add filter parameters
            filter_params = {
                'min_photons': self.min_photons,
                'max_photons': self.max_photons,
                'max_localization_precision': self.max_localization_precision,
                'max_sigma': self.max_sigma,
                'min_sigma': self.min_sigma,
                'max_ellipticity': self.max_ellipticity,
                'min_cluster_molecules': self.min_cluster_molecules,
                'max_cluster_chi_squared': self.max_cluster_chi_squared
            }
            
            for key, value in filter_params.items():
                f.attrs[f'filter_{key}'] = value
            
            # Add filter statistics
            if filter_stats:
                for key, value in filter_stats.items():
                    f.attrs[f'stats_{key}'] = value
            
            if n_locs > 0:
                f.attrs['n_frames'] = int(loc_array['frame'].max() + 1)
                f.attrs['photon_range'] = [float(loc_array['photons'].min()), 
                                         float(loc_array['photons'].max())]
                f.attrs['quality_score_range'] = [float(loc_array['quality_score'].min()),
                                                float(loc_array['quality_score'].max())]
        
        print(f"Saved {n_locs} filtered localizations to {filename}")
    
    def load_localizations_hdf5(self, filename: str) -> List[Dict]:
        """
        Load localizations from HDF5 file.
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        List of localization dictionaries
        """
        try:
            with h5py.File(filename, 'r') as f:
                loc_array = f['locs'][:]
                
                # Convert to list of dictionaries
                localizations = []
                for i in range(len(loc_array)):
                    loc = {}
                    for field_name in loc_array.dtype.names:
                        loc[field_name] = float(loc_array[i][field_name])
                    localizations.append(loc)
                
                print(f"Loaded {len(localizations)} localizations from {filename}")
                return localizations
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []

def test_filter():
    """Test function for the ClusterAwareFilter."""
    import numpy as np
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Generate test localizations with various quality levels
    n_locs = 1000
    localizations = []
    
    for i in range(n_locs):
        # Create localization with random properties
        loc = {
            'x': np.random.uniform(0, 100),
            'y': np.random.uniform(0, 100),
            'photons': np.random.lognormal(7, 1),  # Log-normal distribution
            'sigma_x': np.random.normal(1.2, 0.3),
            'sigma_y': np.random.normal(1.2, 0.3),
            'x_error': np.random.exponential(0.1),
            'y_error': np.random.exponential(0.1),
            'frame': np.random.randint(0, 100),
            'cluster_id': -1,
            'cluster_n_molecules': np.random.randint(1, 8),
            'cluster_chi_squared': np.random.exponential(2),
            'cluster_r_squared': np.random.uniform(0.5, 1.0),
            'amplitude': np.random.lognormal(5, 0.5),
            'background': np.random.normal(10, 2)
        }
        
        # Calculate derived values
        loc['localization_precision'] = np.sqrt(loc['x_error']**2 + loc['y_error']**2) * 10  # Convert to nm
        
        localizations.append(loc)
    
    # Test filter
    filter_obj = ClusterAwareFilter()
    
    print("Testing ClusterAwareFilter...")
    print(f"Input: {len(localizations)} localizations")
    
    filtered_locs, stats = filter_obj.filter_localizations(localizations, verbose=True)
    
    print(f"\nFilter test completed.")
    print(f"Output: {len(filtered_locs)} localizations")
    
    # Test save/load
    filter_obj.save_filtered_data(filtered_locs, '/home/scrapybara/test_filtered.hdf5', stats)
    loaded_locs = filter_obj.load_localizations_hdf5('/home/scrapybara/test_filtered.hdf5')
    
    print(f"Save/load test: {len(loaded_locs)} localizations loaded")
    
    return filtered_locs, stats

if __name__ == "__main__":
    # Run test
    test_filtered, test_stats = test_filter()
    print("Filter test completed successfully.")