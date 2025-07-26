"""
Nuclear Speckle Clustering Module
================================
Advanced clustering for DNA-PAINT localizations to identify nuclear speckle structures.

This module groups individual molecule localizations into biologically relevant
clusters representing nuclear speckles and their substructures.

Key Features:
- Density-based clustering (DBSCAN/HDBSCAN)
- Temporal clustering for DNA-PAINT consistency
- Hierarchical clustering for substructure analysis
- Adaptive parameters based on local density
- Cluster quality metrics and validation
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
import h5py
from typing import List, Dict, Tuple, Optional, Union
import warnings

class NuclearSpeckleClusterer:
    """
    Advanced clusterer for nuclear speckle DNA-PAINT data.
    
    Groups individual molecule localizations into clusters representing
    nuclear speckles and performs quality assessment of identified clusters.
    """
    
    def __init__(self,
                 eps_base: float = 5.0,  # pixels, base clustering distance
                 min_samples: int = 3,
                 max_cluster_size: int = 1000,
                 min_cluster_size: int = 2,
                 temporal_window: int = 10,  # frames
                 spatial_precision: float = 2.0,  # pixels
                 pixel_size: float = 10.0,  # nm/pixel
                 adaptive_eps: bool = True,
                 hierarchical_levels: int = 2):
        """
        Initialize the nuclear speckle clusterer.
        
        Parameters:
        -----------
        eps_base : float
            Base clustering distance in pixels
        min_samples : int
            Minimum samples for core point in DBSCAN
        max_cluster_size : int
            Maximum number of molecules per cluster
        min_cluster_size : int
            Minimum number of molecules per cluster
        temporal_window : int
            Frame window for temporal clustering
        spatial_precision : float
            Spatial precision for temporal clustering (pixels)
        pixel_size : float
            Pixel size in nanometers
        adaptive_eps : bool
            Whether to adapt eps based on local density
        hierarchical_levels : int
            Number of hierarchical clustering levels
        """
        self.eps_base = eps_base
        self.min_samples = min_samples
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.temporal_window = temporal_window
        self.spatial_precision = spatial_precision
        self.pixel_size = pixel_size
        self.adaptive_eps = adaptive_eps
        self.hierarchical_levels = hierarchical_levels
    
    def _estimate_local_density(self, coordinates: np.ndarray, 
                               k_neighbors: int = 10) -> np.ndarray:
        """
        Estimate local density around each point using k-nearest neighbors.
        
        Parameters:
        -----------
        coordinates : ndarray
            Array of (x, y) coordinates
        k_neighbors : int
            Number of neighbors to consider
            
        Returns:
        --------
        Array of local density estimates
        """
        if len(coordinates) < k_neighbors:
            return np.ones(len(coordinates))
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(coordinates)
        distances, _ = nbrs.kneighbors(coordinates)
        
        # Calculate local density as inverse of mean distance to k neighbors
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
        densities = 1.0 / (mean_distances + 1e-6)  # Add small constant to avoid division by zero
        
        return densities
    
    def _adaptive_eps_selection(self, coordinates: np.ndarray) -> float:
        """
        Adaptively select eps parameter based on local density distribution.
        
        Parameters:
        -----------
        coordinates : ndarray
            Array of (x, y) coordinates
            
        Returns:
        --------
        Optimized eps value
        """
        if len(coordinates) < 10:
            return self.eps_base
        
        # Calculate k-distance for eps estimation
        k = max(self.min_samples, 4)
        nbrs = NearestNeighbors(n_neighbors=k).fit(coordinates)
        distances, _ = nbrs.kneighbors(coordinates)
        k_distances = distances[:, k-1]  # Distance to k-th neighbor
        
        # Sort distances and look for knee point
        sorted_distances = np.sort(k_distances)
        
        # Use knee point detection (simplified)
        # Find point where curvature changes most dramatically
        if len(sorted_distances) > 20:
            # Calculate second derivative approximation
            second_deriv = np.diff(sorted_distances, 2)
            knee_idx = np.argmax(second_deriv) + 1
            knee_distance = sorted_distances[knee_idx]
            
            # Use knee distance but constrain to reasonable range
            adaptive_eps = np.clip(knee_distance, 
                                 self.eps_base * 0.5, 
                                 self.eps_base * 2.0)
        else:
            adaptive_eps = self.eps_base
        
        return adaptive_eps
    
    def _spatial_clustering(self, localizations: List[Dict]) -> np.ndarray:
        """
        Perform spatial clustering of localizations.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
            
        Returns:
        --------
        Array of cluster labels (-1 for noise)
        """
        if not localizations:
            return np.array([])
        
        # Extract coordinates
        coordinates = np.array([[loc['x'], loc['y']] for loc in localizations])
        
        # Determine eps parameter
        if self.adaptive_eps:
            eps = self._adaptive_eps_selection(coordinates)
        else:
            eps = self.eps_base
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(coordinates)
        
        return cluster_labels
    
    def _temporal_clustering(self, localizations: List[Dict]) -> Dict[int, List[int]]:
        """
        Group localizations by temporal consistency for DNA-PAINT validation.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
            
        Returns:
        --------
        Dictionary mapping temporal cluster ID to list of localization indices
        """
        if not localizations:
            return {}
        
        # Sort by frame
        frame_sorted_indices = sorted(range(len(localizations)), 
                                    key=lambda i: localizations[i].get('frame', 0))
        
        temporal_clusters = {}
        cluster_id = 0
        
        i = 0
        while i < len(frame_sorted_indices):
            current_idx = frame_sorted_indices[i]
            current_loc = localizations[current_idx]
            current_frame = current_loc.get('frame', 0)
            current_pos = np.array([current_loc['x'], current_loc['y']])
            
            # Start new temporal cluster
            temporal_cluster = [current_idx]
            
            # Look for spatially close localizations in nearby frames
            j = i + 1
            while j < len(frame_sorted_indices):
                next_idx = frame_sorted_indices[j]
                next_loc = localizations[next_idx]
                next_frame = next_loc.get('frame', 0)
                next_pos = np.array([next_loc['x'], next_loc['y']])
                
                # Check temporal and spatial proximity
                frame_diff = abs(next_frame - current_frame)
                spatial_dist = np.linalg.norm(next_pos - current_pos)
                
                if (frame_diff <= self.temporal_window and 
                    spatial_dist <= self.spatial_precision):
                    temporal_cluster.append(next_idx)
                    j += 1
                else:
                    break
            
            # Store temporal cluster if it has enough members
            if len(temporal_cluster) >= self.min_cluster_size:
                temporal_clusters[cluster_id] = temporal_cluster
                cluster_id += 1
            
            i = j if j > i + 1 else i + 1
        
        return temporal_clusters
    
    def _hierarchical_clustering(self, localizations: List[Dict], 
                                cluster_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Perform hierarchical clustering within each spatial cluster.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
        cluster_labels : ndarray
            Spatial cluster labels
            
        Returns:
        --------
        Dictionary mapping cluster ID to hierarchical sub-cluster labels
        """
        hierarchical_results = {}
        
        # Get unique cluster IDs (excluding noise = -1)
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            # Get localizations in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < self.min_cluster_size * 2:
                # Too small for hierarchical clustering
                hierarchical_results[cluster_id] = np.zeros(len(cluster_indices))
                continue
            
            # Extract coordinates for this cluster
            cluster_coords = np.array([[localizations[i]['x'], localizations[i]['y']] 
                                     for i in cluster_indices])
            
            # Perform hierarchical clustering
            n_sub_clusters = min(self.hierarchical_levels, 
                               len(cluster_indices) // self.min_cluster_size)
            
            if n_sub_clusters > 1:
                hierarchical = AgglomerativeClustering(n_clusters=n_sub_clusters)
                sub_cluster_labels = hierarchical.fit_predict(cluster_coords)
            else:
                sub_cluster_labels = np.zeros(len(cluster_indices))
            
            hierarchical_results[cluster_id] = sub_cluster_labels
        
        return hierarchical_results
    
    def _calculate_cluster_properties(self, localizations: List[Dict], 
                                    cluster_labels: np.ndarray) -> Dict[int, Dict]:
        """
        Calculate properties for each identified cluster.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
        cluster_labels : ndarray
            Cluster labels for each localization
            
        Returns:
        --------
        Dictionary of cluster properties
        """
        cluster_properties = {}
        
        # Get unique cluster IDs (excluding noise = -1)
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_locs = [localizations[i] for i in range(len(localizations)) 
                          if cluster_mask[i]]
            
            if not cluster_locs:
                continue
            
            # Basic properties
            n_molecules = len(cluster_locs)
            coordinates = np.array([[loc['x'], loc['y']] for loc in cluster_locs])
            
            # Spatial properties
            centroid_x = np.mean(coordinates[:, 0])
            centroid_y = np.mean(coordinates[:, 1])
            
            # Calculate cluster extent
            x_extent = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
            y_extent = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
            area_pixels = x_extent * y_extent
            area_nm2 = area_pixels * (self.pixel_size ** 2)
            
            # Calculate effective radius (assuming circular cluster)
            effective_radius_pixels = np.sqrt(area_pixels / np.pi)
            effective_radius_nm = effective_radius_pixels * self.pixel_size
            
            # Density calculations
            density_molecules_per_pixel2 = n_molecules / max(area_pixels, 1.0)
            density_molecules_per_um2 = density_molecules_per_pixel2 * (1000 / self.pixel_size) ** 2
            
            # Photon statistics
            photons = [loc.get('photons', 0) for loc in cluster_locs]
            total_photons = sum(photons)
            mean_photons = np.mean(photons)
            std_photons = np.std(photons)
            
            # Quality metrics
            quality_scores = [loc.get('quality_score', 0.5) for loc in cluster_locs]
            mean_quality = np.mean(quality_scores)
            
            # Temporal properties
            frames = [loc.get('frame', 0) for loc in cluster_locs]
            frame_span = max(frames) - min(frames) + 1
            temporal_density = n_molecules / max(frame_span, 1)
            
            # Localization precision
            precisions = [loc.get('localization_precision', 10.0) for loc in cluster_locs]
            mean_precision_nm = np.mean(precisions)
            
            # Cluster compactness (average distance from centroid)
            distances_from_centroid = np.sqrt((coordinates[:, 0] - centroid_x)**2 + 
                                            (coordinates[:, 1] - centroid_y)**2)
            compactness = np.mean(distances_from_centroid) * self.pixel_size  # in nm
            
            cluster_properties[cluster_id] = {
                'n_molecules': n_molecules,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'area_pixels': area_pixels,
                'area_nm2': area_nm2,
                'effective_radius_pixels': effective_radius_pixels,
                'effective_radius_nm': effective_radius_nm,
                'density_molecules_per_pixel2': density_molecules_per_pixel2,
                'density_molecules_per_um2': density_molecules_per_um2,
                'total_photons': total_photons,
                'mean_photons': mean_photons,
                'std_photons': std_photons,
                'mean_quality': mean_quality,
                'frame_span': frame_span,
                'temporal_density': temporal_density,
                'mean_precision_nm': mean_precision_nm,
                'compactness_nm': compactness,
                'x_extent': x_extent,
                'y_extent': y_extent
            }
        
        return cluster_properties
    
    def _validate_temporal_pattern(self, localizations: List[Dict], 
                                  cluster_labels: np.ndarray) -> np.ndarray:
        """
        Validate clusters based on DNA-PAINT temporal blinking patterns.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
        cluster_labels : ndarray
            Initial cluster labels
            
        Returns:
        --------
        Updated cluster labels with invalid clusters marked as noise (-1)
        """
        validated_labels = cluster_labels.copy()
        
        # Get unique cluster IDs (excluding noise = -1)
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_locs = [localizations[i] for i in range(len(localizations)) 
                          if cluster_mask[i]]
            
            if len(cluster_locs) < self.min_cluster_size:
                # Mark as noise
                validated_labels[cluster_mask] = -1
                continue
            
            # Check temporal pattern consistency
            frames = [loc.get('frame', 0) for loc in cluster_locs]
            frame_range = max(frames) - min(frames) + 1
            
            # DNA-PAINT should have sparse temporal pattern
            # Too many molecules in consecutive frames suggests artifact
            temporal_density = len(cluster_locs) / max(frame_range, 1)
            
            if temporal_density > 0.5:  # More than 50% frame occupancy is suspicious
                # Further validation needed
                frame_counts = {}
                for frame in frames:
                    frame_counts[frame] = frame_counts.get(frame, 0) + 1
                
                # Check for too many simultaneous detections
                max_simultaneous = max(frame_counts.values())
                if max_simultaneous > min(5, len(cluster_locs) // 2):
                    # Too many simultaneous detections - likely artifact
                    validated_labels[cluster_mask] = -1
        
        return validated_labels
    
    def cluster_nuclear_speckles(self, localizations: List[Dict], 
                                verbose: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Perform comprehensive clustering of nuclear speckle localizations.
        
        Parameters:
        -----------
        localizations : list
            Input localizations
        verbose : bool
            Print clustering statistics
            
        Returns:
        --------
        Tuple of (clustered_localizations, clustering_statistics)
        """
        if not localizations:
            return [], {}
        
        initial_count = len(localizations)
        
        if verbose:
            print(f"Clustering {initial_count} localizations...")
        
        # Step 1: Spatial clustering
        spatial_labels = self._spatial_clustering(localizations)
        
        # Step 2: Temporal validation
        validated_labels = self._validate_temporal_pattern(localizations, spatial_labels)
        
        # Step 3: Hierarchical sub-clustering
        hierarchical_results = self._hierarchical_clustering(localizations, validated_labels)
        
        # Step 4: Calculate cluster properties
        cluster_properties = self._calculate_cluster_properties(localizations, validated_labels)
        
        # Update localizations with cluster information
        clustered_localizations = []
        for i, loc in enumerate(localizations):
            new_loc = loc.copy()
            cluster_id = validated_labels[i]
            new_loc['cluster_id'] = int(cluster_id)
            
            # Add hierarchical sub-cluster information
            if cluster_id in hierarchical_results:
                sub_cluster_id = hierarchical_results[cluster_id][np.sum(validated_labels[:i] == cluster_id)]
                new_loc['sub_cluster_id'] = int(sub_cluster_id)
            else:
                new_loc['sub_cluster_id'] = 0
            
            clustered_localizations.append(new_loc)
        
        # Compile statistics
        n_clusters = len([cid for cid in np.unique(validated_labels) if cid != -1])
        n_noise = np.sum(validated_labels == -1)
        
        stats = {
            'initial_localizations': initial_count,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_fraction': n_noise / initial_count if initial_count > 0 else 0,
            'cluster_properties': cluster_properties,
            'clustering_parameters': {
                'eps_base': self.eps_base,
                'min_samples': self.min_samples,
                'adaptive_eps': self.adaptive_eps
            }
        }
        
        if verbose:
            print(f"Clustering completed:")
            print(f"  {n_clusters} clusters identified")
            print(f"  {n_noise} localizations classified as noise ({stats['noise_fraction']:.1%})")
            
            if cluster_properties:
                cluster_sizes = [props['n_molecules'] for props in cluster_properties.values()]
                print(f"  Cluster size range: {min(cluster_sizes)} - {max(cluster_sizes)} molecules")
                print(f"  Mean cluster size: {np.mean(cluster_sizes):.1f} molecules")
        
        return clustered_localizations, stats
    
    def save_clustered_data(self, localizations: List[Dict], filename: str, 
                           clustering_stats: Dict = None):
        """
        Save clustered localizations to HDF5 file.
        
        Parameters:
        -----------
        localizations : list
            Clustered localizations
        filename : str
            Output filename
        clustering_stats : dict, optional
            Clustering statistics to include in metadata
        """
        if not localizations:
            print("No localizations to save")
            return
        
        n_locs = len(localizations)
        
        # Define extended dtype including cluster information
        loc_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('photons', 'f4'),
            ('sigma_x', 'f4'), ('sigma_y', 'f4'),
            ('x_error', 'f4'), ('y_error', 'f4'),
            ('frame', 'i4'), ('cluster_id', 'i4'),
            ('sub_cluster_id', 'i4'),
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
            f.attrs['clustering_method'] = 'nuclear_speckle_dbscan_temporal'
            
            # Add clustering parameters
            clustering_params = {
                'eps_base': self.eps_base,
                'min_samples': self.min_samples,
                'max_cluster_size': self.max_cluster_size,
                'min_cluster_size': self.min_cluster_size,
                'temporal_window': self.temporal_window,
                'spatial_precision': self.spatial_precision,
                'adaptive_eps': self.adaptive_eps,
                'hierarchical_levels': self.hierarchical_levels
            }
            
            for key, value in clustering_params.items():
                f.attrs[f'cluster_{key}'] = value
            
            # Add clustering statistics
            if clustering_stats:
                for key, value in clustering_stats.items():
                    if isinstance(value, (int, float, str)):
                        f.attrs[f'stats_{key}'] = value
            
            if n_locs > 0:
                f.attrs['n_frames'] = int(loc_array['frame'].max() + 1)
                cluster_ids = loc_array['cluster_id']
                unique_clusters = np.unique(cluster_ids[cluster_ids != -1])
                f.attrs['n_clusters'] = len(unique_clusters)
        
        print(f"Saved {n_locs} clustered localizations to {filename}")

def test_clusterer():
    """Test function for the NuclearSpeckleClusterer."""
    import matplotlib.pyplot as plt
    
    # Create synthetic test data with clusters
    np.random.seed(42)
    localizations = []
    
    # Create several clusters
    cluster_centers = [(25, 25), (75, 75), (50, 30), (80, 40)]
    loc_id = 0
    
    for center_x, center_y in cluster_centers:
        # Add molecules around each center
        n_molecules = np.random.randint(5, 20)
        
        for i in range(n_molecules):
            # Random position around center
            dx = np.random.normal(0, 3)
            dy = np.random.normal(0, 3)
            
            loc = {
                'x': center_x + dx,
                'y': center_y + dy,
                'photons': np.random.lognormal(7, 0.5),
                'sigma_x': np.random.normal(1.2, 0.2),
                'sigma_y': np.random.normal(1.2, 0.2),
                'x_error': np.random.exponential(0.1),
                'y_error': np.random.exponential(0.1),
                'frame': np.random.randint(0, 100),
                'cluster_id': -1,
                'cluster_n_molecules': 1,
                'cluster_chi_squared': np.random.exponential(1),
                'cluster_r_squared': np.random.uniform(0.7, 1.0),
                'amplitude': np.random.lognormal(5, 0.3),
                'background': np.random.normal(10, 2),
                'quality_score': np.random.uniform(0.5, 1.0)
            }
            
            # Calculate derived values
            loc['localization_precision'] = np.sqrt(loc['x_error']**2 + loc['y_error']**2) * 10
            
            localizations.append(loc)
            loc_id += 1
    
    # Add some noise points
    for i in range(20):
        loc = {
            'x': np.random.uniform(0, 100),
            'y': np.random.uniform(0, 100),
            'photons': np.random.lognormal(6, 0.8),
            'sigma_x': np.random.normal(1.2, 0.3),
            'sigma_y': np.random.normal(1.2, 0.3),
            'x_error': np.random.exponential(0.15),
            'y_error': np.random.exponential(0.15),
            'frame': np.random.randint(0, 100),
            'cluster_id': -1,
            'cluster_n_molecules': 1,
            'cluster_chi_squared': np.random.exponential(2),
            'cluster_r_squared': np.random.uniform(0.3, 0.8),
            'amplitude': np.random.lognormal(4, 0.5),
            'background': np.random.normal(10, 3),
            'quality_score': np.random.uniform(0.2, 0.7)
        }
        
        loc['localization_precision'] = np.sqrt(loc['x_error']**2 + loc['y_error']**2) * 10
        localizations.append(loc)
    
    # Test clusterer
    clusterer = NuclearSpeckleClusterer(eps_base=8.0, min_samples=3)
    
    print("Testing NuclearSpeckleClusterer...")
    print(f"Input: {len(localizations)} localizations")
    
    clustered_locs, stats = clusterer.cluster_nuclear_speckles(localizations, verbose=True)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original data
    x_coords = [loc['x'] for loc in localizations]
    y_coords = [loc['y'] for loc in localizations]
    ax1.scatter(x_coords, y_coords, alpha=0.6, s=20)
    ax1.set_title('Original Localizations')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.grid(True, alpha=0.3)
    
    # Plot clustered data
    cluster_ids = [loc['cluster_id'] for loc in clustered_locs]
    unique_clusters = list(set(cluster_ids))
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = np.array(cluster_ids) == cluster_id
        cluster_x = np.array(x_coords)[cluster_mask]
        cluster_y = np.array(y_coords)[cluster_mask]
        
        if cluster_id == -1:
            # Noise points
            ax2.scatter(cluster_x, cluster_y, c='gray', alpha=0.5, s=15, label='Noise')
        else:
            ax2.scatter(cluster_x, cluster_y, c=[colors[i]], alpha=0.7, s=30, 
                       label=f'Cluster {cluster_id}')
    
    ax2.set_title('Clustered Localizations')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/scrapybara/test_clusterer.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test save/load
    clusterer.save_clustered_data(clustered_locs, '/home/scrapybara/test_clustered.hdf5', stats)
    
    print(f"Clustering test completed.")
    print(f"Output: {len(clustered_locs)} localizations")
    print(f"Check test_clusterer.png for visualization")
    
    return clustered_locs, stats

if __name__ == "__main__":
    # Run test
    test_clustered, test_stats = test_clusterer()
    print("Clustering test completed successfully.")