"""
Nuclear Speckle Render Module
============================
Advanced visualization for clustered DNA-PAINT data in nuclear speckles.

This module provides comprehensive visualization capabilities specifically
designed for nuclear speckle analysis, including cluster-aware rendering
and publication-quality figures.

Key Features:
- Super-resolution image generation
- Cluster analysis visualization
- Temporal analysis plots
- Multi-channel integration displays
- Publication-ready figure generation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
import h5py
from typing import List, Dict, Tuple, Optional, Union
import warnings

class ClusterAwareRenderer:
    """
    Advanced renderer for nuclear speckle DNA-PAINT visualization.
    
    Creates publication-quality visualizations that highlight cluster
    structure and molecular organization within nuclear speckles.
    """
    
    def __init__(self,
                 pixel_size: float = 10.0,  # nm/pixel
                 sr_pixel_size: float = 2.0,  # nm/pixel for super-resolution
                 blur_sigma: float = 0.5,  # pixels for Gaussian blur
                 colormap: str = 'viridis',
                 figure_dpi: int = 150):
        """
        Initialize the cluster-aware renderer.
        
        Parameters:
        -----------
        pixel_size : float
            Original pixel size in nanometers
        sr_pixel_size : float
            Super-resolution pixel size in nanometers
        blur_sigma : float
            Gaussian blur sigma for rendering
        colormap : str
            Default colormap for visualizations
        figure_dpi : int
            DPI for saved figures
        """
        self.pixel_size = pixel_size
        self.sr_pixel_size = sr_pixel_size
        self.blur_sigma = blur_sigma
        self.colormap = colormap
        self.figure_dpi = figure_dpi
        
        # Calculate scaling factor
        self.scale_factor = pixel_size / sr_pixel_size
    
    def _calculate_image_bounds(self, localizations: List[Dict], 
                               padding: float = 5.0) -> Tuple[float, float, float, float]:
        """
        Calculate optimal image bounds for rendering.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
        padding : float
            Padding around data in pixels
            
        Returns:
        --------
        Tuple of (x_min, x_max, y_min, y_max) in original pixels
        """
        if not localizations:
            return 0, 100, 0, 100
        
        x_coords = [loc['x'] for loc in localizations]
        y_coords = [loc['y'] for loc in localizations]
        
        x_min = min(x_coords) - padding
        x_max = max(x_coords) + padding
        y_min = min(y_coords) - padding
        y_max = max(y_coords) + padding
        
        return x_min, x_max, y_min, y_max
    
    def render_super_resolution(self, localizations: List[Dict], 
                               image_bounds: Optional[Tuple[float, float, float, float]] = None,
                               weight_by_photons: bool = True,
                               weight_by_precision: bool = True) -> np.ndarray:
        """
        Render super-resolution image from localizations.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
        image_bounds : tuple, optional
            Image bounds (x_min, x_max, y_min, y_max). If None, auto-calculated
        weight_by_photons : bool
            Weight rendering by photon count
        weight_by_precision : bool
            Weight rendering by localization precision
            
        Returns:
        --------
        Super-resolution image array
        """
        if not localizations:
            return np.zeros((100, 100))
        
        # Calculate image bounds
        if image_bounds is None:
            x_min, x_max, y_min, y_max = self._calculate_image_bounds(localizations)
        else:
            x_min, x_max, y_min, y_max = image_bounds
        
        # Calculate super-resolution image size
        sr_width = int((x_max - x_min) * self.scale_factor)
        sr_height = int((y_max - y_min) * self.scale_factor)
        
        # Initialize super-resolution image
        sr_image = np.zeros((sr_height, sr_width))
        
        # Render each localization
        for loc in localizations:
            # Convert to super-resolution coordinates
            sr_x = (loc['x'] - x_min) * self.scale_factor
            sr_y = (loc['y'] - y_min) * self.scale_factor
            
            # Calculate weights
            weight = 1.0
            
            if weight_by_photons:
                photons = loc.get('photons', 1000)
                # Normalize photon weight (0.1 to 2.0 range)
                photon_weight = np.clip(photons / 1000.0, 0.1, 2.0)
                weight *= photon_weight
            
            if weight_by_precision:
                precision = loc.get('localization_precision', 10.0)  # nm
                # Better precision (lower value) gets higher weight
                precision_weight = np.clip(20.0 / max(precision, 1.0), 0.1, 3.0)
                weight *= precision_weight
            
            # Calculate Gaussian sigma for this localization
            sigma_x_sr = loc.get('sigma_x', 1.2) * self.scale_factor
            sigma_y_sr = loc.get('sigma_y', 1.2) * self.scale_factor
            avg_sigma = (sigma_x_sr + sigma_y_sr) / 2
            
            # Add Gaussian blob
            y_indices, x_indices = np.mgrid[0:sr_height, 0:sr_width]
            gaussian_blob = weight * np.exp(-((x_indices - sr_x)**2 / (2 * avg_sigma**2) + 
                                            (y_indices - sr_y)**2 / (2 * avg_sigma**2)))
            
            sr_image += gaussian_blob
        
        # Apply Gaussian blur for smoothing
        if self.blur_sigma > 0:
            sr_image = gaussian_filter(sr_image, sigma=self.blur_sigma * self.scale_factor)
        
        return sr_image
    
    def render_cluster_analysis(self, localizations: List[Dict], 
                               output_filename: str = None,
                               show_cluster_boundaries: bool = True,
                               show_cluster_labels: bool = True) -> plt.Figure:
        """
        Create comprehensive cluster analysis visualization.
        
        Parameters:
        -----------
        localizations : list
            Clustered localizations
        output_filename : str, optional
            Filename to save figure
        show_cluster_boundaries : bool
            Whether to show cluster boundaries
        show_cluster_labels : bool
            Whether to show cluster ID labels
            
        Returns:
        --------
        Matplotlib figure object
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Get cluster information
        cluster_ids = [loc.get('cluster_id', -1) for loc in localizations]
        unique_clusters = list(set(cluster_ids))
        unique_clusters = [cid for cid in unique_clusters if cid != -1]  # Remove noise
        
        x_coords = [loc['x'] for loc in localizations]
        y_coords = [loc['y'] for loc in localizations]
        photons = [loc.get('photons', 1000) for loc in localizations]
        
        # Define subplot layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Super-resolution image
        ax1 = fig.add_subplot(gs[0, :2])
        sr_image = self.render_super_resolution(localizations)
        im1 = ax1.imshow(sr_image, cmap=self.colormap, aspect='equal')
        ax1.set_title('Super-Resolution Image', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position (super-resolution pixels)')
        ax1.set_ylabel('Position (super-resolution pixels)')
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # 2. Cluster scatter plot
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Color map for clusters
        if unique_clusters:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
            cluster_colors = {cid: colors[i] for i, cid in enumerate(unique_clusters)}
            cluster_colors[-1] = 'gray'  # Noise color
        else:
            cluster_colors = {-1: 'gray'}
        
        for cluster_id in set(cluster_ids):
            cluster_mask = np.array(cluster_ids) == cluster_id
            cluster_x = np.array(x_coords)[cluster_mask]
            cluster_y = np.array(y_coords)[cluster_mask]
            cluster_photons = np.array(photons)[cluster_mask]
            
            if cluster_id == -1:
                label = 'Noise'
                alpha = 0.4
                s = 15
            else:
                label = f'Cluster {cluster_id}'
                alpha = 0.7
                s = 30
            
            scatter = ax2.scatter(cluster_x, cluster_y, 
                                c=[cluster_colors.get(cluster_id, 'gray')], 
                                alpha=alpha, s=s, label=label)
        
        ax2.set_title('Cluster Assignment', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add cluster boundaries if requested
        if show_cluster_boundaries and unique_clusters:
            for cluster_id in unique_clusters:
                cluster_mask = np.array(cluster_ids) == cluster_id
                if np.sum(cluster_mask) >= 3:  # Need at least 3 points for boundary
                    cluster_x = np.array(x_coords)[cluster_mask]
                    cluster_y = np.array(y_coords)[cluster_mask]
                    
                    # Calculate convex hull (simplified as bounding ellipse)
                    center_x, center_y = np.mean(cluster_x), np.mean(cluster_y)
                    std_x, std_y = np.std(cluster_x), np.std(cluster_y)
                    
                    ellipse = patches.Ellipse((center_x, center_y), 
                                            4*std_x, 4*std_y,
                                            fill=False, 
                                            edgecolor=cluster_colors[cluster_id],
                                            linewidth=2, alpha=0.6)
                    ax2.add_patch(ellipse)
                    
                    # Add cluster label
                    if show_cluster_labels:
                        ax2.text(center_x, center_y, str(cluster_id), 
                               fontsize=12, fontweight='bold',
                               ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.8))
        
        # 3. Cluster size distribution
        ax3 = fig.add_subplot(gs[1, 0])
        if unique_clusters:
            cluster_sizes = []
            for cluster_id in unique_clusters:
                cluster_size = sum(1 for cid in cluster_ids if cid == cluster_id)
                cluster_sizes.append(cluster_size)
            
            ax3.hist(cluster_sizes, bins=max(1, len(unique_clusters)//2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('Molecules per Cluster')
            ax3.set_ylabel('Number of Clusters')
            ax3.set_title('Cluster Size Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. Photon distribution by cluster
        ax4 = fig.add_subplot(gs[1, 1])
        if unique_clusters:
            cluster_photons_list = []
            cluster_labels_list = []
            
            for cluster_id in unique_clusters[:min(10, len(unique_clusters))]:  # Limit to 10 clusters
                cluster_mask = np.array(cluster_ids) == cluster_id
                cluster_photons = np.array(photons)[cluster_mask]
                cluster_photons_list.append(cluster_photons)
                cluster_labels_list.append(f'C{cluster_id}')
            
            if cluster_photons_list:
                ax4.boxplot(cluster_photons_list, labels=cluster_labels_list)
                ax4.set_xlabel('Cluster ID')
                ax4.set_ylabel('Photons')
                ax4.set_title('Photon Distribution by Cluster')
                ax4.grid(True, alpha=0.3)
        
        # 5. Quality score analysis
        ax5 = fig.add_subplot(gs[1, 2])
        quality_scores = [loc.get('quality_score', 0.5) for loc in localizations]
        ax5.hist(quality_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax5.set_xlabel('Quality Score')
        ax5.set_ylabel('Number of Localizations')
        ax5.set_title('Quality Score Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 6. Temporal analysis
        ax6 = fig.add_subplot(gs[1, 3])
        frames = [loc.get('frame', 0) for loc in localizations]
        frame_counts = {}
        for frame in frames:
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
        
        if frame_counts:
            sorted_frames = sorted(frame_counts.keys())
            counts = [frame_counts[f] for f in sorted_frames]
            ax6.plot(sorted_frames, counts, 'b-', alpha=0.7, linewidth=2)
            ax6.fill_between(sorted_frames, counts, alpha=0.3)
            ax6.set_xlabel('Frame Number')
            ax6.set_ylabel('Localizations per Frame')
            ax6.set_title('Temporal Distribution')
            ax6.grid(True, alpha=0.3)
        
        # 7. Cluster statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        if unique_clusters:
            # Calculate cluster statistics
            stats_data = []
            headers = ['Cluster ID', 'N Molecules', 'Mean Photons', 'Area (px²)', 'Density (mol/px²)']
            
            for cluster_id in sorted(unique_clusters):
                cluster_mask = np.array(cluster_ids) == cluster_id
                cluster_locs = [localizations[i] for i in range(len(localizations)) if cluster_mask[i]]
                
                n_molecules = len(cluster_locs)
                mean_photons = np.mean([loc.get('photons', 1000) for loc in cluster_locs])
                
                # Calculate area (simplified as bounding box)
                cluster_x = [loc['x'] for loc in cluster_locs]
                cluster_y = [loc['y'] for loc in cluster_locs]
                area = (max(cluster_x) - min(cluster_x)) * (max(cluster_y) - min(cluster_y))
                density = n_molecules / max(area, 1.0)
                
                stats_data.append([
                    cluster_id,
                    n_molecules,
                    f'{mean_photons:.0f}',
                    f'{area:.1f}',
                    f'{density:.3f}'
                ])
            
            # Create table
            table = ax7.table(cellText=stats_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add overall title
        fig.suptitle('Nuclear Speckle Cluster Analysis', fontsize=16, fontweight='bold')
        
        # Save figure if filename provided
        if output_filename:
            plt.savefig(output_filename, dpi=self.figure_dpi, bbox_inches='tight')
            print(f"Cluster analysis saved to {output_filename}")
        
        return fig
    
    def render_temporal_analysis(self, localizations: List[Dict], 
                                output_filename: str = None) -> plt.Figure:
        """
        Create temporal analysis visualization for DNA-PAINT data.
        
        Parameters:
        -----------
        localizations : list
            Localization data with frame information
        output_filename : str, optional
            Filename to save figure
            
        Returns:
        --------
        Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('DNA-PAINT Temporal Analysis', fontsize=16, fontweight='bold')
        
        # Extract temporal data
        frames = [loc.get('frame', 0) for loc in localizations]
        cluster_ids = [loc.get('cluster_id', -1) for loc in localizations]
        photons = [loc.get('photons', 1000) for loc in localizations]
        
        unique_clusters = list(set(cluster_ids))
        unique_clusters = [cid for cid in unique_clusters if cid != -1]
        
        # 1. Overall temporal distribution
        ax = axes[0, 0]
        frame_counts = {}
        for frame in frames:
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
        
        if frame_counts:
            sorted_frames = sorted(frame_counts.keys())
            counts = [frame_counts[f] for f in sorted_frames]
            ax.plot(sorted_frames, counts, 'b-', linewidth=2)
            ax.fill_between(sorted_frames, counts, alpha=0.3)
            
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Localizations per Frame')
        ax.set_title('Overall Temporal Distribution')
        ax.grid(True, alpha=0.3)
        
        # 2. Cluster temporal patterns
        ax = axes[0, 1]
        if unique_clusters:
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_clusters), 10)))
            
            for i, cluster_id in enumerate(unique_clusters[:10]):  # Limit to 10 clusters
                cluster_frames = [f for f, cid in zip(frames, cluster_ids) if cid == cluster_id]
                
                frame_counts_cluster = {}
                for frame in cluster_frames:
                    frame_counts_cluster[frame] = frame_counts_cluster.get(frame, 0) + 1
                
                if frame_counts_cluster:
                    sorted_frames_cluster = sorted(frame_counts_cluster.keys())
                    counts_cluster = [frame_counts_cluster[f] for f in sorted_frames_cluster]
                    ax.plot(sorted_frames_cluster, counts_cluster, 
                           color=colors[i], linewidth=1.5, alpha=0.7, 
                           label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Localizations per Frame')
        ax.set_title('Cluster Temporal Patterns')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. Frame-to-frame intervals
        ax = axes[0, 2]
        if len(frames) > 1:
            sorted_frames = sorted(frames)
            intervals = np.diff(sorted_frames)
            ax.hist(intervals, bins=30, alpha=0.7, color='orange', edgecolor='black')
            
        ax.set_xlabel('Frame Interval')
        ax.set_ylabel('Frequency')
        ax.set_title('Frame-to-Frame Intervals')
        ax.grid(True, alpha=0.3)
        
        # 4. Photon vs frame scatter
        ax = axes[1, 0]
        scatter = ax.scatter(frames, photons, alpha=0.6, c=photons, cmap='viridis', s=20)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Photons')
        ax.set_title('Photons vs Frame')
        plt.colorbar(scatter, ax=ax, label='Photons')
        ax.grid(True, alpha=0.3)
        
        # 5. Blinking density map
        ax = axes[1, 1]
        if frames and photons:
            # Create 2D histogram of frame vs photon
            H, xedges, yedges = np.histogram2d(frames, photons, bins=30)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(H.T, extent=extent, aspect='auto', origin='lower', cmap='hot')
            
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Photons')
        ax.set_title('Blinking Density Map')
        if 'im' in locals():
            plt.colorbar(im, ax=ax, label='Density')
        
        # 6. Cluster lifetime analysis
        ax = axes[1, 2]
        if unique_clusters:
            cluster_lifetimes = []
            
            for cluster_id in unique_clusters:
                cluster_frames = [f for f, cid in zip(frames, cluster_ids) if cid == cluster_id]
                if cluster_frames:
                    lifetime = max(cluster_frames) - min(cluster_frames) + 1
                    cluster_lifetimes.append(lifetime)
            
            if cluster_lifetimes:
                ax.hist(cluster_lifetimes, bins=max(1, len(cluster_lifetimes)//3), 
                       alpha=0.7, color='lightblue', edgecolor='black')
                
        ax.set_xlabel('Cluster Lifetime (frames)')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Cluster Lifetime Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if filename provided
        if output_filename:
            plt.savefig(output_filename, dpi=self.figure_dpi, bbox_inches='tight')
            print(f"Temporal analysis saved to {output_filename}")
        
        return fig
    
    def create_publication_figure(self, localizations: List[Dict],
                                 conventional_images: Dict[str, np.ndarray] = None,
                                 output_filename: str = None) -> plt.Figure:
        """
        Create publication-ready figure combining all analysis results.
        
        Parameters:
        -----------
        localizations : list
            Clustered localization data
        conventional_images : dict, optional
            Dictionary of conventional images (e.g., {'DAPI': array, 'Marker': array})
        output_filename : str, optional
            Filename to save figure
            
        Returns:
        --------
        Matplotlib figure object
        """
        # Determine figure layout based on available data
        if conventional_images:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Extract cluster information
        cluster_ids = [loc.get('cluster_id', -1) for loc in localizations]
        unique_clusters = [cid for cid in set(cluster_ids) if cid != -1]
        
        # Color scheme
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters))) if unique_clusters else []
        cluster_colors = {cid: colors[i] for i, cid in enumerate(unique_clusters)}
        cluster_colors[-1] = 'gray'
        
        current_col = 0
        
        # A. Conventional images (if provided)
        if conventional_images:
            for i, (channel_name, image) in enumerate(conventional_images.items()):
                if i < 2:  # Maximum 2 conventional channels
                    ax = fig.add_subplot(gs[0, i])
                    im = ax.imshow(image, cmap='gray', aspect='equal')
                    ax.set_title(f'{channel_name} Channel', fontsize=12, fontweight='bold')
                    ax.set_xlabel('X (pixels)')
                    ax.set_ylabel('Y (pixels)')
                    plt.colorbar(im, ax=ax, shrink=0.8)
            current_col = 2
        
        # B. Super-resolution image
        ax_sr = fig.add_subplot(gs[0, current_col])
        sr_image = self.render_super_resolution(localizations)
        im_sr = ax_sr.imshow(sr_image, cmap=self.colormap, aspect='equal')
        ax_sr.set_title('Super-Resolution\n(DNA-PAINT)', fontsize=12, fontweight='bold')
        ax_sr.set_xlabel('Position (SR pixels)')
        ax_sr.set_ylabel('Position (SR pixels)')
        plt.colorbar(im_sr, ax=ax_sr, shrink=0.8)
        
        # C. Clustered data
        if conventional_images:
            ax_cluster = fig.add_subplot(gs[0, 3])
        else:
            ax_cluster = fig.add_subplot(gs[0, 2])
        
        x_coords = [loc['x'] for loc in localizations]
        y_coords = [loc['y'] for loc in localizations]
        
        for cluster_id in set(cluster_ids):
            cluster_mask = np.array(cluster_ids) == cluster_id
            cluster_x = np.array(x_coords)[cluster_mask]
            cluster_y = np.array(y_coords)[cluster_mask]
            
            if cluster_id == -1:
                ax_cluster.scatter(cluster_x, cluster_y, c='gray', alpha=0.4, s=10, label='Noise')
            else:
                ax_cluster.scatter(cluster_x, cluster_y, 
                                 c=[cluster_colors[cluster_id]], 
                                 alpha=0.8, s=25, label=f'Cluster {cluster_id}')
        
        ax_cluster.set_title('Cluster Assignment', fontsize=12, fontweight='bold')
        ax_cluster.set_xlabel('X (pixels)')
        ax_cluster.set_ylabel('Y (pixels)')
        ax_cluster.grid(True, alpha=0.3)
        
        # D. Quantitative analysis plots
        
        # Cluster size distribution
        ax_size = fig.add_subplot(gs[1, 0])
        if unique_clusters:
            cluster_sizes = [sum(1 for cid in cluster_ids if cid == cluster_id) 
                           for cluster_id in unique_clusters]
            ax_size.bar(range(len(cluster_sizes)), cluster_sizes, 
                       color='skyblue', alpha=0.7, edgecolor='black')
            ax_size.set_xlabel('Cluster Index')
            ax_size.set_ylabel('Molecules per Cluster')
            ax_size.set_title('Cluster Sizes')
            ax_size.grid(True, alpha=0.3)
        
        # Photon distribution
        ax_photons = fig.add_subplot(gs[1, 1])
        photons = [loc.get('photons', 1000) for loc in localizations]
        ax_photons.hist(photons, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax_photons.set_xlabel('Photons')
        ax_photons.set_ylabel('Frequency')
        ax_photons.set_title('Photon Distribution')
        ax_photons.grid(True, alpha=0.3)
        
        # Quality scores
        ax_quality = fig.add_subplot(gs[1, 2])
        quality_scores = [loc.get('quality_score', 0.5) for loc in localizations]
        ax_quality.hist(quality_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax_quality.set_xlabel('Quality Score')
        ax_quality.set_ylabel('Frequency')
        ax_quality.set_title('Quality Distribution')
        ax_quality.grid(True, alpha=0.3)
        
        # Temporal analysis
        if conventional_images:
            ax_temporal = fig.add_subplot(gs[1, 3])
        else:
            ax_temporal = fig.add_subplot(gs[2, 0])
        
        frames = [loc.get('frame', 0) for loc in localizations]
        frame_counts = {}
        for frame in frames:
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
        
        if frame_counts:
            sorted_frames = sorted(frame_counts.keys())
            counts = [frame_counts[f] for f in sorted_frames]
            ax_temporal.plot(sorted_frames, counts, 'b-', linewidth=2)
            ax_temporal.fill_between(sorted_frames, counts, alpha=0.3)
        
        ax_temporal.set_xlabel('Frame Number')
        ax_temporal.set_ylabel('Localizations')
        ax_temporal.set_title('Temporal Distribution')
        ax_temporal.grid(True, alpha=0.3)
        
        # E. Summary statistics text
        if conventional_images:
            ax_stats = fig.add_subplot(gs[2, :])
        else:
            ax_stats = fig.add_subplot(gs[2, 1:])
        
        ax_stats.axis('off')
        
        # Calculate summary statistics
        n_total = len(localizations)
        n_clusters = len(unique_clusters)
        n_noise = sum(1 for cid in cluster_ids if cid == -1)
        
        if unique_clusters:
            cluster_sizes = [sum(1 for cid in cluster_ids if cid == cluster_id) 
                           for cluster_id in unique_clusters]
            mean_cluster_size = np.mean(cluster_sizes)
            total_photons = sum(photons)
            mean_photons = np.mean(photons)
            mean_quality = np.mean(quality_scores)
        else:
            mean_cluster_size = 0
            total_photons = sum(photons)
            mean_photons = np.mean(photons) if photons else 0
            mean_quality = np.mean(quality_scores) if quality_scores else 0
        
        stats_text = f"""
        NUCLEAR SPECKLE DNA-PAINT ANALYSIS SUMMARY
        
        Total Localizations: {n_total:,}
        Number of Clusters: {n_clusters}
        Noise Localizations: {n_noise} ({100*n_noise/max(n_total,1):.1f}%)
        
        Mean Cluster Size: {mean_cluster_size:.1f} molecules
        Total Photons: {total_photons:,.0f}
        Mean Photons per Localization: {mean_photons:.0f}
        Mean Quality Score: {mean_quality:.3f}
        
        Pixel Size: {self.pixel_size} nm
        Super-Resolution Pixel Size: {self.sr_pixel_size} nm
        """
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Add scale bar to super-resolution image
        if hasattr(self, 'scale_factor'):
            scale_bar_length_nm = 500  # 500 nm scale bar
            scale_bar_length_pixels = scale_bar_length_nm / self.sr_pixel_size
            
            # Add scale bar to bottom right of SR image
            sr_height, sr_width = sr_image.shape
            bar_x = sr_width - scale_bar_length_pixels - 20
            bar_y = sr_height - 30
            
            ax_sr.plot([bar_x, bar_x + scale_bar_length_pixels], [bar_y, bar_y], 
                      'white', linewidth=4)
            ax_sr.text(bar_x + scale_bar_length_pixels/2, bar_y - 10, 
                      f'{scale_bar_length_nm} nm', 
                      color='white', ha='center', fontweight='bold')
        
        # Overall title
        fig.suptitle('Nuclear Speckle DNA-PAINT Analysis', fontsize=18, fontweight='bold')
        
        # Save figure if filename provided
        if output_filename:
            plt.savefig(output_filename, dpi=self.figure_dpi, bbox_inches='tight')
            print(f"Publication figure saved to {output_filename}")
        
        return fig

def test_renderer():
    """Test function for the ClusterAwareRenderer."""
    import numpy as np
    
    # Create synthetic test data
    np.random.seed(42)
    localizations = []
    
    # Create clusters
    cluster_centers = [(30, 30), (70, 70), (50, 40)]
    cluster_id = 0
    
    for center_x, center_y in cluster_centers:
        n_molecules = np.random.randint(8, 20)
        
        for i in range(n_molecules):
            dx = np.random.normal(0, 2.5)
            dy = np.random.normal(0, 2.5)
            
            loc = {
                'x': center_x + dx,
                'y': center_y + dy,
                'photons': np.random.lognormal(7, 0.5),
                'sigma_x': np.random.normal(1.2, 0.2),
                'sigma_y': np.random.normal(1.2, 0.2),
                'x_error': np.random.exponential(0.1),
                'y_error': np.random.exponential(0.1),
                'frame': np.random.randint(0, 100),
                'cluster_id': cluster_id,
                'cluster_n_molecules': n_molecules,
                'cluster_chi_squared': np.random.exponential(1),
                'cluster_r_squared': np.random.uniform(0.7, 1.0),
                'amplitude': np.random.lognormal(5, 0.3),
                'background': np.random.normal(10, 2),
                'quality_score': np.random.uniform(0.6, 1.0),
                'localization_precision': np.random.uniform(5, 15)
            }
            
            localizations.append(loc)
        
        cluster_id += 1
    
    # Add some noise
    for i in range(15):
        loc = {
            'x': np.random.uniform(0, 100),
            'y': np.random.uniform(0, 100),
            'photons': np.random.lognormal(6, 0.8),
            'sigma_x': np.random.normal(1.2, 0.3),
            'sigma_y': np.random.normal(1.2, 0.3),
            'x_error': np.random.exponential(0.15),
            'y_error': np.random.exponential(0.15),
            'frame': np.random.randint(0, 100),
            'cluster_id': -1,  # Noise
            'cluster_n_molecules': 1,
            'cluster_chi_squared': np.random.exponential(3),
            'cluster_r_squared': np.random.uniform(0.2, 0.7),
            'amplitude': np.random.lognormal(4, 0.5),
            'background': np.random.normal(10, 3),
            'quality_score': np.random.uniform(0.2, 0.6),
            'localization_precision': np.random.uniform(8, 25)
        }
        
        localizations.append(loc)
    
    # Test renderer
    renderer = ClusterAwareRenderer(pixel_size=10.0, sr_pixel_size=2.0)
    
    print("Testing ClusterAwareRenderer...")
    print(f"Input: {len(localizations)} localizations")
    
    # Test cluster analysis
    fig1 = renderer.render_cluster_analysis(localizations, 
                                           '/home/scrapybara/test_cluster_analysis.png')
    plt.close(fig1)
    
    # Test temporal analysis  
    fig2 = renderer.render_temporal_analysis(localizations,
                                           '/home/scrapybara/test_temporal_analysis.png')
    plt.close(fig2)
    
    # Test publication figure
    fig3 = renderer.create_publication_figure(localizations,
                                            output_filename='/home/scrapybara/test_publication.png')
    plt.close(fig3)
    
    print("Renderer test completed. Check output files:")
    print("- test_cluster_analysis.png")
    print("- test_temporal_analysis.png")  
    print("- test_publication.png")
    
    return localizations

if __name__ == "__main__":
    # Run test
    test_localizations = test_renderer()
    print("Renderer test completed successfully.")