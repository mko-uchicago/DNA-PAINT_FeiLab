#!/usr/bin/env python3
"""
STORM Nuclear Speckle Analyzer - WITH SUPER-RESOLUTION RENDERING
================================================================
Complete analysis including Picasso-style super-resolution reconstruction.

Usage: python storm_superres_analyzer.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage import segmentation, measure, morphology, transform
from sklearn.cluster import DBSCAN

class SuperResolutionRenderer:
    """
    Picasso-style super-resolution image rendering from localizations.
    """
    
    def __init__(self, pixel_size=108.0, oversampling=5):
        """
        Initialize renderer.
        
        Parameters:
        -----------
        pixel_size : float
            Original pixel size in nanometers
        oversampling : int
            Super-resolution oversampling factor (5x = 5x higher resolution)
        """
        self.pixel_size = pixel_size
        self.oversampling = oversampling
        self.super_pixel_size = pixel_size / oversampling
    
    def render_gaussian(self, localizations, image_shape, mode='precision_weighted'):
        """
        Render super-resolution image using Gaussian rendering.
        
        Parameters:
        -----------
        localizations : list
            List of localization dictionaries
        image_shape : tuple
            Original image shape (height, width)
        mode : str
            'precision_weighted', 'photon_weighted', or 'histogram'
        """
        if not localizations:
            print("   ⚠️  No localizations to render")
            return np.zeros((image_shape[0] * self.oversampling, image_shape[1] * self.oversampling))
        
        print(f"🎨 Rendering super-resolution image...")
        print(f"   Mode: {mode}")
        print(f"   Oversampling: {self.oversampling}x")
        print(f"   Output size: {image_shape[0] * self.oversampling}×{image_shape[1] * self.oversampling}")
        
        # Create high-resolution grid
        super_height = image_shape[0] * self.oversampling
        super_width = image_shape[1] * self.oversampling
        super_image = np.zeros((super_height, super_width), dtype=np.float64)
        
        # Render each localization
        n_rendered = 0
        for i, loc in enumerate(localizations):
            if i % 1000 == 0:
                print(f"     Rendering localization {i}/{len(localizations)}")
            
            # Get coordinates in super-resolution space
            x_super = loc['x'] * self.oversampling
            y_super = loc['y'] * self.oversampling
            
            # Check bounds
            if (x_super < 0 or x_super >= super_width or 
                y_super < 0 or y_super >= super_height):
                continue
            
            # Determine Gaussian width based on mode
            if mode == 'precision_weighted':
                # Use localization precision if available
                precision = loc.get('x_error', 0.5)  # Default 0.5 pixels
                sigma_super = precision * self.oversampling
                sigma_super = max(sigma_super, 0.5)  # Minimum width
            elif mode == 'photon_weighted':
                # Width inversely related to photon count (more photons = tighter PSF)
                photons = loc.get('photons', 1000)
                sigma_super = self.oversampling / np.sqrt(photons / 1000)
                sigma_super = np.clip(sigma_super, 0.5, 3.0)
            else:  # histogram mode
                sigma_super = self.oversampling * 0.5  # Fixed narrow width
            
            # Determine intensity
            if mode == 'photon_weighted':
                intensity = loc.get('photons', 1000) / 1000.0
            else:
                intensity = 1.0
            
            # Render Gaussian
            self._add_gaussian(super_image, x_super, y_super, sigma_super, intensity)
            n_rendered += 1
        
        print(f"   ✅ Rendered {n_rendered}/{len(localizations)} localizations")
        
        # Normalize
        if super_image.max() > 0:
            super_image = super_image / super_image.max()
        
        return super_image
    
    def _add_gaussian(self, image, x_center, y_center, sigma, intensity):
        """Add a 2D Gaussian to the image."""
        # Define region of influence (3 sigma)
        radius = int(3 * sigma) + 1
        
        x_min = max(0, int(x_center - radius))
        x_max = min(image.shape[1], int(x_center + radius) + 1)
        y_min = max(0, int(y_center - radius))
        y_max = min(image.shape[0], int(y_center + radius) + 1)
        
        if x_min >= x_max or y_min >= y_max:
            return
        
        # Create coordinate grids
        x_coords = np.arange(x_min, x_max)
        y_coords = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Calculate Gaussian
        gaussian = intensity * np.exp(-((xx - x_center)**2 + (yy - y_center)**2) / (2 * sigma**2))
        
        # Add to image
        image[y_min:y_max, x_min:x_max] += gaussian
    
    def render_histogram(self, localizations, image_shape):
        """
        Simple histogram rendering (fast alternative).
        """
        print(f"🎨 Rendering histogram super-resolution...")
        
        super_height = image_shape[0] * self.oversampling
        super_width = image_shape[1] * self.oversampling
        super_image = np.zeros((super_height, super_width), dtype=np.float64)
        
        for loc in localizations:
            x_super = int(loc['x'] * self.oversampling)
            y_super = int(loc['y'] * self.oversampling)
            
            if (0 <= x_super < super_width and 0 <= y_super < super_height):
                super_image[y_super, x_super] += 1
        
        # Smooth slightly
        super_image = gaussian(super_image, sigma=0.5)
        
        # Normalize
        if super_image.max() > 0:
            super_image = super_image / super_image.max()
        
        return super_image

class DimensionFixedSTORMAnalyzer:
    """STORM analyzer with super-resolution rendering."""
    
    def __init__(self, pixel_size=108.0, oversampling=8, output_dir="./storm_superres_analysis"):
        self.pixel_size = pixel_size
        self.oversampling = oversampling
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize renderer
        self.renderer = SuperResolutionRenderer(pixel_size, oversampling)
    
    def load_and_resize_data(self, storm_movie_path, marker_path):
        """Load data and fix dimension mismatches."""
        print("🔧 DIMENSION-FIXED ANALYZER WITH SUPER-RESOLUTION")
        print("=" * 60)
        
        # Load STORM movie
        print(f"📁 Loading STORM movie: {storm_movie_path}")
        try:
            from skimage import io
            if storm_movie_path.endswith('.nd2'):
                try:
                    from nd2reader import ND2Reader
                    with ND2Reader(storm_movie_path) as images:
                        storm_movie = np.array([frame for frame in images])
                except ImportError:
                    import imageio
                    storm_movie = imageio.volread(storm_movie_path)
            else:
                storm_movie = io.imread(storm_movie_path)
            
            if storm_movie.ndim == 2:
                storm_movie = storm_movie[np.newaxis, :, :]
            
            storm_movie = storm_movie.astype(np.float32)
            storm_shape = storm_movie.shape[1:]
            print(f"   ✅ STORM: {storm_movie.shape[0]} frames, {storm_shape[0]}×{storm_shape[1]} pixels")
            
        except Exception as e:
            print(f"   ❌ Error loading STORM: {e}")
            return None, None
        
        # Load marker image
        print(f"📁 Loading marker image: {marker_path}")
        try:
            marker_image = io.imread(marker_path)
            if marker_image.ndim == 3:
                marker_image = marker_image[0]
            marker_image = marker_image.astype(np.float32)
            original_marker_shape = marker_image.shape
            print(f"   ✅ Marker: {original_marker_shape[0]}×{original_marker_shape[1]} pixels")
            
        except Exception as e:
            print(f"   ❌ Error loading marker: {e}")
            return None, None
        
        # Fix dimension mismatch
        if storm_shape != original_marker_shape:
            print(f"   🔧 Resizing marker from {original_marker_shape} to {storm_shape}...")
            marker_image = transform.resize(
                marker_image, storm_shape, preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            print(f"   ✅ Marker resized to: {marker_image.shape}")
        
        return storm_movie, marker_image
    
    def localize_molecules(self, storm_movie):
        """Enhanced localization with precision estimates."""
        print(f"\n🔍 STORM Localization with Precision Estimation...")
        
        all_localizations = []
        n_frames = min(storm_movie.shape[0], 1000)
        
        for frame_idx in range(n_frames):
            if frame_idx % 200 == 0:
                print(f"     Frame {frame_idx}/{n_frames}")
            
            frame = storm_movie[frame_idx]
            
            # Localization
            background = np.percentile(frame, 25)
            noise_std = np.std(frame[frame < np.percentile(frame, 25)])
            threshold = background + 5 * noise_std
            
            peaks = peak_local_max(frame, min_distance=3, threshold_abs=threshold, exclude_border=5)
            
            for y, x in peaks:
                # Extract region
                y_min, y_max = max(0, y-4), min(frame.shape[0], y+5)
                x_min, x_max = max(0, x-4), min(frame.shape[1], x+5)
                region = frame[y_min:y_max, x_min:x_max]
                
                if region.size == 0:
                    continue
                
                # Background subtraction
                local_bg = np.percentile(region, 25)
                corrected_region = region - local_bg
                corrected_region[corrected_region < 0] = 0
                
                total_intensity = np.sum(corrected_region)
                if total_intensity < 100:
                    continue
                
                # Centroid
                yy, xx = np.mgrid[0:corrected_region.shape[0], 0:corrected_region.shape[1]]
                y_centroid = np.sum(yy * corrected_region) / total_intensity + y_min
                x_centroid = np.sum(xx * corrected_region) / total_intensity + x_min
                
                # Estimate localization precision (Cramér-Rao lower bound approximation)
                # Precision ~ sigma / sqrt(N) where N is photon count
                estimated_sigma = 1.2  # Typical PSF width in pixels
                precision = estimated_sigma / np.sqrt(total_intensity / 100)  # Rough approximation
                precision = np.clip(precision, 0.05, 2.0)  # Reasonable bounds
                
                all_localizations.append({
                    'x': x_centroid,
                    'y': y_centroid,
                    'photons': total_intensity,
                    'frame': frame_idx,
                    'x_error': precision,
                    'y_error': precision,
                    'sigma_x': estimated_sigma,
                    'sigma_y': estimated_sigma,
                    'background': local_bg
                })
        
        print(f"   ✅ Detected {len(all_localizations)} molecules with precision estimates")
        return all_localizations
    
    def segment_speckles(self, marker_image):
        """Segment nuclear speckles."""
        print(f"\n🎯 Nuclear Speckle Segmentation...")
        
        smoothed = gaussian(marker_image, sigma=1.0)
        
        try:
            threshold = threshold_otsu(smoothed)
        except:
            threshold = np.percentile(smoothed, 85)
        
        binary = smoothed > threshold
        cleaned = morphology.remove_small_objects(binary, min_size=10)
        cleaned = ndimage.binary_fill_holes(cleaned)
        
        distance = ndimage.distance_transform_edt(cleaned)
        local_maxima = peak_local_max(distance, min_distance=3, threshold_abs=2)
        
        markers = np.zeros_like(distance, dtype=int)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        speckle_labels = segmentation.watershed(-distance, markers, mask=cleaned)
        
        regions = measure.regionprops(speckle_labels)
        speckle_info = []
        for region in regions:
            speckle_info.append({
                'label': region.label,
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox
            })
        
        print(f"   ✅ Found {len(speckle_info)} nuclear speckles")
        return speckle_labels, speckle_info
    
    def assign_molecules(self, localizations, speckle_labels):
        """Assign molecules to speckles."""
        print(f"\n🔗 Molecule Assignment...")
        
        if not localizations:
            return localizations, {}
        
        speckle_stats = {}
        unique_labels = np.unique(speckle_labels)
        
        for label in unique_labels:
            if label > 0:
                speckle_stats[label] = {'molecule_count': 0, 'total_photons': 0, 'molecules': []}
        
        assignment_counts = {'speckle': 0, 'background': 0, 'out_of_bounds': 0}
        
        for i, loc in enumerate(localizations):
            x_pixel = int(np.round(loc['x']))
            y_pixel = int(np.round(loc['y']))
            
            if (0 <= y_pixel < speckle_labels.shape[0] and 0 <= x_pixel < speckle_labels.shape[1]):
                speckle_label = speckle_labels[y_pixel, x_pixel]
                
                if speckle_label > 0:
                    loc['speckle_label'] = int(speckle_label)
                    loc['location'] = 'nuclear_speckle'
                    assignment_counts['speckle'] += 1
                    
                    speckle_stats[speckle_label]['molecule_count'] += 1
                    speckle_stats[speckle_label]['total_photons'] += loc['photons']
                    speckle_stats[speckle_label]['molecules'].append(i)
                else:
                    loc['speckle_label'] = 0
                    loc['location'] = 'background'
                    assignment_counts['background'] += 1
            else:
                loc['speckle_label'] = -1
                loc['location'] = 'out_of_bounds'
                assignment_counts['out_of_bounds'] += 1
        
        total = len(localizations)
        enrichment = assignment_counts['speckle'] / max(assignment_counts['background'], 1)
        
        print(f"   ✅ Speckle: {assignment_counts['speckle']} ({assignment_counts['speckle']/total*100:.1f}%)")
        print(f"   📈 Enrichment: {enrichment:.2f}x")
        
        speckle_stats['summary'] = {
            'total_molecules': total,
            'speckle_molecules': assignment_counts['speckle'],
            'background_molecules': assignment_counts['background'],
            'enrichment_factor': enrichment
        }
        
        return localizations, speckle_stats
    
    def create_super_resolution_images(self, localizations, image_shape, base_filename):
        """Create multiple super-resolution renderings."""
        print(f"\n🎨 Creating Super-Resolution Images...")
        
        # Create different rendering modes
        renderings = {}
        
        # 1. Precision-weighted Gaussian
        print("   Rendering precision-weighted Gaussian...")
        renderings['precision'] = self.renderer.render_gaussian(
            localizations, image_shape, mode='precision_weighted'
        )
        
        # 2. Photon-weighted Gaussian  
        print("   Rendering photon-weighted Gaussian...")
        renderings['photon'] = self.renderer.render_gaussian(
            localizations, image_shape, mode='photon_weighted'
        )
        
        # 3. Simple histogram
        print("   Rendering histogram...")
        renderings['histogram'] = self.renderer.render_histogram(
            localizations, image_shape
        )
        
        # Save individual super-resolution images
        for mode, sr_image in renderings.items():
            # Save as TIFF
            sr_file = os.path.join(self.output_dir, f"{base_filename}_superres_{mode}.tif")
            from skimage import io
            io.imsave(sr_file, (sr_image * 65535).astype(np.uint16))
            print(f"   💾 Saved {mode}: {sr_file}")
            
            # Save as high-quality PNG
            png_file = os.path.join(self.output_dir, f"{base_filename}_superres_{mode}.png")
            plt.figure(figsize=(10, 10))
            plt.imshow(sr_image, cmap='hot', interpolation='nearest')
            plt.title(f'Super-Resolution ({mode.capitalize()})\n{sr_image.shape[0]}×{sr_image.shape[1]} pixels, {self.oversampling}x oversampling', 
                     fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        return renderings
    
    def create_comprehensive_visualization(self, localizations, speckle_labels, marker_image, 
                                         speckle_stats, super_res_images, base_filename):
        """Create comprehensive visualization including super-resolution."""
        print(f"\n📊 Creating comprehensive visualization...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Layout: 3 rows, 3 columns
        
        # Row 1: Original data
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(marker_image, cmap='viridis')
        ax1.contour(speckle_labels, colors='white', linewidths=1)
        ax1.set_title('Marker + Segmentation', fontweight='bold')
        
        ax2 = plt.subplot(3, 3, 2)
        ax2.imshow(marker_image, cmap='gray', alpha=0.7)
        if localizations:
            x_coords = [loc['x'] for loc in localizations]
            y_coords = [loc['y'] for loc in localizations]
            ax2.scatter(x_coords, y_coords, c='red', s=1, alpha=0.7)
        ax2.set_title('STORM Localizations', fontweight='bold')
        
        ax3 = plt.subplot(3, 3, 3)
        summary = speckle_stats['summary']
        categories = ['Background', 'Nuclear Speckles']
        counts = [summary['background_molecules'], summary['speckle_molecules']]
        bars = ax3.bar(categories, counts, color=['lightblue', 'red'], alpha=0.7)
        ax3.set_title(f'Enrichment: {summary["enrichment_factor"]:.2f}x', fontweight='bold')
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Row 2: Super-resolution images
        ax4 = plt.subplot(3, 3, 4)
        ax4.imshow(super_res_images['precision'], cmap='hot')
        ax4.set_title('Precision-Weighted\nSuper-Resolution', fontweight='bold')
        ax4.axis('off')
        
        ax5 = plt.subplot(3, 3, 5)
        ax5.imshow(super_res_images['photon'], cmap='hot')
        ax5.set_title('Photon-Weighted\nSuper-Resolution', fontweight='bold')
        ax5.axis('off')
        
        ax6 = plt.subplot(3, 3, 6)
        ax6.imshow(super_res_images['histogram'], cmap='hot')
        ax6.set_title('Histogram\nSuper-Resolution', fontweight='bold')
        ax6.axis('off')
        
        # Row 3: Analysis details
        ax7 = plt.subplot(3, 3, 7)
        ax7.imshow(marker_image, cmap='gray', alpha=0.7)
        speckle_molecules = [loc for loc in localizations if loc.get('location') == 'nuclear_speckle']
        background_molecules = [loc for loc in localizations if loc.get('location') == 'background']
        
        if speckle_molecules:
            speckle_x = [loc['x'] for loc in speckle_molecules]
            speckle_y = [loc['y'] for loc in speckle_molecules]
            ax7.scatter(speckle_x, speckle_y, c='red', s=8, alpha=0.8, label=f'Speckle ({len(speckle_molecules)})')
        
        if background_molecules:
            bg_x = [loc['x'] for loc in background_molecules]
            bg_y = [loc['y'] for loc in background_molecules]
            ax7.scatter(bg_x, bg_y, c='cyan', s=2, alpha=0.6, label=f'Background ({len(background_molecules)})')
        
        ax7.set_title('Assignment Results', fontweight='bold')
        ax7.legend()
        
        ax8 = plt.subplot(3, 3, 8)
        if localizations:
            precisions = [loc.get('x_error', 0.5) for loc in localizations]
            ax8.hist(precisions, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax8.set_title(f'Localization Precision\n(Mean: {np.mean(precisions):.2f} pixels)', fontweight='bold')
            ax8.set_xlabel('Precision (pixels)')
            ax8.set_ylabel('Count')
        
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        stats_text = f"""
ANALYSIS SUMMARY

Molecules Detected: {summary['total_molecules']:,}
Nuclear Speckles: {len([k for k in speckle_stats.keys() if k != 'summary' and isinstance(k, int)])}

Assignment:
• Speckle: {summary['speckle_molecules']:,} ({summary['speckle_molecules']/summary['total_molecules']*100:.1f}%)
• Background: {summary['background_molecules']:,}

Enrichment: {summary['enrichment_factor']:.2f}x

Super-Resolution:
• Oversampling: {self.oversampling}x
• Resolution: {self.renderer.super_pixel_size:.1f} nm/pixel
• Output size: {super_res_images['precision'].shape[0]}×{super_res_images['precision'].shape[1]}
"""
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
        
        plt.tight_layout()
        viz_file = os.path.join(self.output_dir, f"{base_filename}_complete_analysis.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🖼️  Saved comprehensive visualization: {viz_file}")
    
    def run_analysis(self, storm_movie_path, marker_path):
        """Run complete analysis with super-resolution rendering."""
        start_time = time.time()
        
        # Load and process data
        storm_movie, marker_image = self.load_and_resize_data(storm_movie_path, marker_path)
        if storm_movie is None or marker_image is None:
            return None
        
        base_filename = os.path.splitext(os.path.basename(storm_movie_path))[0]
        
        # Analysis pipeline
        localizations = self.localize_molecules(storm_movie)
        speckle_labels, speckle_info = self.segment_speckles(marker_image)
        assigned_locs, speckle_stats = self.assign_molecules(localizations, speckle_labels)
        
        # Super-resolution rendering
        super_res_images = self.create_super_resolution_images(assigned_locs, marker_image.shape, base_filename)
        
        # Comprehensive visualization
        self.create_comprehensive_visualization(
            assigned_locs, speckle_labels, marker_image, speckle_stats, super_res_images, base_filename
        )
        
        # Save results
        if assigned_locs:
            df = pd.DataFrame(assigned_locs)
            csv_file = os.path.join(self.output_dir, f"{base_filename}_localizations.csv")
            df.to_csv(csv_file, index=False)
            print(f"   📄 Saved localizations: {csv_file}")
        
        # Summary
        processing_time = time.time() - start_time
        summary = speckle_stats['summary']
        
        print(f"\n✅ COMPLETE ANALYSIS FINISHED!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"🔬 Molecules detected: {summary['total_molecules']:,}")
        print(f"🎯 Nuclear speckles: {len(speckle_info)}")
        print(f"📈 Enrichment factor: {summary['enrichment_factor']:.2f}x")
        print(f"🎨 Super-resolution images created with {self.oversampling}x oversampling")
        print(f"💾 Results saved to: {self.output_dir}")
        
        return speckle_stats

def main():
    """Main function."""
    print("🎨 STORM NUCLEAR SPECKLE ANALYZER - WITH SUPER-RESOLUTION")
    print("=" * 70)
    
    # ==========================================
    # CHANGE THESE TO YOUR FILE NAMES
    # ==========================================
    storm_movie_file = "r2.tif"           # ← Your STORM movie
    nuclear_speckle_marker_file = "b2_epi.tif"     # ← Your marker image
    
    # Analysis parameters
    oversampling = 5  # 5x super-resolution (adjust as needed)
    
    # Check files
    missing_files = []
    if not os.path.exists(storm_movie_file):
        missing_files.append(storm_movie_file)
    if not os.path.exists(nuclear_speckle_marker_file):
        missing_files.append(nuclear_speckle_marker_file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print(f"📁 Available files:")
        for f in os.listdir('.'):
            if f.endswith(('.tif', '.tiff', '.nd2')):
                print(f"   • {f}")
        return
    
    # Run analysis
    analyzer = DimensionFixedSTORMAnalyzer(
        pixel_size=108.0,  # Adjust for your camera
        oversampling=oversampling
    )
    
    try:
        results = analyzer.run_analysis(storm_movie_file, nuclear_speckle_marker_file)
        if results:
            print(f"\n🎉 SUCCESS! Check the super-resolution images and analysis!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()