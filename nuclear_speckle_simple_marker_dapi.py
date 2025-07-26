import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def find_image_files():
    """Find all image files in current directory"""
    current_dir = Path.cwd()
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.nd2', '.czi']
    
    image_files = []
    found_files = set()  # To avoid duplicates
    
    for ext in image_extensions:
        files = list(current_dir.glob(f'*{ext}'))
        files.extend(list(current_dir.glob(f'*{ext.upper()}')))  # Check uppercase too
        
        for f in files:
            if f.name not in found_files:
                image_files.append(f)
                found_files.add(f.name)
    
    print(f"🔍 Found {len(image_files)} image files:")
    for f in image_files:
        print(f"   {f.name}")
    
    return image_files

def identify_channels(image_files):
    """Identify marker and DAPI channels from filenames"""
    
    marker_files = []
    dapi_files = []
    other_files = []
    
    # Keywords to identify different channels
    marker_keywords = ['marker', 'speckle', 'sc35', 'u2af', 'srsf', 'hnrnp', 'malat', 'son', 'cy3', 'alexa', 'red', 'ch1', 'c1']
    dapi_keywords = ['dapi', 'hoechst', 'nucleus', 'nuclear', 'blue', 'ch2', 'c2']
    
    for img_file in image_files:
        filename_lower = img_file.name.lower()
        
        if any(keyword in filename_lower for keyword in marker_keywords):
            marker_files.append(img_file)
        elif any(keyword in filename_lower for keyword in dapi_keywords):
            dapi_files.append(img_file)
        else:
            other_files.append(img_file)
    
    print(f"\n📋 Channel identification:")
    print(f"   Marker files: {[f.name for f in marker_files]}")
    print(f"   DAPI files: {[f.name for f in dapi_files]}")
    if other_files:
        print(f"   Other files: {[f.name for f in other_files]}")
    
    return marker_files, dapi_files, other_files

def load_nd2_image(filepath):
    """Load .nd2 file specifically"""
    try:
        from nd2reader import ND2Reader
        
        print(f"📖 Loading .nd2 file: {filepath.name}...")
        
        with ND2Reader(str(filepath)) as images:
            print(f"   Metadata: {images.metadata}")
            print(f"   Sizes: {images.sizes}")
            print(f"   Channels: {list(images.metadata.get('channels', []))}")
            
            # Get dimensions
            if 'c' in images.sizes:  # Multi-channel
                num_channels = images.sizes['c']
                print(f"   Found {num_channels} channels")
                
                # Try to get first two channels
                images.default_coords['c'] = 0
                channel1 = np.array(images[0])
                
                channel2 = None
                if num_channels > 1:
                    images.default_coords['c'] = 1  
                    channel2 = np.array(images[0])
                
                print(f"   Channel 1 shape: {channel1.shape}, dtype: {channel1.dtype}")
                if channel2 is not None:
                    print(f"   Channel 2 shape: {channel2.shape}, dtype: {channel2.dtype}")
                
                return channel1, channel2
            else:
                # Single channel
                image = np.array(images[0])
                print(f"   Single channel shape: {image.shape}, dtype: {image.dtype}")
                return image, None
                
    except ImportError:
        print(f"❌ nd2reader not available. Installing now...")
        return None, None
    except Exception as e:
        print(f"❌ Error loading .nd2 file: {e}")
        return None, None

def load_image(filepath):
    """Load image with error handling"""
    try:
        # Handle .nd2 files specifically
        if filepath.suffix.lower() == '.nd2':
            return load_nd2_image(filepath)
        
        # For other formats
        from skimage import io
        
        print(f"📖 Loading {filepath.name}...")
        image = io.imread(str(filepath))
        
        # Handle multi-channel images
        if len(image.shape) > 2:
            print(f"   Multi-channel image: {image.shape}")
            if image.shape[2] > 1:
                print(f"   Taking first channel")
                image = image[:, :, 0]
            else:
                image = image.squeeze()
        
        print(f"   Final shape: {image.shape}, dtype: {image.dtype}")
        return image, None
        
    except Exception as e:
        print(f"❌ Error loading {filepath.name}: {e}")
        return None, None

def segment_nuclei(dapi_image):
    """Segment nuclear boundaries"""
    from skimage import filters, morphology, measure
    from scipy import ndimage
    
    print("\n🔬 Segmenting nuclei from DAPI...")
    
    # Convert to float
    dapi_float = dapi_image.astype(np.float32)
    if np.max(dapi_float) > 1:
        dapi_float = dapi_float / np.max(dapi_float)
    
    # Gaussian blur and threshold
    blurred = filters.gaussian(dapi_float, sigma=1.5)
    threshold = filters.threshold_otsu(blurred)
    binary = blurred > threshold
    
    # Clean up
    binary = morphology.remove_small_objects(binary, min_size=5000)
    binary = morphology.closing(binary, morphology.disk(3))
    binary = ndimage.binary_fill_holes(binary)
    
    # Label nuclei
    nuclear_masks = measure.label(binary)
    num_nuclei = len(np.unique(nuclear_masks)) - 1
    
    print(f"✅ Found {num_nuclei} nuclei")
    return nuclear_masks

def segment_speckles(marker_image, nuclear_masks=None):
    """Segment nuclear speckles"""
    from skimage import filters, morphology, measure, segmentation
    
    print("\n🎯 Segmenting nuclear speckles...")
    
    # Convert to float
    marker_float = marker_image.astype(np.float32)
    if np.max(marker_float) > 1:
        marker_float = marker_float / np.max(marker_float)
    
    # Background subtraction
    background = morphology.white_tophat(marker_float, morphology.disk(30))
    processed = marker_float - background
    processed[processed < 0] = 0
    
    # Find seeds and apply watershed
    h_value = np.percentile(processed, 90) * 0.1
    if h_value > 0:
        seeds = morphology.h_maxima(processed, h=h_value)
        markers = measure.label(seeds)
        
        threshold = filters.threshold_otsu(processed)
        mask = processed > threshold
        
        if np.max(markers) > 0:
            speckle_labels = segmentation.watershed(-processed, markers, mask=mask)
        else:
            speckle_labels = measure.label(mask)
    else:
        # Fallback
        threshold = filters.threshold_otsu(processed)
        speckle_labels = measure.label(processed > threshold)
    
    # Size filtering
    speckle_labels = morphology.remove_small_objects(speckle_labels, min_size=100)
    
    # Filter by nuclear membership
    if nuclear_masks is not None:
        filtered_labels = np.zeros_like(speckle_labels)
        new_label = 1
        
        for region in measure.regionprops(speckle_labels):
            centroid = region.centroid
            y, x = int(centroid[0]), int(centroid[1])
            
            if 0 <= y < nuclear_masks.shape[0] and 0 <= x < nuclear_masks.shape[1]:
                if nuclear_masks[y, x] > 0:  # Inside nucleus
                    filtered_labels[speckle_labels == region.label] = new_label
                    new_label += 1
        
        speckle_labels = filtered_labels
    
    num_speckles = len(np.unique(speckle_labels)) - 1
    print(f"✅ Found {num_speckles} nuclear speckles")
    
    return speckle_labels

def analyze_and_visualize(marker_image, dapi_image, nuclear_masks, speckle_masks):
    """Create analysis and save results"""
    from skimage import measure, segmentation
    
    print("\n📊 Analyzing and creating visualizations...")
    
    # Analysis
    composition = {}
    if np.max(speckle_masks) > 0:
        for region in measure.regionprops(speckle_masks, intensity_image=marker_image):
            composition[region.label] = {
                'area': region.area,
                'mean_intensity': region.mean_intensity,
                'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0,
                'centroid': region.centroid
            }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Nuclear Speckle Analysis Results', fontsize=16, fontweight='bold')
    
    # Original images
    if marker_image is not None:
        axes[0, 0].imshow(marker_image, cmap='Reds')
        axes[0, 0].set_title('Marker Channel')
        axes[0, 0].axis('off')
    
    if dapi_image is not None:
        axes[0, 1].imshow(dapi_image, cmap='Blues')
        axes[0, 1].set_title('DAPI Channel')
        axes[0, 1].axis('off')
    
    # Nuclear boundaries
    if nuclear_masks is not None and marker_image is not None:
        axes[0, 2].imshow(marker_image, cmap='gray', alpha=0.7)
        boundaries = segmentation.find_boundaries(nuclear_masks, mode='outer')
        axes[0, 2].contour(boundaries, colors='red', linewidths=2)
        axes[0, 2].set_title('Nuclear Boundaries')
        axes[0, 2].axis('off')
    
    # Speckle segmentation
    if speckle_masks is not None:
        axes[1, 0].imshow(marker_image, cmap='gray', alpha=0.6)
        axes[1, 0].imshow(speckle_masks, cmap='Set3', alpha=0.8)
        axes[1, 0].set_title('Speckle Segmentation')
        axes[1, 0].axis('off')
    
    # Analysis plots
    if composition:
        areas = [comp['area'] for comp in composition.values()]
        intensities = [comp['mean_intensity'] for comp in composition.values()]
        
        # Area histogram
        axes[1, 1].hist(areas, bins=max(3, len(areas)//3), alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Speckle Area (pixels)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Size Distribution')
        
        # Scatter plot
        axes[1, 2].scatter(areas, intensities, alpha=0.7, c='orange', s=50)
        axes[1, 2].set_xlabel('Area (pixels)')
        axes[1, 2].set_ylabel('Mean Intensity')
        axes[1, 2].set_title('Size vs Intensity')
    
    plt.tight_layout()
    
    # Save results
    plt.savefig('nuclear_speckle_results.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: nuclear_speckle_results.png")
    
    # Save report
    with open('analysis_report.txt', 'w') as f:
        f.write("NUCLEAR SPECKLE ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Nuclei detected: {len(np.unique(nuclear_masks)) - 1 if nuclear_masks is not None else 0}\n")
        f.write(f"Speckles detected: {len(np.unique(speckle_masks)) - 1}\n\n")
        
        if composition:
            areas = [comp['area'] for comp in composition.values()]
            intensities = [comp['mean_intensity'] for comp in composition.values()]
            f.write(f"Mean speckle area: {np.mean(areas):.1f} ± {np.std(areas):.1f} pixels\n")
            f.write(f"Mean intensity: {np.mean(intensities):.1f} ± {np.std(intensities):.1f}\n")
    
    print(f"✅ Saved: analysis_report.txt")
    
    # Show summary
    print("\n" + "="*50)
    print("📋 ANALYSIS SUMMARY")
    print("="*50)
    print(f"Nuclei: {len(np.unique(nuclear_masks)) - 1 if nuclear_masks is not None else 0}")
    print(f"Speckles: {len(np.unique(speckle_masks)) - 1}")
    if composition:
        areas = [comp['area'] for comp in composition.values()]
        print(f"Mean speckle area: {np.mean(areas):.1f} pixels")
        print(f"Speckles per nucleus: {len(composition) / (len(np.unique(nuclear_masks)) - 1):.1f}" if nuclear_masks is not None and len(np.unique(nuclear_masks)) > 1 else "")

def main():
    """Main analysis function"""
    
    print("🔬 NUCLEAR SPECKLE ANALYZER")
    print("=" * 40)
    
    # Find image files
    image_files = find_image_files()
    if not image_files:
        print("❌ No image files found in current directory!")
        print("   Make sure your .tif, .png, or other image files are in the same folder as this script")
        return
    
    # Identify channels
    marker_files, dapi_files, other_files = identify_channels(image_files)
    
    # Load images
    marker_image = None
    dapi_image = None
    
    # Try to load from .nd2 file first
    if other_files and other_files[0].suffix.lower() == '.nd2':
        print("\n🔄 Detected .nd2 file - loading both channels...")
        channel1, channel2 = load_image(other_files[0])
        
        if channel1 is not None:
            # Assume first channel is marker, second is DAPI
            marker_image = channel1
            print("   Using channel 1 as marker")
            
            if channel2 is not None:
                dapi_image = channel2 
                print("   Using channel 2 as DAPI")
            else:
                print("   Only one channel found")
        else:
            print("❌ Could not load .nd2 file. Try installing: pip install nd2reader")
            return
    else:
        # Try to load marker channel
        if marker_files:
            result = load_image(marker_files[0])
            marker_image = result[0] if isinstance(result, tuple) else result
        elif other_files:
            print("⚠️  No marker channel identified, trying first file")
            result = load_image(other_files[0])
            marker_image = result[0] if isinstance(result, tuple) else result
        
        # Try to load DAPI channel  
        if dapi_files:
            result = load_image(dapi_files[0])
            dapi_image = result[0] if isinstance(result, tuple) else result
        elif len(other_files) > 1:
            print("⚠️  No DAPI channel identified, trying second file")
            result = load_image(other_files[1])
            dapi_image = result[0] if isinstance(result, tuple) else result
    
    if marker_image is None:
        print("❌ Could not load any marker channel image!")
        return
    
    # Run analysis
    nuclear_masks = None
    if dapi_image is not None:
        nuclear_masks = segment_nuclei(dapi_image)
    else:
        print("⚠️  No DAPI image - analyzing speckles without nuclear filtering")
    
    speckle_masks = segment_speckles(marker_image, nuclear_masks)
    
    if np.max(speckle_masks) == 0:
        print("❌ No speckles detected! Check your marker channel image.")
        return
    
    # Create visualizations and report
    analyze_and_visualize(marker_image, dapi_image, nuclear_masks, speckle_masks)
    
    print("\n🎉 Analysis complete! Check the generated files:")
    print("   📊 nuclear_speckle_results.png")
    print("   📝 analysis_report.txt")

if __name__ == '__main__':
    main()