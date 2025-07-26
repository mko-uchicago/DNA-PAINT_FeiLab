#!/usr/bin/env python3
"""
Simple runner script for Picasso-faithful STORM analysis.
Adjust the file paths below to match your data.
"""

from picasso_faithful_storm_analyzer import PicassoFaithfulSTORMAnalyzer
import os

def main():
    print("🧬 Running Picasso-Faithful STORM Analysis")
    print("=" * 50)
    
    # === ADJUST THESE PATHS FOR YOUR DATA ===
    storm_file = "r2.tif"      # Your STORM movie (256x256 or 400x400)
    marker_file = "b2_epi.tif"         # Your marker image (1250x1250 or 2000x2000)
    
    # Check if files exist
    if not os.path.exists(storm_file):
        print(f"❌ STORM file not found: {storm_file}")
        print("Please update the 'storm_file' path in this script")
        return
    
    if not os.path.exists(marker_file):
        print(f"❌ Marker file not found: {marker_file}")
        print("Please update the 'marker_file' path in this script")
        return
    
    # Initialize analyzer with optimized parameters
    analyzer = PicassoFaithfulSTORMAnalyzer(
        pixel_size=108.0,      # Camera pixel size in nm (adjust if different)
        oversampling=5,        # 5x super-resolution (can increase to 8-10x for more detail)
        output_dir="./picasso_faithful_results"
    )
    
    # Run complete analysis
    try:
        analyzer.run_complete_analysis(storm_file, marker_file)
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved in: ./picasso_faithful_results/")
        print("📊 Check the comprehensive visualization for results summary")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💡 Try adjusting parameters or check file formats")

if __name__ == "__main__":
    main()