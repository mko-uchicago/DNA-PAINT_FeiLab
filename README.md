# DNA-PAINT_FeiLab SUMMARY OF COMMITS
***all python scripts run through terminal, Python 3.13

1) Basic image analysis of DAPI/marker channels (DAPI/marker images)
- nuclear_speckle_simple_marker_dapi.py -> visuals in nuclear_speckle_results_marker_dapi.png
- 6 panels with marker, dapi channel, nuclear boundaries, speckle heatmap segmentation, size distribition, comparison of relative intensity of molecule signals 

2) Preparation for DNA-PAINT scripts (STORM IMAGES) 
- storm_nuclear_speckle_analysis.py -> r1_complete_analysis.png + r2_complete_analysis.png
- 8 panels using STORM image + marker image (1:5 pixel ratio for r1, r2): marker + segmentation, storm localization of molecules, comparison of speckle vs background counts, 3 super resolution images, localization results, localization precision based on pixels, summary of analysis

3) Alternate form of DNA-PAINT preparation scripts - picasso suite faithful vers. (STORM IMAGES)
- new_render_test.py + picasso_faithful_storm_analyzer.py -> r1_picasso_faithful_analysis.png + r2_picasso_faithful_analysis.png
- 11 panels using STORM image + marker image (1:5 pixel ratio for r1, r2): nuclear segmentation based on picasso methods, localization, picasso precision weighted rendering, picasso histogram, molecule assignment resolution of speckles versus backgrounds, backgrounds vs speckle counts, localization precision graph, photon counts, super resolution precision, super resolution histogram, localizion from frame distribution, summary of localization, segmentation, analysis, super resolution 


CREDITS: 
- Super-resolution images rendered using precision-weighted Gaussian rendering following methodology described in Picasso (Schnitzbauer, jungmannlab) with custom analysis code
- UChicago Fei Lab provided images for dapi, marker, STORM nuclear speckle images
