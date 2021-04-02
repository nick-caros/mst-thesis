"""

This program takes two shapefiles representing "edge", a block or partial
block shape served by a bus route. Each shapefile should represent a different
period of service. It also takes two dictionaries containing some performance
metric for the bus service represented by each of the shapefiles. 

The output is a combined shapefile containins all of the unique edges across
both input shapefiles. A new property is appended: the difference in the
performance metric between the baseline period and the comparison period.

"""

import geopandas as gpd
import json


base_shapes_path = '../data/your_edge_shapes_file.geojson'
base_metrics_path = '../data/your_metric_file.json'

comp_shapes_path = '../data/your_edge_shapes_file.geojson'
comp_metrics_path = '../data/your_metric_file.json'

outpath = '../output/your_results.geojson'

with open(base_metrics_path) as f:
  base_metrics = json.load(f)

with open(comp_metrics_path) as f:
  comp_metrics = json.load(f)
  
base_shapes = gpd.read_file(base_shapes_path)
comp_shapes = gpd.read_file(comp_shapes_path)

# Cycle through base shapes first, find matches and append metrics
geom_index = base_shapes.columns.get_loc("geometry")
seg_index = base_shapes.columns.get_loc("indices")
edge_index = base_shapes.columns.get_loc("edge")
poly_index = base_shapes.columns.get_loc("polyline")

segment_list = []
geometry_list = []
metric_list = []
polyline_list = []
edge_list = []
indicator_list = [] # 0 = dropped service; 1 = new service; 2 = maintained service
matched_lines = {}

for piece in base_shapes.values.tolist():
    match = 0
    indicator = 0
    base_total = 0
    comp_total = 0
    comp_segments = None
    
    edge = piece[edge_index]
    base_segments = piece[seg_index]
    base_polyline = piece[poly_index]
    base_line = piece[geom_index]

    for segment in base_segments:
        segment_key = base_segments[segment]
        try:
            base_total += base_metrics[segment_key]
        except KeyError:
            #print("No match in base metrics")
            continue
            
    potential_matches = comp_shapes[comp_shapes['edge'] == edge].values.tolist()

    for potential_match in potential_matches:
        comp_line = potential_match[geom_index]
        if base_line.intersects(comp_line):
            match = 1
            indicator = 2
            comp_segments = potential_match[seg_index]
            for segment in comp_segments:
                segment_key = comp_segments[segment]
                try:
                    comp_total += comp_metrics[segment_key]
                except KeyError:
                    #print("No match in comp metrics")
                    continue
            break
    
    geometry_list.append(base_line)
    polyline_list.append(base_polyline)
    edge_list.append(edge)
    indicator_list.append(indicator)
    output_segments = {}
    output_segments['base'] = base_segments
    output_segments['comp'] = comp_segments
    segment_list.append(output_segments)
    metric_list.append(comp_total - base_total)
    matched_lines[base_polyline] = match
    
# Add metrics to any leftover comparison shapes and add to combined dict
for piece in comp_shapes.values.tolist():
    comp_polyline = piece[poly_index]
    edge = piece[edge_index]
    comp_total = 0
    
    if comp_polyline in matched_lines:
        continue
    
    comp_segments = piece[seg_index]
    for segment in comp_segments:
        segment_key = comp_segments[segment]
        try:
            comp_total += comp_metrics[segment_key]
        except KeyError:
            #print("No match in comp metrics")
            continue
    
    geometry_list.append(piece[geom_index])
    polyline_list.append(comp_polyline)
    edge_list.append(edge)
    indicator_list.append(1)
    output_segments = {}
    output_segments['base'] = None
    output_segments['comp'] = comp_segments
    segment_list.append(output_segments)
    metric_list.append(comp_total)
    matched_lines[base_polyline] = 0

gdf = gpd.GeoDataFrame(geometry = geometry_list)
gdf['polyline'] = polyline_list
gdf['segments'] = segment_list
gdf['metric'] = metric_list
gdf['edge'] = edge_list
gdf['service_indicator'] = indicator_list

# Run a sense check on the total gain/loss in service
total_base = sum([base_metrics[x] for x in base_metrics])
total_comp = sum([comp_metrics[x] for x in comp_metrics])
total_diff = sum(metric_list)

gdf = gdf.sort_values(by = ['edge'])
gdf.to_file(outpath, driver='GeoJSON')         



