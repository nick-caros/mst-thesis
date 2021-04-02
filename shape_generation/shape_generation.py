"""
This program is used to take the stop and (optionally) shape coordinates
from a standard GTFS feed and use them as inputs to the Valhalla Map Matching 
API. These are exported as a JavaScript file containina a single JSON object.

The output object contains one entry for each stop pair in the network. Each
entry contains eight fields: 
    
1) route_id from GTFS
2) The stop pair as a list of GTFS stop_id: [first stop, last stop]
3) direction_id from GTFS
4) The pattern, which is a concatenation of 'route_id - direction - rank', 
    where rank is determined by the number of trips that follow that pattern
5) Distance travelled between stops in km 
6) Segment Index, which is a concatenation of 'route_id - first_stop - last_stop'
7) Timepoint Index, which is a concatenation of 'route_id - timepoint_sequence'
8) A Google Maps polyline encoded representation of the line geometries.

This program makes use of the open-source Valhalla routing engine,
which can be found here: https://github.com/valhalla/valhalla

Inspiration for the map matching portion comes from Matthew Conway's project:
https://indicatrix.org/identifying-high-quality-transit-corridors-a8e4eec37ed8


The remaining code also combines segments into corridors where appropriate. 
Three dictionaries are constructed: 

"""

import math
import pandas as pd
import partridge as ptg
import numpy as np
import requests
import json
import time

from shapely.ops import nearest_points
from shapely.geometry import LineString, Point


def generate_shapes(gtfs_inpath):
    
    turn_penalty_factor = 100000 # Penalizes turns in Valhalla routes. Range 0 - 100,000.
    stop_radius = 35 # Radius used to search when matching stop coordinates (meters)
    intermediate_radius = 100 # Radius used to search when matching intermediate coordinates (meters)
    
    stop_distance_threshold  = 1000 # Stop-to-stop distance threshold for including intermediate coordinates (meters)
    maneuver_penalty = 43200 # Penalty when a route includes a change from one road to another (seconds). Range 0 - 43,200. 
    
    # Initialize Valhalla input dictionary with some empty values
    point_parameters = {'lon': None,
                        'lat': None,
                        'type': None,
                        'radius': None,
                        'rank_candidates': 'true',
                        'preferred_side': 'same',
                        'node_snap_tolerance': 0,
                        'street_side_tolerance':0
                        }
    
    request_parameters = {'shape': None,
                          'costing': 'bus',
                          'shape_match': 'map_snap',
                          'filters':{
                              'attributes': ['edge.id', 'edge.length', 'shape'],
                              'action':'include'
                              },
                          'costing_options':{
                              'bus':{
                                  'maneuver_penalty': maneuver_penalty
                                  }
                              },
                          'trace_options.turn_penalty_factor': turn_penalty_factor
                          }
    
    """ -------------Objects------------- """
    
    class Pattern: # Attributes for each unique pattern of stops that create one or more route variant
        def __init__(self, route, direction, stops, trips, stop_coords, shape, timepoints):
            self.route = route
            self.direction = direction
            self.stops = stops
            self.shape = shape
            self.trips = trips
            self.timepoints = timepoints
            self.stop_coords = stop_coords
            self.shape_coords = 0
            self.v_input = 0
            self.coord_types = 0
            self.radii = 0
        
    class Segment: # Attributes for each segment which make up a pattern
        def __init__(self, geometry, distance):
            self.geometry = geometry
            self.distance = distance
            
    class Corridor: # Attributes for each corridor
        def __init__(self, edges, segments):
            self.edges = edges
            self.segments = segments
            self.passenger_shared = []
            self.stop_shared = []
            
        def get_edges(self):
            return self.edges
        
        def get_segments(self):
            return self.segments
        
        def get_pass_shared(self):
            return self.passenger_shared
        
        def get_stop_shared(self):
            return self.stop_shared
    
    """ -------------Functions------------- """
    
    # Function to get distance (in m) from a pair of lat, long coord tuples
    def get_distance(start, end):
        R = 6372800 # earth radius in m
        lat1, lon1 = start
        lat2, lon2 = end
        
        phi1, phi2 = math.radians(lat1), math.radians(lat2) 
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1) 
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2    
        return round(2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a)),0)
    
    """ 
    Takes a set of route coordinates and bus stop coordinates, then finds the 
    route coordinate pair that is closest to each bus stop. Returns an array of 
    strings of the same length as the trip coordinate input, with 'break_through' 
    for coordinates at bus stops and 'through' for other coordinates.
    """
    def locate_stops_in_shapes(shape_coords, stop_coords, stop_radius, intermediate_radius):
        coordinate_types = [1] * len(stop_coords)
        radii = [stop_radius] * len(stop_coords)
        stop_indices = [0] * len(stop_coords)
        shape_coord_list = shape_coords.values.tolist()
        
        last_stop = 0
        #count = 1
        coordinate_list = []
        shape_line = LineString([Point(x, y) for x, y in zip(shape_coords.shape_pt_lon, shape_coords.shape_pt_lat)])
    
        # Get index of point closest to each bus stop            
        for stop_number, stop in enumerate(stop_coords):
            
            stop_point = Point(stop[1], stop[0])
            new_stop = nearest_points(shape_line, stop_point)[0] # index 0 is nearest point on the line
            coordinate_list.append((new_stop.y, new_stop.x))
            
            benchmark = 10**9
            index = 0
            best_index = 0
            for point in shape_coord_list[last_stop:]: # Ensure stops occur sequentially
                test_dist = get_distance(point, stop)
                if test_dist+2 < benchmark: # Add 2m to ensure that loop routes don't the later stop 
                    benchmark = test_dist
                    best_index = index + last_stop
                index += 1
            stop_indices[stop_number] = best_index
            last_stop = best_index + 1
            #print("Stop #", count, "; Best Index:", best_index)
            #count += 1
        
        added_stop_count = 0
        
        # Add intermediate coordinates if stops are far apart
        for stop_number in range(len(stop_coords)-1):
            current_stop = stop_coords[stop_number]
            next_stop = stop_coords[stop_number + 1]
            current_pos = stop_indices[stop_number]
            next_pos = stop_indices[stop_number + 1]
            
            distance = get_distance(current_stop, next_stop)
        
            if distance > stop_distance_threshold:    
                
                coords_to_add = math.floor(distance/stop_distance_threshold )
                num_available_coords = next_pos - current_pos            
                interval = int(num_available_coords / (coords_to_add + 1))
    
                # If there aren't enough available coords to fill the shape, just add all coords
                if coords_to_add > num_available_coords:
                    
                    for new_coord in range(num_available_coords):    
                        
                        coordinate_list.insert(stop_number + 1 + added_stop_count, shape_coord_list[current_pos + new_coord])
                        coordinate_types.insert(stop_number + 1 + added_stop_count, 0)
                        radii.insert(stop_number + 1 + added_stop_count, intermediate_radius)
                        added_stop_count += 1
                    
                else:
                    for new_coord in range(coords_to_add):
                        
                        coordinate_list.insert(stop_number + 1 + added_stop_count, shape_coord_list[current_pos + (interval * new_coord)])
                        coordinate_types.insert(stop_number + 1 + added_stop_count, 0)
                        radii.insert(stop_number + 1 + added_stop_count, intermediate_radius)
                        added_stop_count += 1
                
        return coordinate_types, coordinate_list, radii
    
    # Create index for each pattern
    def get_pattern_index(patterns):
        patterns = patterns.sort_values(by=['route_id','direction_id','count'], ascending=[True, True, False])
        prev_dir = 0
        prev_route = 0
        index = []
        for pattern in patterns.values.tolist():
            route = pattern[2]
            direction = pattern[4]
            if route != prev_route or direction != prev_dir:
                pattern_count = 1
            else: pattern_count += 1
            index.append(str(route)+'-'+str(direction)+'-'+str(pattern_count))
            prev_dir = direction
            prev_route = route
        patterns['pattern_index'] = index
        return patterns
        
    def get_skipped_segments(coords, request_data):
        # If request times out, try twice more and then raise an error
        to_count = 1
        while to_count < 4:
            try:
                # Use Valhalla map matching engine to snap shapes to the road network
                request_data['shape'] = coords
                req = requests.post('http://localhost:8002/trace_attributes',
                                    data = json.dumps(request_data),
                                    timeout = 100)
                to_count = 10
            except:
                print("Timeout #", to_count)
                to_count += 1
            if to_count == 4:
                raise Exception('Request timed out 3x')
    
        # Extract Valhalla response
        return req.json()
                                 
    def store_geometry_and_distance(result, leg):
        geometry = result['trip']['legs'][leg]['shape']
        distance = result['trip']['legs'][leg]['summary']['length']      
        return Segment(geometry, distance)
    
    def match_segs_to_edges(pair_list, pair_dict, request_parameters):
        cm_count = 0 
        start_time = time.time()
       
        for pair in pair_list:
            geometry = pair[1]
            pair_index = pair[0]
            
            if pair_index in pair_dict: # If edges already identified, skip
                cm_count += 1 
                continue
            else:
                
                # If request times out, try twice more and then raise an error
                to_count = 1
                while to_count < 4:
                    try:
                        # Use Valhalla map matching engine to snap shapes to the road network
                        request_data = request_parameters.copy()
                        request_data['shape'] = geometry
                        req = requests.post('http://localhost:8002/trace_attributes',
                                            data = json.dumps(request_data),
                                            timeout = 100)
                        to_count = 10
                    except:
                        print("Timeout #", to_count)
                        to_count += 1
                    if to_count == 4:
                        raise Exception('Request ', cm_count,' timed out 3x')
            
                # Extract Valhalla response and store as pair object attribute
                result = req.json()
                edges = []
                for edge in result['edges']:
                    edges.append(edge['id'])
                pair_dict[pair_index] = edges
            
            cm_count += 1
            if cm_count % 100 == 0:
                elapsed_time = time.time() - start_time
                print(cm_count, "of", len(pair_list), "edge ids identified.",
                      "Elapsed time:", round(elapsed_time,0))
                start_time = time.time()
        
        return pair_dict
    
    """ -------------Main Program------------- """
    
    # Import GTFS feed and filter down to normal bus routes only
    route_type = ['3']
    route_desc = ['Key Bus', 'Commuter Bus', 'Local Bus']
    view = {'routes.txt': {'route_type': route_type, 'route_desc': route_desc}}
    feed = ptg.load_feed(gtfs_inpath, view)
    
    # Check if shapes.txt exists in GTFS feed
    try:
        feed_shapes = feed.shapes[['shape_id', 'shape_pt_lat', 'shape_pt_lon']]
        has_shapes = True
    except:
        has_shapes = False
    
    # Check if timepoints included in GTFS feed
    
    try:
        feed_stop_events = feed.stop_times[['trip_id', 'stop_id', 'stop_sequence', 'checkpoint_id']]
        has_timepoints = True
    except:
        feed_stop_events = feed.stop_times[['trip_id', 'stop_id', 'stop_sequence']]
        has_timepoints = False
    
    # Get relevant tables from GTFS feed: trips, routes and stop sequences
    feed_trips = feed.trips[['route_id','trip_id','direction_id']]
    all_stops = pd.merge(feed_trips, feed_stop_events, on='trip_id', how='inner')
    all_stops = all_stops.sort_values(by=['trip_id', 'stop_sequence'])
    stops_dict = all_stops.groupby('trip_id')['stop_id'].agg(list).to_dict()
    
    # Get timepoints and change timepoints from binary to increasing count
    if has_timepoints == True:
        tp_dict = all_stops.groupby('trip_id')['checkpoint_id'].agg(list).to_dict()
        for trip in tp_dict:
            tp_list = tp_dict[trip]
            new_list = []
            tp_count = 0
            for stop in tp_list:
                if type(stop) == str:
                    tp_count += 1
                new_list.append(tp_count)
            tp_dict[trip] = new_list
            
    else:
        # Enter zeros
        tp_dict = {}
        for trip in stops_dict:
            tp_dict[trip] = [0] * len(stops_dict[trip])
            
    # Get coordinates for each stop from gtfs
    feed_stops = feed.stops[['stop_id','stop_lat','stop_lon']].copy()
    stop_coordinates = list(zip(feed_stops.stop_lat, feed_stops.stop_lon))
    feed_stops['coords'] = stop_coordinates.copy()
    feed_stops = feed_stops[['stop_id','coords']]
    stop_df = pd.merge(all_stops, feed_stops, on='stop_id', how='inner')
    stop_df = stop_df.sort_values(by=['trip_id', 'stop_sequence'])
    coords_dict = stop_df.groupby('trip_id')['coords'].agg(list).to_dict()
    
    # Find the unique sequences of stops (patterns)
    hash_list = list(stops_dict.values())
    hashes = []
    for sequence in hash_list: # hashing function for the coordinates so that they can be compared
        new_hash = 0
        count = 1
        for stop in sequence:
            try: num = int(stop)
            except: num = sum([ord(x) for x in stop])
            new_hash += (2*count)**2 + num**3 # Arbitrary hashing function
            count += 1
        hashes.append(new_hash)
    all_trips = feed_trips.sort_values(by='trip_id')
    all_trips['hash'] = hashes
    
    # Count how many times each route-hash combination appears
    pattern_counts = all_trips.groupby(['route_id','hash','direction_id']).size().reset_index(name='count')
    
    # Get the trip_ids associated with each route-hash combination as a list of lists
    trip_dict = all_trips.groupby(['route_id','hash'])['trip_id'].agg(list).to_dict()
    
    # Create a dataframe for the patterns with route_ids, direction, count and representative trip id
    all_trips = all_trips.drop_duplicates(subset=['route_id','hash', 'direction_id'])
    pattern_counts = pd.merge(pattern_counts[['count','hash','route_id']], all_trips, on=['hash','route_id'], how='inner')
    pattern_counts = get_pattern_index(pattern_counts)
    
    # Create dict of Pattern objects
    pattern_list = pattern_counts['pattern_index'].values.tolist()
    pattern_dict = {}
    shape_dict = {}
    
    if has_shapes == True:
        trip_shapes = feed.trips[['trip_id', 'shape_id']]
        trip_shapes = trip_shapes[trip_shapes['trip_id'].isin(pattern_counts['trip_id'])]
        shape_dict = dict(zip(trip_shapes['trip_id'], trip_shapes['shape_id']))
    
    for pattern in pattern_list:
        pattern_data = pattern_counts.loc[pattern_counts['pattern_index']==pattern].values.tolist()[0]
        index = pattern
        route = pattern_data[2]
        direction = pattern_data[4]
        trip_id = pattern_data[3]
        pattern_hash = pattern_data[1]
        stops = stops_dict[trip_id]
        coords = coords_dict[trip_id]
        trips = trip_dict[(route,pattern_hash)]
        timepoints = tp_dict[trip_id]
        
        if len(shape_dict) > 0:
            shape = shape_dict[str(trip_id)]
        else:
            shape = 0
            
        pattern_dict[index] = Pattern(route, direction, stops, trips, coords, shape, timepoints)
    
    # If there are no shapes in GTFS, default to the stop coordinates
    if has_shapes == False:
        for pattern in pattern_list:
            stop_coords = pattern_dict[pattern].stop_coords
            coord_json = []
            for stop in stop_coords:  
                input_data = point_parameters.copy()
                input_data['lon'] = stop[1]
                input_data['lat'] = stop[0]
                input_data['type'] = 'break_through'
                input_data['radius'] = stop_radius
                coord_json.append(input_data)
                
            pattern_dict[pattern].v_input = coord_json
            pattern_dict[pattern].coord_types = [1] * len(stop_coords)
    
    # Otherwise, include some coordinate points between each pair of stops if stops are far apart
    else:
        feed_shapes = feed.shapes[['shape_id', 'shape_pt_lat', 'shape_pt_lon']]
        count = 0
        for pattern in pattern_list:
            shape = pattern_dict[pattern].shape
            stop_coords = pattern_dict[pattern].stop_coords
            shape_coords = feed_shapes.loc[feed_shapes['shape_id']==shape][['shape_pt_lat', 'shape_pt_lon']]
            
            coordinate_type, coordinate_list, radii = locate_stops_in_shapes(shape_coords, stop_coords, stop_radius, intermediate_radius)
            pattern_dict[pattern].coord_types = coordinate_type
            pattern_dict[pattern].radii = radii
              
            # Unrelated, but we'll need this dictionary later
            pattern_dict[pattern].shape_coords = coordinate_list
            count +=1 
            if count % 100 == 0: print('Coordinates prepared for', count, 'of', len(pattern_list), 'patterns')
    
        # Check that the number of 'break's is equal to number of stops in the pattern
        for pattern in pattern_list:
        
            coord_types = pattern_dict[pattern].coord_types
            radii = pattern_dict[pattern].radii
            num_stops = len(pattern_dict[pattern].stops)
            num_breaks = coord_types.count(1)
            if num_breaks - num_stops != 0:
                print("Error: Breaks - Stops =", num_breaks - num_stops,"for Pattern", pattern)
              
            coords = pattern_dict[pattern].shape_coords
            coord_list = []
            point_count = 0
            for point in coords:
                
                if coord_types[point_count]:
                    point_type = 'break_through'
                else:
                    point_type = 'through'
                
                input_data = point_parameters.copy()
                input_data['lon'] = point[1]
                input_data['lat'] = point[0]
                input_data['type'] = point_type
                input_data['radius'] = radii[point_count]
                coord_list.append(input_data)
                point_count += 1
                
            pattern_dict[pattern].v_input = coord_list
    
    # Use map matching to convert the GTFS polylines to matched, encoded polylines
    mm_count = 0
    segment_dict = {}
    skipped_segs = {}
    start_time = time.time()
    
    for pattern in pattern_list:
        coords = pattern_dict[pattern].v_input
        coordinate_types = pattern_dict[pattern].coord_types
        pattern_segs = len(pattern_dict[pattern].stops)-1
        pattern_legs = 0
        start_point = 0
        
        # Send multiple requests to Valhalla if the response is cut off
        while pattern_legs < pattern_segs:
            # If request times out, try twice more and then raise an error
            to_count = 1
            while to_count < 6:
                try:
                    # Use Valhalla map matching engine to snap shapes to the road network
                    request_data = request_parameters.copy()
                    request_data['shape'] = coords[start_point:]
                    req = requests.post('http://localhost:8002/trace_route',
                                        data = json.dumps(request_data),
                                        timeout = 60)
                    to_count = 10
                except:
                    print("Timeout #", to_count)
                    to_count += 1
                    
                if to_count == 6:
                    # Add all segments to skipped_segments
                    coords = pattern_dict[pattern].v_input
                    input_points = [i - start_point for i, x in enumerate(coordinate_types) if(x == 1 and i >= start_point)]
                    for point_idx, point in enumerate(input_points[:-1]):
                        skipped_segs[(pattern, point_idx)] = coords[point:input_points[point_idx+1]]
                    break
           
            if to_count == 6:
                mm_count += 1
                break
              
            # Extract encoded polyline from Valhalla response
            result = req.json()
            try: 
                result_legs = len(result['trip']['legs'])
            except:
                # Assume timeout caused by high turn penalty - temporarily set lower
                for coord in coords:
                    radius = int(coord['radius'])
                    coord['radius'] = str(radius + 10)
                    if radius > 500:
                        raise Exception('No path found')
                continue
            
            # Check that the result 'matched points' match the input break points
            matched_points = [location['original_index'] for location in result['trip']['locations']]
            input_points = [i - start_point for i, x in enumerate(coordinate_types) if(x == 1 and i >= start_point)]
            
            # If no points were matched, skip to the next one
            if len(matched_points) == 0:
                last_point = input_points[0] + start_point
                start_point += input_points[1]
                skipped_segs[(pattern, pattern_legs)] = coords[last_point:start_point+1]
                pattern_legs += 1
                continue
            
            internal_missed = []
            
            # If they are not identical, there are 2 possible cases:
            # 1) Break points were skipped and 2) Response stopped short
            if matched_points != input_points:
    
                # Get missing points
                missing = np.setdiff1d(input_points,matched_points)
                
                # If the first coord is missing, skip first segment (2)
                if np.any(missing == 0):
                    last_point = input_points[0] + start_point
                    start_point += input_points[1]
                    skipped_segs[(pattern, pattern_legs)] = coords[last_point:start_point+1]
                    pattern_legs += 1
                    continue
                
                # If some inputs were skipped over (1)
                if min(missing) < max(matched_points):
                    
                    # Get skipped inputs
                    internal_missed = [i for i in missing if i < max(matched_points)]
                    previous_match = 0
                    skip_count = 0
                    
                    for missed_point in internal_missed:
                        input_index = input_points.index(missed_point)
                        previous_input = input_points[input_index - 1]
                        next_input = input_points[input_index + 1]
                        
                        # Add segments on both sides of skipped stop to skipped list
                        skipped_segs[(pattern, pattern_legs + input_index - 1)] = coords[previous_input:missed_point + 1]
                        skipped_segs[(pattern, pattern_legs + input_index)] = coords[missed_point:next_input + 1]
    
                        # Find leg before skipped point
                        last_good_match = max([matched_points.index(i) for i in input_points[:input_index] if i in matched_points])
                        next_good_match = min([matched_points.index(i) for i in input_points[input_index:] if i in matched_points])
    
                        # Store geometry, distance for segments preceding skipped point
                        for leg in range(previous_match, last_good_match):
                            segment_dict[(pattern, pattern_legs + leg + skip_count)] = store_geometry_and_distance(result, leg)
                        
                        skip_count += 1
                        previous_match = next_good_match
                    
                    rem_count = 0
                    # Store geometry, distance for segments after last skipped point
                    for leg in range(next_good_match, result_legs):
                        segment_dict[(pattern, input_index + pattern_legs + 1 + rem_count)] = store_geometry_and_distance(result, leg)
                        rem_count += 1
                    
                    # Start next matching at latest matched point
                    start_point += max([i for i in input_points if i in matched_points])
                    
                # If all missing inputs are after last matched point (2)
                elif len(missing) > 0 and min(missing) > max(matched_points):
                    
                    # Next request should start from first missing point
                    prev_stop = input_points[input_points.index(min(missing))-1] + start_point
                    start_point += min(missing)
                    
                    # Determine whether cutoff happened at a stop or in between
                    if max(matched_points) not in input_points:
                        del_last_seg = 1  
                    else:
                        del_last_seg = 0
                    
                    # Add segment between last matched point and missing point to skip list
                    skipped_segs[(pattern, pattern_legs+result_legs-del_last_seg)] = coords[prev_stop:start_point+1]
                        
                    # Store geometry, distance for segments preceding skipped point
                    for leg in range(result_legs - del_last_seg):  
                        segment_dict[(pattern, pattern_legs + leg)] = store_geometry_and_distance(result, leg)
            
                    # If we keep last segment, we need to add 1 to pattern legs
                    pattern_legs += (1 - del_last_seg)
            
            # Store distance and geometry        
            else:
                for leg in range(result_legs):
                    segment_dict[(pattern, pattern_legs + leg)] = store_geometry_and_distance(result, leg)                        
            
            pattern_legs += result_legs + len(internal_missed)
    
        mm_count += 1
        if mm_count % 100 == 0:
            elapsed_time = time.time() - start_time
            print(mm_count, "of", len(pattern_list), "patterns snapped to road network.", "Elapsed time:", round(elapsed_time,0))
            start_time = time.time()
    
    # Run a check that all segments are either in the matched segments or skipped segments
    for pattern in pattern_list:
        pattern_segs = len(pattern_dict[pattern].stops) - 1
        for segment in range(pattern_segs):
            if (pattern, segment) not in segment_dict and (pattern, segment) not in skipped_segs:
                print("Error: Pattern " + pattern + ", Seg " + str(segment) + " not assigned.")
    
    # Run a check that the number of segments in each pattern is less than (#stops - 1)
    for key in segment_dict:
        pattern = key[0]
        segment = key[1]
        if segment > len(pattern_dict[pattern].stops) - 1:
            print("Error: Too many segments assigned to pattern " + pattern)
    
    # Run the skipped shapes through trace_attributes to get shapes and distance
    pair_dict = {}
    pair_geom = {}
    for seg in skipped_segs:
        pattern = seg[0]
        sequence = seg[1]
        pair = tuple(pattern_dict[pattern].stops[sequence:sequence+2])
        
        # If this pair has already been matched as part of another pattern
        if pair in pair_geom: 
            segment_dict[seg] = Segment(pair_geom[pair][0], pair_geom[pair][1])   
            continue
        
        coords = skipped_segs[seg].copy()
        result = get_skipped_segments(coords, request_parameters)
        no_match = False
        while len(result) == 4:
            for coord in coords:
                point_radius = coord['radius']
                coord['radius'] = point_radius + 10
            result = get_skipped_segments(coords, request_parameters)
            if point_radius > 150:
                no_match = True
                break
            
        if no_match:
            continue
            
        seg_length = 0
        edge_ids = []
        for edge in result['edges']:
            seg_length += edge['length']
            edge_ids.append(edge['id'])
        segment_dict[seg] = Segment(result['shape'], seg_length)      
        
        # Store edge ids to avoid any duplicate requests
        pair_geom[pair] = [result['shape'], seg_length]
        pair_dict[pair] = edge_ids
    
    # Construct a dataframe sorted by pattern, sequence with encoded polylines
    route_dict = {}
    used_route_pairs = set()
    df_route = []
    df_pair = []
    df_dir = []
    df_pattern = []
    df_dist = []
    df_index = []
    df_tp = []
    df_encodedline = []
    for pattern in pattern_list:
        route = pattern_dict[pattern].route
        direction = pattern_dict[pattern].direction
        stops = pattern_dict[pattern].stops
        timepoints = pattern_dict[pattern].timepoints
        for stop in range(len(stops)-1):
            pair = (stops[stop], stops[stop+1])
            tp = timepoints[stop]
            if (pair + (route,)) not in used_route_pairs:
                df_route.append(route)                
                df_pair.append(pair)
                df_dir.append(direction)
                df_pattern.append(pattern)
                df_encodedline.append(segment_dict[(pattern, stop)].geometry)
                df_dist.append(segment_dict[(pattern, stop)].distance)
                df_index.append(str(route) + '-' + str(pair[0]) + '-' + str(pair[1]))
                df_tp.append(str(route) + '-' + str(tp))
                used_route_pairs.add((pair + (route,)))
            
                if pair in route_dict:
                    
                    route_dict[pair].append(route)
                else:
                    route_dict[pair] = list([route])
    
    return pd.DataFrame(list(zip(df_route, df_pair, df_dir, df_pattern, df_dist, df_index, df_encodedline)), columns = ['route_id', 'stop_pair', 'direction', 'pattern', 'distance', 'seg_index', 'geometry'])
    

# Allow this file to be run independently if needed
if __name__ == '__main__':
    df = generate_shapes(r'../../data/gtfs/your_gtfs_file.zip')

