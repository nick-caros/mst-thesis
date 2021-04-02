## Shape Generation

The ROVE Map requires an input file containing the geometry of the bus network, divided into stop-to-stop segments.
This file is also required to contain properties for each segment, such as the start and end stops, the route ID, and so on.
These are generated from a standard GTFS feed. 
 
To generate the shape file, add the appropriate GTFS feed (.zip) to the `rove/data/` folder. 
Then specify the GTFS file location and desired output filename in the `main.py` script and run.
The output shapes file (and associated lookup table) will be saved in the `rove/bustool/static/inputs/` folder, where it can be loaded by the ROVE Map.

If this is a new period of data, then the appropriate config file in `rove/bustool/static/inputs/` should be updated to include information about the new data period.

**Note:**

The `shape_generation.py` script is the default script for generating shapes. Each route pattern is matched to the street network using Valhalla, and Valhalla splits the patterns into segments at specified points corresponding to bus stops. An alternative is to use the    `shape_generation_fullroutematch.py` script is a separate method for generating the shapes, which uses a separate GIS library to split the patterns at bus stops. Generally the `fullroutematch` method is faster but produces mixed results. Further investigation of these methods ongoing.
