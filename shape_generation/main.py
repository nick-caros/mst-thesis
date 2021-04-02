"""
This is the main program for preparing shapes files.
The user specifies the location of the desired GTFS feed, along with the 
preferred output location. This program generates uses the GTFS feed to 
generate a shapes file.

Note that this program will require the user to have Valhalla installed and 
running locally. Installation and configuration instructions are available
at https://github.com/nick-caros/mst-thesis/tree/main/documentation/valhalla_readme.md

"""
from shape_generation import generate_shapes

def main():
    
    gtfs_inpath = r'../data/gtfs/your_gtfs_here.zip'
    shapes_output_path = r'../output/your_output_filename.json'
    
    
    # Use map matching to generate bus shapes from GTFS
    df = generate_shapes(gtfs_inpath)
    
    # Export shapes to .js file 
    geojson_str = df.to_json(orient='records')
    with open(shapes_output_path, 'w') as output_file:
        output_file.write(format(geojson_str))
    
if __name__ == '__main__':
    main()