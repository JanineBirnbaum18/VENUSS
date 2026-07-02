import numpy as np

def initialize_topography(topo_file, x_height, y_height):
    topo_data = np.genfromtxt("example_slope_profile.txt", delimiter="\t")
    if (np.max(topo_data[1:,0])-np.min(topo_data[1:,0]))<1e-14:
        xs = topo_data[1:,3]
        ys = np.append(0,np.cumsum(-(topo_data[2:,3]-topo_data[1:-1,3])*np.tan(topo_data[1:-1,4]/180*np.pi)))
        ys -= np.min(np.interp(x_height,xs,ys))
    else:
        xs = topo_data[1:,0]
        ys = topo_data[1:,2]

    return (y_height - np.interp(x_height,xs,ys))


