import fnmatch
import os
import random
import subprocess
import time

import meshio
import numpy as np
import networkx as nx
from shutil import copyfile
from graph_nets import graphs


# point locations and velocity data from solution to form graph
def create_graph(path):
    mesh = meshio.read(path)
    point_location = np.delete(mesh.points, 2, 1)
    point_vel = np.delete(mesh.point_data['Velocity'], 2, 1)
    point_connections = mesh.cells_dict['triangle']

    return point_location, point_vel, point_connections


# find all relevant solution files and generate list of graphs
def create_graph_list(folder_path):
    list_of_files = os.listdir(folder_path)
    pattern = "flow_*.vtu"
    list_of_names = []

    for entry in list_of_files:
        if fnmatch.fnmatch(entry, pattern):
            list_of_names.append(entry)

    list.sort(list_of_names)

    list_of_paths = [folder_path + '/' + s for s in list_of_names]

    meshes = []
    i = 0
    while i < len(list_of_paths):
        print(str(i))
        a, b, c = create_graph(list_of_paths[0])
        meshes.append()
        i = i + 1

    return meshes


solution_path = '/Volumes/Samsung/Jet Mesh'
rollout_num = 1200

start = time.time()

graph_list = []
for i in range(0, 1):
    print('creating graph ' + str(i))
    graph_list.append(create_graph_list(solution_path + '/run' + str(i)))

end = time.time()

t = (end-start)/60

print('Total time: ' + str(t) + ' minutes')
