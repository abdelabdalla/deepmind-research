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

    point_list = []

    i = 0
    while i < point_location.shape[0]:
        point_list.append([i + 1,
                           {
                               'pos_x': point_location[i, 0],
                               'pos_y': point_location[i, 1],
                               'vel_x': point_vel[i, 0],
                               'vel_y': point_vel[i, 1]
                           }])
        i = i + 1

    g = nx.Graph()
    g.add_nodes_from(point_list)

    for points in point_connections:
        g.add_edge(points[0], points[1])
        g.add_edge(points[0], points[2])
        g.add_edge(points[1], points[2])

    return g


def create_g(path):
    mesh = meshio.read(path)
    point_location = np.delete(mesh.points, 2, 1)
    point_vel = np.delete(mesh.point_data['Velocity'], 2, 1)
    point_connections = mesh.cells_dict['triangle']

    return point_location, point_vel, point_connections


# find all relevant solution files and generate list of graphs
def create_graph_list(folder_path):
    list_of_files = os.listdir(folder_path)
    pattern = "flow_*.vtk"
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
        meshes.append(create_graph(list_of_paths[i]))
        i = i + 1

    return meshes


# generates the set of meshes to run in CFD
def meshing(path, cfg, num_meshes, density=0.05):
    p = os.path.split(path)
    p_new = p[0] + '/run'

    # original_geo = open(path, 'r')
    # original_geo_lines = original_geo.readlines()
    # original_geo.close()
    #
    # for i in range(0, num_meshes):
    #     b = round(random.uniform(0.1, 2.5), 3)
    #     h = round(random.uniform(0.1, 5), 3)
    #
    #     if not os.path.exists(p_new + str(i)):
    #         os.mkdir(p_new + str(i))
    #
    #     geo_file = open(p_new + str(i) + '/' + 'jet.geo', 'w')
    #     geo_lines = original_geo_lines
    #
    #     geo_lines[5] = 'Point(3) = {20, ' + str(h) + ' ,0, 0.05};\n'
    #     geo_lines[7] = 'Point(4) = {' + str(b) + ', ' + str(h) + ', 0, 0.05};\n'
    #     geo_lines[9] = 'Point(5) = {0, ' + str(h) + ', 0, 0.05};\n'
    #
    #     for line in geo_lines:
    #         geo_file.write(line)
    #
    #     geo_file.close()
    #
    #     subprocess.run(['gmsh', p_new + str(i) + '/' + 'jet.geo', '-2', '-format', 'su2', '-save_all'])
    #
    original_cfg = open(p[0] + '/' + cfg, 'r')
    original_cfg_lines = original_cfg.readlines()
    original_cfg.close()

    for i in range(21, num_meshes):
        vel = round(random.uniform(10, 50), 3)

        cfg_file = open(p_new + str(i) + '/' + 'jet.cfg', 'w')
        cfg_lines = original_cfg_lines

        cfg_lines[150] = 'MARKER_INLET= ( inlet, 288.15, ' + str(vel) + ', 0.0, -1.0, 0.0 )\n'
        cfg_lines[253] = 'MESH_FILENAME= jet.su2\n'

        for line in cfg_lines:
            cfg_file.write(line)

        cfg_file.close()

        print('running CFD run number ' + str(i))

        p_clean = p_new.replace(' ', '\\ ')
        command = 'cd ' + p_clean + str(i) + ' && ~/su2/SU2_CFD jet.cfg'

        #cfd_output_file = open(p_new + str(i) + '/' + 'telemetry.txt', 'w')

        subprocess.check_output(command, shell=True)

        #cfd_output_file.close()

    return p


solution_path = '/Volumes/Samsung/Jet Updated'
geo_path = '/Users/abdelabdalla/Google Drive/Jet Mesh/jetbase.geo'
config_name = 'jettime.cfg'
rollout_num = 1200

start = time.time()

# meshing(geo_path, config_name, rollout_num)

graph_list = []
for i in range(0, 1):
    print('creating graph ' + str(i))
    graph_list.append(create_graph_list(solution_path + '/run' + str(i)))

end = time.time()

t = (end-start)/60

print('Total time: ' + str(t) + ' minutes')
