import os
from shutil import copyfile

folder = 'D:\\Users\\abdel\\Documents\\Disso_New_Data'
base = folder + '\\Runbase'

for i in range(1, 201):
    new_folder = folder + '\\Run' + str(i)
    os.mkdir(new_folder)
    copyfile(base + '\\jet.cfg', new_folder + '\\jet.cfg')
    copyfile(base + '\\jet.geo', new_folder + '\\jet.geo')
    copyfile(base + '\\jet.su2', new_folder + '\\jet.su2')

    cfg_file = open(new_folder + '\\jet.cfg', 'r')
    cfg_lines = cfg_file.readlines()
    cfg_lines[150] = 'MARKER_INLET= ( inlet, 288.15, ' + str(i) + ', 0.0, -1.0, 0.0 )'

    cfg_file = open(new_folder + '\\jet.cfg', 'w')
    cfg_file.writelines(cfg_lines)
    cfg_file.close()
