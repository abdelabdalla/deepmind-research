# trace generated using paraview version 5.9.0

#### import the simple module from the paraview
import fnmatch
import os

from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

files_path = 'D:\\Users\\abdel\\Documents\\Disso_New_Data'

for i in range(1, 201):

    print('folder: ' + str(i))
    f_p = files_path + '/Run' + str(i)

    run = False

    try:
        list_of_files = os.listdir(f_p)
        run = True

    except:
        print('folder: ' + str(i) + ' doesn\'t exist')
        run = False

    if run:
        pattern = "flow_*.vtu"
        list_of_names = []

        for entry in list_of_files:
            if fnmatch.fnmatch(entry, pattern):
                list_of_names.append(entry)

        list.sort(list_of_names)

        list_of_paths = [f_p + '/' + s for s in list_of_names]

        # create a new 'XML Unstructured Grid Reader'
        flow_00 = XMLUnstructuredGridReader(registrationName='flow_*',
                                            FileName=list_of_paths)
        flow_00.CellArrayStatus = []
        flow_00.PointArrayStatus = ['Pressure', 'Velocity', 'Nu_Tilde', 'Pressure_Coefficient', 'Density',
                                    'Laminar_Viscosity', 'Skin_Friction_Coefficient', 'Heat_Flux', 'Y_Plus',
                                    'Eddy_Viscosity']
        flow_00.TimeArray = 'TimeValue'

        # save data
        os.makedirs('D:\\Users\\abdel\\Documents\\Disso_New_Data_Updated\\run' + str(i))
        SaveData('D:\\Users\\abdel\\Documents\\Disso_New_Data_Updated\\run' + str(i) + '\\flow.vtk', proxy=flow_00,
                 ChooseArraysToWrite=0,
                 PointDataArrays=['Density', 'Eddy_Viscosity', 'Heat_Flux', 'Laminar_Viscosity', 'Nu_Tilde', 'Pressure',
                                  'Pressure_Coefficient', 'Skin_Friction_Coefficient', 'Velocity', 'Y_Plus'],
                 CellDataArrays=[],
                 FieldDataArrays=[],
                 VertexDataArrays=[],
                 EdgeDataArrays=[],
                 RowDataArrays=[],
                 Writetimestepsasfileseries=1,
                 Firsttimestep=0,
                 Lasttimestep=-1,
                 Timestepstride=1,
                 Filenamesuffix='_%04d',
                 FileType='Binary')
