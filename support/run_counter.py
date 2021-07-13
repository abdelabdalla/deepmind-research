import os

DIR = 'D:\\Users\\abdel\\Documents\\Disso_New_Data\\Run'
for i in range(1, 201):
    print(str(i) + ': ' + str(len([name for name in os.listdir(DIR + str(i)) if os.path.isfile(os.path.join(DIR + str(i), name))])))
