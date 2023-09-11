import os
from shutil import copyfile

input_path = r'D:\GDrive\Thesis\CodeBackUp (ThesisFinalWar)\Dataset\CleanFinal'
output_path = r'D:\GaitAuthentication30Nov2019\Storage\HighEffortRawData'
attack_folders = ['Attempt1', 'Attempt2', 'Attempt3']
files = ['clean_Gyr.txt', 'clean_LAcc.txt', 'clean_Mag.txt', 'clean_RVec.txt']
user_list = []
for i in range(1, 19):
    user_list.append("User" + str(i))
print(user_list)

for user in user_list:
    curr_user_folder = os.path.join(output_path, user)

    try:
        os.mkdir(curr_user_folder)
    except:
        print('Failed to create a folder')

    for folder in attack_folders:
        curr_input_folder = os.path.join(input_path, user, folder)
        curr_output_folder = os.path.join(curr_user_folder, folder)
        try:
            os.mkdir(curr_output_folder)
        except:
            print('Failed to create a folder')

        for file in files:
            curr_inputfile_path = os.path.join(curr_input_folder, file)
            curr_outputfile_path = os.path.join(curr_output_folder, file)
            print('curr_inputfile_path', curr_inputfile_path)
            print('curr_outputfile_path', curr_outputfile_path)
            copyfile(curr_inputfile_path, curr_outputfile_path)
