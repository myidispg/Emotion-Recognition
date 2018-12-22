import os
import gc
import matplotlib.pyplot as plt

data_dir_base = '../Emotion-Dataset/cohn-kanade-images/'

# Get a list of all the subject directories in the dataset folder.
folders_list = os.listdir(data_dir_base)

# All the subfolders in the directories for each subject
sub_folders_dict = {}
for folder in folders_list:
    sub_folders_dict[folder] = os.listdir(data_dir_base + folder)
# Since some folders have a .DS_Store, removing it
for folder in sub_folders_dict:
    valid_folders = []
    for sub_folder in sub_folders_dict[folder]:
        if sub_folder != '.DS_Store':
            valid_folders.append(sub_folder)
    sub_folders_dict[folder] = valid_folders
    
# Cleaning up some unused variables
del folder, folders_list, sub_folder, valid_folders
gc.collect()

count_total_samples = {
        '001': 0,
        '002': 0,
        '003': 0,
        '004': 0,
        '005': 0,
        '006': 0,
        '007': 0,
        '008': 0,
        '009': 0,
        '010': 0,
        '011': 0,
        '012': 0,
        '013': 0
        }

for folder in sub_folders_dict:
    for sub_folder in sub_folders_dict[folder]:
        count_total_samples[sub_folder] += len(os.listdir('../Emotion-Dataset/cohn-kanade-images/' + folder + '/' + sub_folder))
        
# We will keep only the first 7 emotions since only they have a good amount of images.
plot_x = []
plot_y = []
for folder in count_total_samples:
    plot_x.append(folder)
    plot_y.append(count_total_samples[folder])
    
plt.bar(plot_x, plot_y)
#plt.yticks(plot_y)
#plt.xticks(plot_x)
plt.xlabel('Count')
plt.ylabel('Folder')
plt.show()

# Again, remove some unwanted variables
del folder, plot_x, plot_y, sub_folder
gc.collect()

"""
Since the bar graph shows there is a respectable amount of samples in directories 1 to 7, we will remove 8 to 13.
"""
folders_to_remove = ['008', '009', '010', '011', '012', '013']

for folder in sub_folders_dict:
    for sub_folder in sub_folders_dict[folder]:
        new_path = os.path.join(data_dir_base, folder, sub_folder)
        if sub_folder in folders_to_remove and os.path.exists(new_path):
            os.system('rm -rf ' + new_path)

# To remove the unnecessary category folders from sub_folders_dict
sub_folder_dict = {}
for folder in sub_folders_dict:
    sub_folder_dict[folder] = []
    for sub_folder in sub_folders_dict[folder]:
        sub_folder_dict[folder].append(sub_folder)
        
# Clear unnecessary memory
del count_total_samples, folder,  new_path, sub_folder, sub_folders_dict, folders_to_remove
gc.collect()
