import os
import shutil
import random

def count_image(path):
    count = 0
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            count+=1
    return count


def merge_image(folder_path):

    subfolders = os.listdir(folder_path)

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path,subfolder)
        if os.path.isdir(subfolder_path):
            images = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path,f)) and f.endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(images)
    
            for i, image in enumerate(images):
                src_path = os.path.join(subfolder_path,image)
                dst_path = os.path.join(folder_path, f"{i+1}_{image}")
                shutil.move(src_path, dst_path)

            os.rmdir(subfolder_path)
    return "MERGE IMAGE SUCCESS"

def delete_image(folder_path, num_images):
    allFiles = os.listdir(folder_path)
    images = [file for file in allFiles if file.endswith(('.jpg', '.jpeg', '.png'))]
    
    for i in range(num_images,-1,-1):
        if i < len(images):
            os.remove(os.path.join(folder_path, images[i]))
        else:
            print(i)
            break
    
    return "DELETE SUCCESS {} IMAGE IN {} FOLDER!".format(num_images,folder_path)


incorrect_num = 1494
mask_num = 789
nomask_num = 746

incorrect_path = 'datasets/face_mask/incorrect_mask'
with_mask = 'datasets/face_mask/with_mask'
without_mask = 'datasets/face_mask/without_mask'

path = 'datasets/test_folder'
num = 2
# path_mc = 'dataset/face_mask/incorrect_mask/mc'
# path_mmc = 'dataset/face_mask/incorrect_mask/mmc'
# path_mask_simple = 'dataset/face_mask/with_mask/simple'
# path_nomask_simple = 'dataset/face_mask/without_mask/simple'

# delete_image(path_mc,incorrect_num)
# delete_image(path_mmc,incorrect_num)
# delete_image(path_mask_simple,mask_num)
# delete_image(path_nomask_simple,nomask_num)

# merge_image(incorrect_path)
# merge_image(with_mask)
merge_image(without_mask)
# delete_image(path,num)