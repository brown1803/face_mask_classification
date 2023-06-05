import os



def count_image(path):
    count = 0
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):

            count+=1
    return count

path = 'datasets/face_mask/with_mask'
print(count_image(path))