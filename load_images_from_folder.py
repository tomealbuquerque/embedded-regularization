from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import os
import time

def create_image_list_from_folder(folder):
    image_list = []
    for filename in os.listdir(folder):
        image_list.append(filename)
        
    return image_list


def preprocess(folder=os.getcwd()):
    start_time=time.time()
    os.mkdir("preprocess_imgs")
    image_list = create_image_list_from_folder(folder=folder)
    for filename in image_list:
        img = io.imread(os.path.join(folder,filename))
        #Do preprocessing stuf....
        pre_process_img = resize(image=img, output_shape=(224, 224))
        new_folder = "preprocess_imgs"
        io.imsave(fname=os.path.join(new_folder,filename), arr=pre_process_img)
    elapsed_time = time.time() - start_time
    return elapsed_time

create_image_list_from_folder(folder="data_NCI")

tempo=preprocess(folder="data_NCI")

print('tempo total: ',tempo)