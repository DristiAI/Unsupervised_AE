import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from keras.utils import get_file
import subprocess
import os 

# paths definition
#abspath : path to project
#train_path : path to store the downloaded images

abs_path = '/home/aidris/Videos/task/ImageDuplication/'

train_PATH = os.path.join(abs_path,'url2')


def get_absolute_path(path,files):
    
    f = lambda x: os.path.join(path,x)
    files = list(map(f,files))
    return files


def download_data(path):

    """
    arguments: 
    path: pth to download images to

    """
    #print(path)
    files = [i for i in os.listdir(path) if os.path.splitext(i)[1]=='.txt']
    print(files)
    #get absolute path of file names
    abs_path_maker = lambda x : os.path.join(path,x) 

    files = list(map(abs_path_maker,files))
    #print('ooo')
    #print(files)
    for file in files:

        with open(file) as f:

            url = f.readline().rstrip()
        
            i=0
        
            while(url):
                
                #filename = os.path.splitext(file)[0]+'image'+str(i)+'.jpg'
                filename = 'url8'+'image'+str(i)+'.jpg'
                filename = os.path.join(path,filename)

                try:

                    #check if the file already present 

                    if not os.path.exists(filename):
                        img = get_file(fname=filename,origin=url)
                    else:
                        print('EXISTS!!!!!!!')
                    
                    i+=1
                    url = f.readline().rstrip()
                                        
                except:

                    url = f.readline().rstrip()



if __name__=='__main__':           
    download_data(train_PATH)

 
        
         








