##
##final code
##

import docx2txt
import os
import subprocess

#path to base project 
PATH = os.path.dirname('/home/aidris/Videos/task/duplicateimages/ImageDuplication/')

#path to the folder containing the docx files 
SOURCE = os.path.join(PATH,'Image_Duplication_Use_Case 1')


def doc_to_docx():

    """
    converts doc files to docx

    """


    #print(os.listdir(SOURCE))
    files_doc = [file for file in os.listdir(SOURCE) if os.path.splitext(file)[1]=='.doc']
    #print(print(files_doc))
    f = lambda x: os.path.join(SOURCE,x)
    files_doc = list(map(f,files_doc))
    #print(files_doc)
    #print('Found {} doc files'.format(len(files_doc)))
    #print(files_doc)

    print('Converting to docx if on linux installation....')
    print('\nPress 1 to Continue or  2 to Quit')

    #stores the user input 
    i = int(input())

    if i==2:

        """
        this option works when the user doesn't want to
        convert the .doc files to .docx format
        """ 

        exit

    elif i==1:
        print(files_doc)
        for filename in files_doc:
            subprocess.call(['soffice','--headless','--convert-to','docx',filename])
    
    converted_files =  [file for file in os.listdir(SOURCE) if os.path.splitext(file)[1]=='.docx']
    converted_files= list(map(f,converted_files))
    
    for i in converted_files:
        subprocess.call(['mv',i,'UseCase1'])

source_new = os.path.join(PATH,'UseCase1')
print(source_new)
def main(SOURCE=source_new):

    """
    main function that runs through the source tree to get extract the images
    """


    for root, dirs, filenames in os.walk(SOURCE):
        
        for f in filenames:

            try:
                #debug
                filename, file_extension = os.path.splitext(f)
                print(filename,file_extension)
                
                directory = os.path.join(PATH, "images/%s" % filename)
                
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print('trying to extract images')
                
                source_file = os.path.join(SOURCE,f)
                print(source_file)

                #function for extracting the images from docx file
                docx2txt.process(source_file, directory)
                print('done')

            except:
                
                pass

if __name__ == "__main__":
    print('press 1 on first run')
    i = int(input())
    if i==1:
       doc_to_docx()
        

    print('trying to extract images from data')
    main()
