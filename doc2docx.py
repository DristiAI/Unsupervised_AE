import subprocess
import os
 
files_doc = [file for file in os.listdir('.') if os.path.splitext(file)[1]=='.doc']
print('Found {} files'.format(len(files_doc)))
print(files_doc)

print('Converting to docx if on linux installation....')
print('\nPress 1 to Continue or  2 to Quit')
i = int(input())
if i==2:
    exit
elif i==1:
    for filename in files_doc:
        subprocess.call(['soffice','--headless','--convert-to','docx',filename])
converted_files =  [file for file in os.listdir('.') if os.path.splitext(file)[1]=='.docx']
for i in converted_files:
    subprocess.call(['mv',i,'UseCase1'])
