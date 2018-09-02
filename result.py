import pandas as pd
import numpy as np
import scipy.spatial
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

ENCODINGS = '/home/aidris/Videos/task/duplicateimages/ImageDuplication/src/encodings.pkl'

def similarity_metric(x1,x2,type = 'euclidean'):
    """
    CURRENTLY IMPLEMENTS EUCLIDEAN METRIC

    """
    if type == 'euclidean':
        return sum((x1-x2)**2)**(1/2)
    
    if type=='cosine':
        return scipy.spatial.distance.cosine(x1,x2)

#LOAD ENCODINGS

def load_encodings(path=ENCODINGS):
    encodings = pd.read_pickle(path)
    #encodings.set_index(['images'],inplace=True)
    return encodings

data = load_encodings()
print(data.head())

encodings = data['ENCODINGS'].values
images_ids = data['images'].values
distance=defaultdict(list)
d=[]
for i,encoding1 in enumerate(encodings):
    dis=[]
    for j,encoding2 in enumerate(encodings):
        
        similarity = similarity_metric(encoding1,encoding2)
        distance[images_ids[i]].append([images_ids[j],similarity])
        dis.append([images_ids[j],similarity])
    d.append(dis)

dict_={}
for i,list1 in enumerate(d):
    list1.sort(key=lambda x: x[1])
    dict_[images_ids[i]]=list1

#print(dict_.keys())

print(dict_['../test_images/17 Oncoanaesthesia CEM Review_1/image26.jpg'][1:10])

l = plt.imread('../test_images/17 Oncoanaesthesia CEM Review_1/image26.jpg')
plt.imshow(l)
plt.show()
data.set_index(['images'],inplace=True)
print(similarity_metric(data.loc['../test_images/17 Oncoanaesthesia CEM Review_1/image26.jpg','ENCODINGS'],data.loc['../test_images/17 Oncoanaesthesia CEM Review_1/image19.jpg','ENCODINGS'],type='cosine'))









"""

#print(data.head())
distances=[]
result_data=[]

for j, x2 in enumerate(data['ENCODINGS']):
    imp=[]
    for i,x1 in enumerate(data['ENCODINGS']):
        
        dist=similarity_metric(x1,x2)
        imp_data = [i,dist]
        imp.append(imp_data)
        distances.append(dist)
    result_data.append(imp)


#result = np.array(result_data)
#print(result.shape)
    #min_e= np.min(distances)
    #print('Index ',min_e)

#n=[]
#for list1 in result_data:
#    list1.sort(key=lambda x: x[1])
#    n.append(list1)
#n=np.array(n)
#print(n[145])
    
#print(data.iloc[145,0])
#print(data.iloc[142../test_images2/17 Oncoanaesthesia CEM Review_1/image1,0])

data.set_index(['images'],inplace=True)
print(data.head())
print(similarity_metric(data.loc['../test_images2/17 Oncoanaesthesia CEM Review_1/image26.png','ENCODINGS'],data.loc['../test_images2/17 Oncoanaesthesia CEM Review_1/image19.jpeg','ENCODINGS']))
print(similarity_metric(data.loc['../test_images2/17 Oncoanaesthesia CEM Review_1/image14.tiff','ENCODINGS'],data.loc['../test_images2/17 Oncoanaesthesia CEM Review_1/image14.tiff','ENCODINGS'],type='cosine'))
"""






