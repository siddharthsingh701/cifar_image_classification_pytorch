import matplotlib.pyplot as plt
import numpy as np 
import glob
import os

def show_image(data,class_names):
    image = data[0] 
    label = class_names[data[1]]
    plt.imshow(np.transpose(image.numpy(),(1,2,0)))
    print('\t\tLabel :',label,'\n')

def test_model(test_data,class_names,model):
  x = np.random.randint(0,10000)
  TEST = test_data[x]
  show_image(TEST,class_names)
  with torch.no_grad():
    new_prediction = model(TEST[0])
    print("Predicted Label ----> ",class_names[new_prediction.argmax().item()])

def count_folders_inside(directory):
    # Use glob to find all folders inside the given directory
    folders = glob.glob(os.path.join(directory, '*/'))
    # Count the number of folders
    num_folders = len(folders)
    return num_folders