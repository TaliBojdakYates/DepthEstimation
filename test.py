import numpy as np
import cv2
import h5py
import os
import tensorflow as tf
from keras.models import load_model
from defs import Data_Entry, depth_loss


def rmse(true_values,predicted_values):
    diff = true_values - predicted_values
    squared_diff= diff **2
    mean_squared_diff = np.mean(squared_diff)
    root_mean_squared_error = np.sqrt(mean_squared_diff)

    return root_mean_squared_error


def absolute_relative_error(true_values, predicted_values):
    
    valid_indices = true_values != 0 #only get the none zero values otherwise we would be dividing by 0
    true_values = true_values[valid_indices]
    predicted_values = predicted_values[valid_indices]
    errors = np.abs((true_values - predicted_values) / true_values)
    mean_are = np.mean(errors)
   
    return mean_are


model = load_model('Models/ResNet101_Model.keras', custom_objects={'depth_loss': depth_loss})
test_root = 'test/'

test_names = []
for file in os.listdir(test_root):
    test_names.append(file)


total_rmse = 0
total_are = 0

length = len(test_names)

for f in test_names:

    try:
        path =  'test/' + f
        
        with h5py.File(path) as data_file:

            entry = Data_Entry(data_file)
            image_data = np.array(entry.image)
            true_depth = np.array(entry.depths)
            # entry.get_images()

        t_point = np.expand_dims(image_data, axis=0)
        depths = model.predict(t_point)

        depths = np.squeeze(depths, axis=-1)
        depths = np.squeeze(depths, axis=0)
        predicted_depths_normalized = (depths - np.min(depths)) / (np.max(depths) - np.min(depths)) 
        true_depth_normalized = (true_depth - np.min(true_depth)) / (np.max(true_depth) - np.min(true_depth))

        # depth_image = np.uint8(predicted_depths_normalized* 255)
        # depth_map = cv2.applyColorMap(depth_image, cv2.COLORMAP_MAGMA)
        # cv2.imshow("Depth Map", depth_map)
        # cv2.waitKey(0)
        
        rmse_current = rmse(true_depth_normalized, predicted_depths_normalized)
        are_current = absolute_relative_error(true_depth_normalized, predicted_depths_normalized)
        total_rmse += rmse_current
        total_are += are_current

    except Exception as e:
        print(e)
        length -= 1

average_rmse = total_rmse / length
average_are = total_are / length

print(length)
print(average_are)
print(average_rmse)
