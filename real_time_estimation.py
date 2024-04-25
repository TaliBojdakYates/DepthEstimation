import cv2 
import numpy as np
import time
from defs import Data_Entry, depth_loss
from keras.models import load_model

model = load_model('Models/ResNet101_Model.keras', custom_objects={'depth_loss': depth_loss})

cap = cv2.VideoCapture(0) 

while True: 
  
    # get image from camera
    _, frame = cap.read() 
    image = np.array(frame)
   
    start_time = time.time()

    #format data and predict
    image_data = np.expand_dims(image, axis=0)
    predicted_depths = model.predict(image_data)

    #get depth from prediction and normalize
    predicted_depths = np.squeeze(predicted_depths, axis=-1)
    predicted_depths = np.squeeze(predicted_depths, axis=0)
    
    predicted_depths_normalized = (predicted_depths - np.min(predicted_depths)) / (np.max(predicted_depths) - np.min(predicted_depths)) 

    depth_image = np.uint8(predicted_depths_normalized* 255)
    depth_map = cv2.applyColorMap(depth_image, cv2.COLORMAP_MAGMA)
   
 
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = round(1/elapsed_time)
    cv2.putText(depth_map, f"FPS: {fps}", (30, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2) 
    
    cv2.imshow("frame", depth_map) 
  
    # quit with q
    if cv2.waitKey(1) == ord("q"): 
        break
  
# closing the camera and close opened windows
cap.release() 
cv2.destroyAllWindows() 