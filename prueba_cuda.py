
from ultralytics import YOLO
import torch
import cv2
import time
import numpy as np


threshold = 0.1

kernel = np.ones((5,5),np.uint8)

#torch.cuda.set_device(0) # Set to your desired GPU number
#model = YOLO('best.pt')
model = YOLO('yolov8n-pose.pt')
model.to('cuda')


# Open the video file

cap = cv2.VideoCapture(0)


_,bg=cap.read()

time.sleep(2)


print("backgorund adquired")
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video and status of the device 
    
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        result = model(frame,conf=0.5,classes=0)
        bbox = result[0].plot()
        cv2.imshow('imagenoriginal',cv2.resize(bbox, (720, 480)))
        boxes=result[0].boxes
        classID=boxes.cls.cpu().numpy()
        #mask=result[0].masks   
        if classID.all()==0:               
            
            
            
            mask=result[0].masks              
            boxes=result[0].boxes.xyxy.cpu().numpy().astype(int)
            (H, W) = frame.shape[:2]
              
                           
                          
            mask=mask.masks[0].cpu().numpy()

            mask = cv2.resize(mask, (W,H),interpolation=cv2.INTER_LINEAR)

            full_mask=np.zeros_like(frame[:,:,0],dtype=np.uint8)
                
            full_mask = (mask > threshold).astype(np.uint8)*255
                
            full_mask = cv2.dilate(full_mask,kernel,iterations=20)

            frame[full_mask==255] = bg[full_mask==255]
            
        
        cv2.imshow('imagen',cv2.resize(frame, (720, 480)))
        
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window

cap.release()
cv2.destroyAllWindows()