import time
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scipy.spatial.distance import euclidean as euc
from google.colab import drive

drive.mount('/content/drive')
label='drive/My Drive/dataset/AI/darknet/coco.names'
LABELS = open(label).read().strip().split("\n")

weight='drive/My Drive/dataset/AI/darknet/yolov3.weights'
config='drive/My Drive/dataset/AI/darknet/yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(config, weight)


def midpoint(top,bottom):
  ans=[0.5*abs(top[0]+bottom[0]),0.5*abs(top[1]+bottom[1])]
  return ans
  
  
  
def calweight(x,y,w,h):
  topleft,bottomleft=(x,y+h),(x,y)
  topright,bottomright=(x+w,y+h),(x+w,y)
  #print(topright,bottomright)
  centertop=midpoint(topleft,topright)
  #print(centertop)
  centerleft=midpoint(topleft,bottomleft)
  centerright=midpoint(topright,bottomright)
  #print(centerleft)
  midway=(centerleft[0]+w/2,centertop[1]-h/2)
  ##For circumference
  radius=abs((midway[1]-centertop[1])/3)
  print(midway[1])
  print(centertop[1])
  bodylength=midway[0]+(midway[0]+centerright[0])/2
  heartgrith=2*3.14*radius
  weight=(heartgrith*heartgrith*bodylength*((0.393700787)**3))/300
  return weight
  
  
cap = cv2.VideoCapture('new.mp4')
boxes=[]
confidences=[]
classIDs=[]
idx_flatten=[]
image=[]
count=0
while(cap.isOpened()):
  count=count+5
  if(count==120):
      break
  cf = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1 
  cap.set(cv2.CAP_PROP_POS_FRAMES, cf+50)
  ret,img=cap.read()
  (H, W) = img.shape[:2]
  ln = net.getLayerNames()
  ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
      swapRB=True, crop=False)
  net.setInput(blob)
  start = time.time()
  layerOutputs = net.forward(ln)
  end = time.time()
    # show timing information on YOLO
  print("[INFO] YOLO took {:.6f} seconds".format(end - start))

  for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if (confidence > 0.60 and (classID==19)):
          image.append(img)
          box = detection[0:4] * np.array([W, H, W, H])
          (centerX, centerY, width, height) = box.astype("int")
          x = int(centerX - int(width / 2))
          y = int(centerY - int(height / 2))
          boxes.append([x, y, int(width), int(height)])
          confidences.append(float(confidence))
          classIDs.append(classID)
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6,0.6)
  idx_flatten.append(idxs)



a=np.amax(boxes, axis=0)
index=0
for box in boxes:
  if(np.array(box).all()==a.all()):
    break
  else:
    index=index+1
copy=image[index]
(x,y)=(boxes[index][0],boxes[index][1])
(w,h)=(boxes[index][2],boxes[index][3])
weight=int(calweight(x,y,w,h))
cv2.rectangle(copy, (x, y),(x + w, y + h), 2)
text = "{}: {:.4f}".format('Weight(lbs)', weight)
cv2.putText(copy, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, 2)
cv2_imshow(copy)
