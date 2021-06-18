import numpy as np
import cv2
import sys

#detectron imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo

#sort imports
from sort import *
tracker = Sort()
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.DEVICE='cpu'#remove this if you are using gpu
input_path = sys.argv[1]
cap=cv2.VideoCapture(input_path)
writer = None
while True:
  ret,image=cap.read()
  predictor=DefaultPredictor(cfg)
  outputs = predictor(image)
  v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5,)
  dets=[]
  for ind,data in enumerate(outputs["instances"].pred_classes):
    num=data.item()
    if num==0:
      box=outputs['instances'].pred_boxes[ind]
      box=box.tensor.cpu().numpy()
      box=box[0].tolist()
      score=float(outputs['instances'].scores[ind].item())
      #print(score)
      dets.append(box+[score])
      #print(dets)
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  
  dets = np.asarray(dets)
  #print(dets)
  tracks = tracker.update(dets)    
  boxes = []
  indexIDs = []
  c = []
  for track in tracks:
    boxes.append([track[0], track[1], track[2], track[3]])
    center = (int(((track[0]) + (track[2]))/2), int(((track[1])+(track[3]))/2))
    cv2.circle(image,center,3,(0,255,0),2)
    indexIDs.append(int(track[4]))
  if len(boxes) > 0:
    i = int(0)
    for box in boxes:
      # extract the bounding box coordinates
      (x, y) = (int(box[0]), int(box[1]))
      (w, h) = (int(box[2]), int(box[3]))

      color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
      cv2.rectangle(image, (x, y), (w, h), color, 2)
      text = "{}".format(indexIDs[i])
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
      i += 1
  if writer is None:
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('out.mp4', fourcc, 15,
      (image.shape[1], image.shape[0]), True)

  writer.write(image)      
  cv2.imshow('out',image)
  cv2.waitKey(3)
cap.release()
writer.release()
cv2.destroyAllWindows()  

  


