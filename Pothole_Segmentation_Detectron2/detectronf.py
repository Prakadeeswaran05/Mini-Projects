from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import cv2
cfg = get_cfg()
cfg.merge_from_file('/home/prak/my_cfg.yaml')

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = '/home/prak/output/model_final.pth'
cfg.MODEL.DEVICE='cpu'
predictor=DefaultPredictor(cfg)




def predict(image):
  MetadataCatalog.get(cfg.DATASETS.TRAIN).set(thing_classes=['pothole'])
  MetadataCatalog.get(cfg.DATASETS.TRAIN).set(thing_colors=[(128, 128, 128)])
  pothole_metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN)
  
  outputs = predictor(image)
  v = Visualizer(image[:, :, ::-1], pothole_metadata, scale=0.5,instance_mode=ColorMode.SEGMENTATION)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  
  out_im=v.get_image()[:, :, ::-1]
  return out_im
  


