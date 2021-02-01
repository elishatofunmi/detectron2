import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import json
import cv2
from detectron2.structures import BoxMode
os.environ["CUDA_VISIBLE_DEVICES"]=""

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

setup_logger()
   

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


# import some common libraries
import numpy as np
import os, sys, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

video_dir = sys.argv[1]

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)



outputs = predictor(im)


cap = cv2.VideoCapture(video_dir)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(vid_directory,fourcc,20.0,(640,480))
import time

while (cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outputs = predictor(gray)
        cv2.imshow('frame',outputs)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    # out.write(frame)
        
cap.release()
# out.release()
cv2.destroyAllWindows()


