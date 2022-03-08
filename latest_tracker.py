''' command: !python tracking_on_images.py --source '/content/images_folder' --csv 'detection_results.csv' --labels 'all_classes.txt' \
            --deep_sort_model 'resnet152' --fullcsv allImages.csv --save-vid --save-txt
'''

# limit the number of cpus used by high performance libraries
from cmath import nan
import os
import torch 
torch.cuda.empty_cache()
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import glob
import random
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from tqdm import tqdm

from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



import numpy as np

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def detect(opt):
    half, project, name, exist_ok = opt.half, opt.project, opt.name, opt.exist_ok

    # Get all required inputs
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    cfg.merge_from_file(opt.inputpaths)
    (source, csv_path, classes_path, fullcsv_path, 
    show_vid, save_vid, save_txt, flag ) = (cfg.DATA.IMAGESPATH, cfg.DATA.INFERENCECSV , cfg.DATA.ALLCLASSES , cfg.DATA.ALLIMAGESCSV,
                                      cfg.DATA.SHOWVIDEO ,cfg.DATA.SAVEVIDEO, cfg.DATA.SAVETXT, cfg.DATA.ASSETTYPE  )

    # Read csv , modify filenames and sort the df according to filename
    AllImage_data = pd.read_csv(fullcsv_path)
    data = AllImage_data.sort_values(['filename'], ascending=True )
    # data = data[1400:1600] 
    
    # Initialize deep sort
    deepsort = DeepSort(cfg.DEEPSORT.MODEL_TYPE, 
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, # len(AllImage_data)
                        n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True) 

    # Initialize device
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # List out all images
    if not os.path.isdir(source):
        print("[NOTE] : Please give correct images directory path")
    else:
        images =  sorted(glob.glob(os.path.join(source, '*.*')))

    # extract what is in between the last '/' and last '.' (Extract the filename)
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'
    save_path = str(Path(save_dir)) + '/' + txt_file_name + '.mp4'

    # Video writer
    im = cv2.imread(random.choice(images))
    # fps, w, h = 2, im.shape[1], im.shape[0] 
    fps, w, h = 2, 3840, 2160
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # read all class names
    names = {}
    with open(classes_path, 'r') as f:
        for i, line in enumerate(f):
            names[i] = str(line.strip()).replace("'", "")

    # Get the Detection results
    infer_data = pd.read_csv(csv_path)
    Infrdata = infer_data.sort_values(['filename'],ascending=True )
    InfrDataFilenameList = list(infer_data['filename'])
    # print(InfrDataFilenameList)

    # Initialize important variables
    Trackingdata = {}  # Tracking data
    frame_idx = 0     # 
    frame_index = 0    
    ids_bboxes = {}   # To resolve the bounding box shift 
    prev_ids = {}     # tracking ids as keys and classes as values
    frames_classes = []  # To store Previous frames's data

    # Start tracking  and except keyboard interruption
    try:   
      for iSeq, name in data.iterrows():
          # print(name)
          name = name['filename']
          frame_idx = frame_idx + 1
          # print(f'{name}')

          # check for the filename in inference csv 
          if name in InfrDataFilenameList:
                print("Name", name)
                # paath = os.path.join(source, name.split("_")[0],"ROW", name.split("_")[1]) # While running on batches 
                img_path = os.path.join(source, name)
                # print("img_path", img_path)
                img = cv2.imread(img_path) 
                h,w, _ = img.shape
                point1 = (w/2, h)
                # Except the cv2.error in case of pole tracking with segmentation results
                try:
                  annotator = Annotator(img, font_size=14, line_width=2, pil=not ascii) 
                  # Filter bounding boxes
                  bboxes =[]
                  classes = []
                  confs = []
                  Infrgroup = infer_data[(infer_data['filename']==name)]
                  for i, row in Infrgroup.iterrows():  
                    # if "POLE" in row['class_name']:              
                      box = [row['X1'], row['Y1'], row['X2'], row['Y2']] 
                      # In case of signboard u need to uncomment this block
                      if flag == 'signboard':
                        if type(row['Conf']) == str:
                          confs.append(0.78)
                        else:
                          confs.append(float(row['Conf'])) 
                        if row['class_name'] == "US ROUTE (FOR INDEPENDENT USE) (1":
                          row['class_name'] = "US ROUTE (FOR INDEPENDENT USE) (1,2 DIGITS)" 
                      elif flag == "pole":
                        confs.append(0.8)
                        # confs.append(float(row['Conf']))  # comment incase of signboard           
                      classes.append(list(names.values()).index(row['class_name']))
                      bboxes.append(box)
                  core_boxes = bboxes
                  # Convert bbox list to tensor data type
                  bboxes = torch.tensor(bboxes, dtype=torch.float)
                  clss = torch.tensor(classes, dtype = torch.float)
                  confs = torch.tensor(confs, dtype=torch.float)
                  print("\n")
                  # pass detections to deepsort
                  # print(bboxes)
                  xywhs = xyxy2xywh(bboxes)              
                  t4 = time.time()          
                  outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img, prev_ids, frames_classes, frame_index)  
                  # outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img)
                  t5 = time.time()
                  print("Frame Index : {}".format(frame_index))
                  # Iterate through each tracked box
                  if len(outputs) > 0: 
                      for j, output  in enumerate(outputs):                   
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            actual_ID = id 

                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]

                            # to PASCAL VOC format
                            xmin, ymin, xmax, ymax = bbox_left, bbox_top, bbox_left + bbox_w, bbox_top + bbox_h
                            point2 = (int((xmin+xmax)/2), int((ymin + ymax)/2))
                            angle = angle_between(point1, point2)
                            dist = np.linalg.norm(np.array(point1) - np.array(point2))

                            # Append bboxes corresponding to Tracking id and Finally to resolve the bounding box shift 
                            if len(ids_bboxes) == 0 or not id in ids_bboxes.keys(): 
                                ids_bboxes[id] = [] 
                            idboxes = ids_bboxes[id] 
                            if not bboxes.tolist() in idboxes: 
                                idboxes.append(bboxes.tolist()) 
                            else: 
                                continue

                            # Load previous frame tracking id, classid, frameindex
                            if len(prev_ids) == 0 and not id in list(prev_ids.keys()):
                                prev_ids[id] = []
                            prev_ids[id] = int(cls)
                            
                            c = int(cls)  # integer class
                            new_id = f"{id}_{c}"
                            ang = round(angle,2)
                            dist = round(dist, 2)
                            label = f'{id}_{c} {names[c]}'#_{ang}_{dist}' # {names[c]}
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            print(f"class {names[c]}")
                            print(f"Tracking ID {id}")
                            # Create a list for each tracking id 
                            if len(Trackingdata) == 0 or not new_id in list(Trackingdata.keys()):
                                Trackingdata[new_id] = []
                            modified_ID = id
                            save_txt = True

                            if save_txt:
                                  # Write MOT compliant results to file
                                  with open(txt_path, 'a') as f:
                                      line = f'{frame_index+1} {id}_{c} {bbox_left} {bbox_top} {bbox_w} {bbox_h} {ang} {dist}'
                                      f.write(line)
                                      f.write("\n")

                                  # Append bounding box information of each asset
                                  li = Trackingdata[new_id]
                                  
                                  if flag == "pole":
                                    val1 = Infrdata.loc[(Infrdata['filename'] == name) & (Infrdata['class_name'] == names[c]) & (Infrdata['X1'] == xmin)]#, Infrdata['class_name'] == names[c] ] #
                                    # print(val1)
                                    valid1 = str(val1['Asset_Id']).split("\n")[0].split(" ")[-1]
                                    li.append([name, bbox_left, bbox_top, 
                                                bbox_left+bbox_w, bbox_top+bbox_h, names[c], actual_ID, modified_ID , valid1])
                                  else:
                                    li.append([name, bbox_left, bbox_top, 
                                                bbox_left+bbox_w, bbox_top+bbox_h, names[c], actual_ID, modified_ID ])

                except (cv2.error):
                  print(name)
          else:
              deepsort.increment_ages()
              # LOGGER.info(f'{name} No detections')
          frame_index +=1

          # Stream results
          if name in InfrDataFilenameList:
            im0 = annotator.result()
            show_vid = False
            if show_vid:
                cv2.imshow(str(name), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
        
            # Save results (image with detections)
            if save_vid:       
                cv2.imwrite(f"./Images/{name}.jpg",im0)              
                vid_writer.write(im0)

    except KeyboardInterrupt:
      print("Keyboard Interrupted")
      pass

    if save_txt or save_vid:
          print('Results saved to %s' % save_path)
          if platform == 'darwin':  # MacOS
              os.system('open ' + save_path)

    # Save tracked data
    if flag == 'signboard':
      new_data = []
      singles = []
      for key, value in Trackingdata.items():
          if len(value) == 1:
            singles.append(value[0])
          elif len(value) == 2:
            new_data.append(value[1]+value[0])
          elif len(value) >2:
            new_data.append(value[-1] + value[-2])
      new_df1 = pd.DataFrame(new_data, columns = ["Filename", "xmin", "ymin", "xmax", "ymax", "Classname", "actual_ID", "modified_ID",
                                    "Filename", "xmin", "ymin", "xmax", "ymax",
                                          "Classname", "actual_ID", "modified_ID"])
      tracker_results_path1 = str(Path(save_dir)) + '/' + txt_file_name+"_SignboardTracker_Results.csv"
      new_df1.to_csv( tracker_results_path1, index=False)
      print('Results saved to %s' % tracker_results_path1)

      new_df2 = pd.DataFrame(singles, columns = ["Filename", "xmin", "ymin", "xmax", "ymax", 
                                                "Classname", "actual_ID", "modified_ID"])
      tracker_results_path2= str(Path(save_dir)) + '/' + txt_file_name+"_Singles_SignboardTracker_Results.csv"   
      new_df2.to_csv( tracker_results_path2, index=False)
      print('Results saved to %s' % tracker_results_path2)

    elif flag == 'pole':
      new_data = []
      for key, value in Trackingdata.items():
          if len(value) == 1:
            continue
            new_data.append(value[0])
          elif len(value) == 2:
            new_data.append(value[0])
            new_data.append(value[1])
          elif len(value) > 2:
            new_data.append(value[-2])
            new_data.append(value[-1])

      # print(new_data)
      new_df = pd.DataFrame(new_data, columns = ["Filename", "xmin", "ymin", "xmax", "ymax", 
                                  "Classname", "actual_ID", "modified_ID", "uniqueid"])
      tracker_results_path = str(Path(save_dir)) + '/' + txt_file_name+"_Pole_Tracker_Results.csv"
      new_df.to_csv( tracker_results_path, index=False)
      print('Results saved to %s' % tracker_results_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--inputpaths', type=str, default='Datapaths.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')


    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
