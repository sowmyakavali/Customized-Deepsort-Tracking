# limit the number of cpus used by high performance libraries
import os
import torch 
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import cv2
import time
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from save_utils import savecsvs, create_visualization

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (LOGGER,xyxy2xywh, increment_path)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def Track(opt):
    t1 = time.time()
    half, project, name, exist_ok = opt.half, opt.project, opt.name, opt.exist_ok

    # Get all required inputs
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    cfg.merge_from_file(opt.inputpaths)
    (source, csv_path, classes_path, fullcsv_path, 
    save_txt, flag, shape, classwisevid, multiclassvid ) = (cfg.DATA.IMAGESPATH, cfg.DATA.INFERENCECSV , cfg.DATA.ALLCLASSES , cfg.DATA.ALLIMAGESCSV,
                                      cfg.DATA.SAVETXT, cfg.DATA.ASSETTYPE, cfg.DATA.SIZE, cfg.DATA.CLASSWISEVIDEO, cfg.DATA.MULTICLASSVIDEO )

    # Read csv , modify filenames and sort the df according to filename
    AllImage_data = pd.read_csv(fullcsv_path)
    data = AllImage_data.sort_values(['filename'], ascending=True )
    # data = data[1400:1600] 
    
    # Initialize deep sort
    deepsort = DeepSort(cfg.DEEPSORT.MODEL_TYPE, 
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, 
                        n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True) 

    # Initialize device
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Directories to save results
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # List out all images
    if not os.path.isdir(source):
        print("[NOTE] : Please give correct images directory path")

    # path to save the files (Extract the filename)
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'
    save_path = str(Path(save_dir)) + '/' + txt_file_name #+ '.mp4'

    # Video writer parameters
    fps, w, h = 2, shape[0], shape[1]

    # read all class names
    names = {}
    names_list = []
    with open(classes_path, 'r') as f:
        for i, line in enumerate(f):
            names[i] = str(line.strip()).replace("'", "")
            names_list.append(str(line.strip()).replace("'", ""))

    # Initialize important variables
    Trackingdata = {}  # Tracking data
    frame_index = 0    
    ids_bboxes = {}   # To resolve the bounding box shift 
    prev_ids = {}     # tracking ids as keys and classes as values
    frames_classes = {}
    total_text = "ClassName     Count\n"

    # Inference results
    infer_data = pd.read_csv(csv_path)

    # Iterate through each class
    for ClassName in names_list:
        singles = []  
        for i, row in infer_data.iterrows():

            if row['class_name'] == ClassName:

                if flag == 'signboard':
                    singles.append([row['filename'], row['class_name'], row['X1'], row['Y1'], row['X2'], row['Y2'], row['Conf']])
                else:
                    singles.append([row['filename'], row['class_name'], row['X1'], row['Y1'], row['X2'], row['Y2'], 0.8, row['uniqueid']])

        if flag == 'pole':
            singleclassdf = pd.DataFrame(singles, columns = ['filename', 'class_name', 'X1', 'Y1', 'X2', 'Y2', 'conf', 'uniqueid']) 
        else:     
            singleclassdf = pd.DataFrame(singles, columns = ['filename', 'class_name', 'X1', 'Y1', 'X2', 'Y2', 'conf'])
        # dataframe for each class
        Infrdata = singleclassdf.sort_values(['filename'], ascending=True )
        InfrDataFilenameList = list(singleclassdf['filename'])
        # print(ClassName, len(Infrdata))
        total_text = total_text + f"{ClassName} :   {len(Infrdata)}\n"

        # Start tracking  and except keyboard interruption
        if len(singleclassdf):
            if classwisevid:
                save_path2 = str(Path(save_dir)) + '/' + txt_file_name +'_' + ClassName + '.mp4'
                vid_writer1 = cv2.VideoWriter(save_path2, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            try:  
                # Flow with the sequence 
                for iSeq, row in tqdm(data.iterrows()):
                    name = row['filename']

                    # check for the filename in inference csv 
                    if name in InfrDataFilenameList:
                        paath = os.path.join(source, name.split("_")[0],"ROW", name.split("_")[1]) # While running on batches
                        img = cv2.imread(paath)

                        # Except the cv2.error in case of pole tracking with segmentation results
                        try:
                            # Initialize annotator to draw bboxes
                            annotator = Annotator(img, font_size=14, line_width=2, pil=not ascii) 

                            # Filter bounding boxes
                            bboxes =[]
                            classes = []
                            confs = []
                            Infrgroup = Infrdata[(Infrdata['filename']==name)]
                            # Collect bounding boxes and pass them to deepsort
                            for i, row in Infrgroup.iterrows():  

                                box = [row['X1'], row['Y1'], row['X2'], row['Y2']] 

                                if flag == 'signboard':

                                    if type(row['conf']) == str:
                                        confs.append(0.78)
                                    else:
                                        confs.append(float(row['conf'])) 
                                    if row['class_name'] == "US ROUTE (FOR INDEPENDENT USE) (1":
                                        row['class_name'] = "US ROUTE (FOR INDEPENDENT USE) (1,2 DIGITS)" 

                                elif flag == "pole":
                                    confs.append(0.8)

                                classes.append(list(names.keys())[list(names.values()).index(row['class_name'])])
                                bboxes.append(box)

                            # Convert bbox list to tensor data type
                            bboxes = torch.tensor(bboxes, dtype=torch.float)
                            clss = torch.tensor(classes, dtype = torch.float)
                            confs = torch.tensor(confs, dtype=torch.float)
                            
                            # pass detections to deepsort
                            xywhs = xyxy2xywh(bboxes)   
                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img, prev_ids, frames_classes,frame_index)  

                            # Iterate through each tracked box
                            if len(outputs) > 0: 
                                for j, output  in enumerate(outputs):                   
                                    bboxes = output[0:4]
                                    id = output[4]
                                    cls = output[5]

                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]

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
                                    prev_ids[id] = int(cls)  # Using while tracking chevrons

                                    # integer class
                                    c = int(cls)  

                                    # Modify the track id
                                    new_id = f"{c}_{id}" 

                                    # Text to display on image
                                    label = f'{c}_{id} {names[c]}'
                                    annotator.box_label(bboxes, label, color=colors(c, True))

                                    # Create a list for each tracking id 
                                    if len(Trackingdata) == 0 or not new_id in list(Trackingdata.keys()):
                                        Trackingdata[new_id] = []

                                    if save_txt:
                                        # Write MOT compliant results to text file
                                        with open(txt_path, 'a') as f:
                                            line = f'{name} {names[c]} {bbox_left} {bbox_top} {bbox_left+bbox_w} {bbox_top + bbox_h} {c}_{id}'
                                            f.write(line)
                                            f.write("\n")

                                    # Append bounding box information of each asset
                                    li = Trackingdata[new_id]
                                    if flag == "pole":
                                        val1 = Infrdata.loc[(Infrdata['filename'] == name) & (Infrdata['class_name'] == names[c]) & (Infrdata['X1'] == bbox_left)]#, Infrdata['class_name'] == names[c] ] #
                                        valid1 = str(val1['uniqueid']).split("\n")[0].split(" ")[-1]
                                        li.append([name, bbox_left, bbox_top, 
                                                    bbox_left+bbox_w, bbox_top+bbox_h, names[c], new_id, valid1])
                                    else:
                                        li.append([name, bbox_left, bbox_top, 
                                                bbox_left+bbox_w, bbox_top+bbox_h, names[c], new_id]) 

                        except (cv2.error):
                            print('cv2 Error',name)
                        if classwisevid:
                            im0 = annotator.result()
                            # # cv2.imwrite(f"/content/drive/MyDrive/Results/{frame_index}{name}.jpg",im0)              
                            vid_writer1.write(im0)
                    else:
                        deepsort.increment_ages()
                        # LOGGER.info(f'{name} No detections')
                    frame_index +=1
            except KeyboardInterrupt:
                print("Keyboard Interrupted")
                pass

    savecsvs(flag, Trackingdata, save_path)       
    read_path = str(Path(save_dir)) + '/' + "Readme.txt"
    t2 = time.time()
    with open(read_path, 'w') as f:
        text = total_text + f"\nTime Taken for Tracking is : {t2 - t1} s"
        f.write(text)

    if multiclassvid:
        create_visualization(list(data['filename']), txt_path, source, names, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        Track(opt)
