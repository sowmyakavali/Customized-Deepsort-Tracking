import os 
import cv2
import pandas as pd
from tqdm import tqdm


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def savecsvs(flag, trackers_dict, savepath):
    TotalTrackers = []
    SingleTracker = []
    DoubleTrackers = []

    if flag == 'signboard':
        for key, value in trackers_dict.items():
            if len(value) == 1:
                SingleTracker.append(value[0])
                TotalTrackers.append(value[0])
            elif len(value) == 2:
                DoubleTrackers.append(value[1]+value[0])
                TotalTrackers.append(value[0])
                TotalTrackers.append(value[1])
            elif len(value) > 2:
                DoubleTrackers.append(value[-1] + value[-2])
                TotalTrackers.append(value[-2])
                TotalTrackers.append(value[-1])
        new_df1 = pd.DataFrame(DoubleTrackers, columns = ["Filename", "xmin", "ymin", "xmax", "ymax", "Classname", "Track_ID",
                                        "Filename", "xmin", "ymin", "xmax", "ymax", "Classname", "Track_ID"])
        tracker_results_path1 = savepath + "_SignboardTracker_Results.csv"
        new_df1.to_csv( tracker_results_path1, index=False)
        print('Results saved to %s' % tracker_results_path1)

        new_df2 = pd.DataFrame(SingleTracker, columns = ["Filename", "xmin", "ymin", "xmax", "ymax", "Classname", "Track_ID"])
        tracker_results_path2= savepath + "_Singles_SignboardTracker_Results.csv"   
        new_df2.to_csv( tracker_results_path2, index=False)
        print('Results saved to %s' % tracker_results_path2)

        total_df2 = pd.DataFrame(TotalTrackers, columns = ["Filename", "xmin", "ymin", "xmax", "ymax", "Classname", "Track_ID"])
        tracker_results_path3= savepath + "_Total_SignboardTracker_Results.csv"   
        total_df2.to_csv( tracker_results_path3, index=False)
        print('Results saved to %s' % tracker_results_path3)

    elif flag == 'pole':
        for key, value in trackers_dict.items():
            if len(value) == 1:
                TotalTrackers.append(value[0])
            elif len(value) == 2:
                TotalTrackers.append(value[0])
                TotalTrackers.append(value[1])
            elif len(value) > 2:
                TotalTrackers.append(value[-2])
                TotalTrackers.append(value[-1])

        new_df = pd.DataFrame(TotalTrackers, columns = ["Filename", "xmin", "ymin", "xmax", "ymax", 
                                    "Classname", "Track_ID", "uniqueid"])
        tracker_results_path = savepath + "_Pole_Tracker_Results.csv"
        new_df.to_csv( tracker_results_path, index=False)
        print('Results saved to %s' % tracker_results_path)


def create_visualization(allfilenames, trackedtxtfile, imagespath, allclasses, save_path):

    colors = Colors()  # create instance for 'from utils.plots import colors'
    df = []
    with open(trackedtxtfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split(" ")
            if len(values) == 8:
                x1, y1, x2, y2 = map(int, values[3:7])
                label = values[1] + values[2]
            elif len(values) == 9 :
                x1, y1, x2, y2 = map(int, values[4:8])
                label = values[1] + values[2] + values[3]
            else:
                x1, y1, x2, y2 = map(int, values[2:6])
                label = values[1] 
                
            classs, trckid = map(int, values[-1].split("_"))
            newid = f'{classs}_{trckid} {label}'
            df.append([values[0], newid, x1, y1, x2, y2])

    df = pd.DataFrame(df, columns=['filename', 'class_name', 'X1', 'Y1', 'X2', 'Y2'])

    fps, w, h = 2, 3840, 2160
    print("Writing images to video.....")
    vid_writer = cv2.VideoWriter(save_path + "Multiclass_Trackers_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    try:
        for filename in tqdm(allfilenames): #df['filename'].unique()
            if filename in df['filename'].unique():
                rows = df[(df['filename'] == filename)]
                image = cv2.imread(os.path.join(imagespath, filename.split("_")[0], "ROW", filename.split("_")[1])) #For batches

                for i, row in rows.iterrows():
                    x1 , y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
                    p1, p2 = (x1, y1), (x2, y2) 

                    label = row['class_name']
                    c = int(label.split("_")[0])
                    color = colors(c, True)
                    # label = f'{label} {allclasses[c]}'
                    lw = max(round(sum(image.shape) / 2 * 0.001), 2)
                    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                    txt_color=(255, 255, 255)

                    if label:
                            tf = max(lw - 1, 1)  # font thickness
                            w, h = cv2.getTextSize(label, 0, fontScale=lw / 2.3 , thickness=tf+1)[0]  # text width, height
                            outside = p1[1] - h - 3 >= 0  # label fits outside box
                            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled

                            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 2.3, txt_color,
                                        thickness=tf+3, lineType=cv2.LINE_AA)
                vid_writer.write(image)
    except KeyboardInterrupt:
        print("Keyboard Interrupted")
