import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch




if __name__ == '__main__':
    #with torch.no_grad():

        model = YOLO('yolov8RFAConv1.yaml')

        # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度

        model.train(data='flow.yaml',
                # 如果大家任olov8smallaim2务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                #cache=False,
                imgsz=640,
                epochs=300,
                single_cls=False,  # 是否是单类别检测
                batch=8,
                close_mosaic=10,

                device='0',
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                lr0=0.01
                )