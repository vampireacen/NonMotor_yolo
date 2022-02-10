import threading
import torch.nn as nn
import torch
import requests
import cv2
import numpy as np
class MyThread(threading.Thread):
    
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func # 
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None
def videocapture():
    return frame
class RetrogradeModel(nn.Module):
    def __init__(self, *args):
        super(RetrogradeModel, self).__init__()
        
if __name__ == '__main__':
    frame = videocapture()
    thread_retrograde = MyThread(func=inference_detector,args=(frame,))
    thread_vehicle_type = MyThread(func=inference_detector,args=(frame,))
    thread_helmet = MyThread(func=inference_detector,args=( frame,))
    thread_manned = MyThread(func=inference_detector,args=(frame,))
    thread_protective_gear = MyThread(func=inference_detector,args=(frame,))