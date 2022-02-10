# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image,ImageDraw, ImageFont 
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from tqdm import tqdm
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box, one_box
from utils.torch_utils import select_device, time_sync
from util import CTCLabelConverter, AttnLabelConverter
from model import Model

@torch.no_grad()
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_color = text_color[::-1]
    text_color_bg = text_color_bg[::-1]
    # text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img_PIL)
    text_w,text_h = draw.textsize(text,font)
    draw.rectangle(((x,y),(x+text_w,y+text_h)),fill=text_color_bg)
    draw.text((x,y), text, font=font, fill=text_color)

    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    
    # cv2.putText(img, text, (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

    return img_OpenCV
class ResizeNormalize(object):
    
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        weights_=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        opt=False,
        ):
    source = str(source)
    font = ImageFont.truetype('simsun.ttc', 30) 
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    model_retrograde = DetectMultiBackend(weights_, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    stride_, names_, pt_, jit_, onnx_, engine_ = model_retrograde.stride, model_retrograde.names, model_retrograde.pt, model_retrograde.jit, model_retrograde.onnx, model_retrograde.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Load plate model
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    plate_model = Model(opt)
    print('plate_model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    plate_model = torch.nn.DataParallel(plate_model).to(device)
    print('loading pretrained plate_model from %s' % opt.saved_model)
    plate_model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
        model_retrograde.model.half() if half else model_retrograde.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    model_retrograde.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    model.eval()
    model_retrograde.eval()
    for path, im, im0s, vid_cap, s in tqdm(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        pred_retrograde = model_retrograde(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred_retrograde = non_max_suppression(pred_retrograde, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # Process predictions
        pred_retrograde_list = [pr_ for pr_ in pred_retrograde[0]]
        for i, det in enumerate(pred):  # per image
            det_retrograde = pred_retrograde[i]
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_))
            

            if len(det_retrograde):
                # Rescale boxes from img_size to im0 size
                det_retrograde[:, :4] = scale_coords(im.shape[2:], det_retrograde[:, :4], im0.shape).round()

                # Print results
                for c in det_retrograde[:, -1].unique():
                    n = (det_retrograde[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names_[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det_retrograde):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names_[c] if hide_conf else f'{names_[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names_[c] / f'{p.stem}.jpg', BGR=True)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    for x1,y1,x2,y2,conf,cls in pred_retrograde_list:
                        if xyxy[0] >= x1-5 and xyxy[1] >= y1-5 and xyxy[2] <= x2+5 and xyxy[3] <= y2+5:
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box(xyxy, color=colors(c, True))

                                crop = one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                                crop = Image.fromarray(crop).convert('L')
                                transform = ResizeNormalize((opt.imgW, opt.imgH))
                                image_tensors = transform(crop)
                                image_tensors = torch.cat([image_tensors.unsqueeze(0)], 0)
                                batch_size = image_tensors.size(0)
                                image = image_tensors.to(device)
                                text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
                                length_for_pred = torch.IntTensor([opt.batch_max_length] * 1).to(device)
                                #
                                if 'CTC' in opt.Prediction:
                                    preds = plate_model(image, text_for_pred)

                                    # Select max probabilty (greedy decoding) then decode index to character
                                    preds_size = torch.IntTensor([preds.size(1)] * 1)
                                    _, preds_index = preds.max(2)
                                    # preds_index = preds_index.view(-1)
                                    preds_str = converter.decode(preds_index, preds_size)

                                else:
                                    preds = plate_model(image, text_for_pred, is_train=False)

                                    # select max probabilty (greedy decoding) then decode index to character
                                    _, preds_index = preds.max(2)
                                    preds_str = converter.decode(preds_index, length_for_pred)
                                    preds_prob = F.softmax(preds, dim=2)
                                    preds_max_prob, _ = preds_prob.max(dim=2)
                                    plate = preds_str[0].split('[s]')[0]
                                    for pla_ in plate:
                                        if pla_.isalpha():
                                            plate = '鲁'+plate
                                            break
                                    for pred, pred_max_prob in zip( preds_str, preds_max_prob):
                                        if 'Attn' in opt.Prediction:
                                            pred_EOS = pred.find('[s]')
                                            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                                            pred_max_prob = pred_max_prob[:pred_EOS]

                                        # calculate confidence score (= multiply of pred_max_prob)
                                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                                    im0 = annotator.result()
                                    text_ = plate + ' |' +str(round(float(confidence_score),2))
                                    
                                    im0 = draw_text(im0,text_,font,(int(xyxy[0]),int(xyxy[1])-10 if int(xyxy[1])-10 >=0 else 0),1,2,(255,255,255),colors(c, True))
                                    annotator.result_(im0)
                            break
                                    # cv2.putText(im0,plate,(int(xyxy[0]),int(xyxy[3])+30),cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
  


            # Print time (inference-only)
            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train_plate/exp/weights/best.pt', help='model path(s)')
    parser.add_argument('--weights_', nargs='+', type=str, default=ROOT / 'runs/train_retrograde/exp/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=  '../video/0.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',action='store_true', help='show results')
    parser.add_argument('--save-txt',action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop',default=True,action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave',action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='aaa', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def parse_opt_plate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model',default='../deep-text-recognition-benchmark-master/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation',default='TPS',type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction',default='ResNet',type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling',default='BiLSTM', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', default='Attn',type=str, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    opt_plate = parser.parse_args()
    return opt_plate

def main(opt,opt_plate):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt),opt = opt_plate)


if __name__ == "__main__":
    opt = parse_opt()
    opt_plate = parse_opt_plate()
    
    main(opt,opt_plate)
