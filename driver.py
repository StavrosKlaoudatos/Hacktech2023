from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, Lock
from queue import Queue


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def convert2original_scuffed(bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w = (730, 1430)

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted



def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()

PINS = [38, 37, 36, 35, 33, 32, 31, 29, 26]
class LedState:
    def __init__(self):
        self.activations_l = [0 for _ in range(len(PINS) + 2)]
        self.activations_r = [0 for _ in range(len(PINS) + 2)]
        self.lock = Lock()


def inference(darknet_image_queue, detections_queue, fps_queue, ledstate, mock):
    while cap.isOpened():
        if not mock:
            darknet_image = darknet_image_queue.get()
            prev_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
            #detections_queue.put(detections)
            fps = int(1/(time.time() - prev_time))
            #fps_queue.put(fps)
            print("FPS: {}".format(fps))
        else:
            detections = darknet_image_queue.get()
        darknet.print_detections(detections, args.ext_output)
        for detection in detections:
            bbox_adj = convert2original_scuffed(detection[2])
            side = 'left' if (bbox_adj[0] - 160) - bbox_adj[1] * (2500 - 160)/1000 < 0 else 'right'
            print(side)
            ledstate.lock.acquire()
            if side == 'left':
                ledstate.activations_l[1 + 4] = 1
            elif side == 'right':
                ledstate.activations_r[-2 - 4] = 1
            ledstate.lock.release()
        if not mock:
            darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            if args.out_filename is not None:
                video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    video.release()
    cv2.destroyAllWindows()

def led_control(ledstate):
    import Jetson.GPIO as gpio
    import time
    gpio.setmode(gpio.BOARD)
    for pin in PINS:
        gpio.setup(pin, gpio.OUT)
    
    cycle_pos = 0
    while True:
        ledstate.lock.acquire()
        if cycle_pos == 10 * 3:
            ledstate.activations_l = [0] + ledstate.activations_l[:-1]
            ledstate.activations_r = ledstate.activations_r[1:] + [0]
            cycle_pos = 0
        activations_l = ledstate.activations_l
        activations_r = ledstate.activations_r
        for i, p in enumerate(PINS):
            if i >= len(PINS):
                continue
            
            a = 0
            if i+1 >= 5:
                a |= activations_l[i + 1]
                a |= activations_l[i + 2]
            if i+1 <= 5:
                a |= activations_r[i]
                a |= activations_r[i + 1]

            gpio.output(p, gpio.HIGH if a else gpio.LOW)
        ledstate.lock.release()

        time.sleep(0.1)
        cycle_pos += 1

def mock_inference(darknet_image_queue):
    logf = open('detections.log')
    start_time = time.time()
    frame = 0
    for line in logf:
        detection = eval(line)
        darknet_image_queue.put(detection)
        wait_time = frame * 1/30 + start_time - time.time()
        print(frame * 1/30)
        if wait_time > 0:
            time.sleep(wait_time)
        frame += 1


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ledstate = LedState()
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    #Thread(target=mock_inference, args=(darknet_image_queue,)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue, ledstate, True)).start()
    #Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
    Thread(target=led_control, args=(ledstate,)).start()
