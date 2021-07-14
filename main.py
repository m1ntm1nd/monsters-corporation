import sys
import cv2
import numpy as np
import argparse
import logging as log
from openvino.inference_engine import IECore

from time import perf_counter

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml file with a trained model.    ', required=False, type=str, dest='model_xml')
    parser.add_argument('-w', '--weights', help='Path to an .bin file with a trained weights.', required=False, type=str, dest='model_bin')
    parser.add_argument('-i', '--input', help='Data for input                                ', required=False, type=str, nargs='+', dest='input')
    parser.add_argument('-l', '--extension', help='Path to MKLDNN (CPU, MYRIAD) custom layers', type=str, default=None, dest='extension')    
    parser.add_argument('--default_device', help='Default device for heterogeneous inference', 
                        choices=['CPU', 'GPU', 'MYRIAD', 'FGPA'], default=None, type=str, dest='default_device')
    parser.add_argument('--labels', help='Labels mapping file', default=None, type=str, dest='labels')  
    return parser

def prepare_input():
    pass


def detect_face(image, exec_net, input_blob, out_blob):
    INPUT_HEIGHT, INPUT_WIDTH = 384, 672

    input = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    res = input
    input = input.transpose(2, 0, 1)

    output = exec_net.infer(inputs={input_blob : input})

    
    output = output[out_blob]
    output = np.squeeze(output)

    threshold = 0.5
    color = (0, 255, 0)
    line_width = 2

    flag = 0
    for detection in output:
        #log.info(detection)
        if detection[2] > threshold:
            point1, point2 = (int(detection[3]*INPUT_WIDTH), int(detection[4]*INPUT_HEIGHT)), (int(detection[5]*INPUT_WIDTH), int(detection[6]*INPUT_HEIGHT))
            xmin, ymin, xmax, ymax = int(detection[3]*INPUT_WIDTH), int(detection[4]*INPUT_HEIGHT), int(detection[5]*INPUT_WIDTH), int(detection[6]*INPUT_HEIGHT)         
            #cv2.rectangle(image, point1, point2, color, line_width)
            res = res[ymin:ymax, xmin:xmax]
            #cv2.imwrite('images/{}.jpg'.format(flag), res) 
        if flag > 5:
            break
    
    res = cv2.resize(res, (256, 256))

    return res, image


def recognize_smile(image, exec_net, input_blob, out_blob, frame):
    input = cv2.resize(image, (64, 64))
    input = input.transpose(2, 0, 1)
    result = frame

    output = exec_net.infer(inputs={input_blob : input})
    if np.argmax(output[out_blob]) == 1:
        cv2.putText(result, "Smile!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    return image


# def draw_detections(frame, detections, labels, threshold):

#     size = frame.shape[:2]
#     for detection in detections:
#         score = detection.score
        
#         # If score more than threshold, draw rectangle on the frame
#         if score >= threshold:
#             point1, point2 = (int(detection.xmin), int(detection.ymax)), (int(detection.xmax), int(detection.ymin))
#             color = (0, 255, 0)
#             line_width = 2
#             cv2.rectangle(frame, point1, point2, color, line_width)
            
#             id = detection.id
#             text_size = 1
#             text = labels[detection.id + 1]
#             cv2.putText(frame, text, (int(detection.xmin), int(detection.ymin)), cv2.FONT_HERSHEY_COMPLEX, text_size, color)
        
#     return frame

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    cap = cv2.VideoCapture(0)
    ie = IECore()
    net = ie.read_network(model="face-detection-adas-0001/FP16/face-detection-adas-0001.xml")
    exec_net = ie.load_network(network=net, device_name="CPU")
    out_blob = next(iter(net.outputs))
    input_blob = next(iter(net.inputs))

    net2 = ie.read_network(model="emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml")
    exec_net2 = ie.load_network(network=net2, device_name="CPU")
    out_blob2 = next(iter(net2.outputs))
    input_blob2 = next(iter(net2.inputs))

    while True:
        start_time = perf_counter()

        ret, frame = cap.read()
        face, frame = detect_face(frame, exec_net, input_blob, out_blob)
        face = recognize_smile(face, exec_net2, input_blob2, out_blob2, frame)

        end_time = perf_counter()
        log.info("1 frame consuming {} sec".format(end_time - start_time))


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    sys.exit(main() or 0)