import sys
import cv2
import numpy as np
import argparse
import logging as log

import sounddevice as sd
import queue

from openvino.inference_engine import IECore

from time import perf_counter

audio = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio.put(indata.copy())

def prepare_audio_input(samplerate, channels):
    sd.default.samplerate = samplerate
    sd.default.channels = channels
    sd.default.blocksize = 2

def chunks(batch_size, channels, length):
    chunk = np.zeros((batch_size, channels, length),dtype=np.float32)
    n = 0
    while n < batch_size:
        data = audio.get()
        for i in range(7999):
            example = audio.get()
            data = np.concatenate((data, example))
        #data = (data - np.mean(data)) / (np.std(data) + 1e-15)
        data = data.T
        chunk[n, :, :] = data[:, :]
        n += 1
    yield chunk

def detect_laugh(exec_net, input_blob, output_blob, batch_size, channels, length, input_shape):
    result = -1
    for idx, chunk in enumerate(chunks(batch_size, channels, length)):
        chunk.shape = input_shape
        output = exec_net.infer(inputs={input_blob: chunk})
        output = output[output_blob]
        for batch, data in enumerate(output):
            label = np.argmax(data)
            if data[label] > 0.8:
                result = label
    if result == 26:
        return True
    else:
        return False

INPUT_HEIGHT, INPUT_WIDTH = 384, 672

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
    

    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    input = image.copy()    
    res = image.copy()

    input = input.transpose(2, 0, 1)

    output = exec_net.infer(inputs={input_blob : input})

    output = output[out_blob]
    output = np.squeeze(output)

    threshold = 0.5
    flag = 0

    for detection in output:
        confidence = detection[2]        
        if  confidence > threshold:
            xmin, ymin, xmax, ymax = int(detection[3]*INPUT_WIDTH), int(detection[4]*INPUT_HEIGHT), int(detection[5]*INPUT_WIDTH), int(detection[6]*INPUT_HEIGHT)         
            res = res[ymin:ymax+1, xmin:xmax+1]

        if flag > 5:
            break
    
    return res, image

def display_score():
    pass

def count_fps():
    pass


    emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

class Score:
    def __init__(self):
        self.scorings = {'neutral' : 0, 'happy' : 0, 'sad' : 0, 'surprise' : 0, 'anger' : 0}

    def increase_score(self, key):
        self.scorings[key] += 1

    def display_score_badge(self, frame):
        BADGE_LENGTH, BADGE_HEIGHT = 300, 200
        line_width = 1
        point1, point2 = (0, INPUT_HEIGHT-1), (BADGE_LENGTH, INPUT_HEIGHT-BADGE_HEIGHT)
        cv2.rectangle(frame, point1, point2, color2, line_width)

        for key in self.scorings.keys():
            pass





def recognize_smile(image, exec_net, input_blob, out_blob, frame, score=0):
    input = cv2.resize(image, (64, 64))
    input = input.transpose(2, 0, 1)
    result = frame
    
    output = exec_net.infer(inputs={input_blob : input})

    emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    #point1, point2 = (int(detection.xmin), int(detection.ymax)), (int(detection.xmax), int(detection.ymin))
    point1, point2 = (0, INPUT_HEIGHT-1), (400, INPUT_HEIGHT-50)
    color = (0, 0, 255)
    line_width = 1
    thickness = -1
    color2 = (0,255, 0)
    point3, point4 = (1, INPUT_HEIGHT-2), (score, INPUT_HEIGHT-49)
    cv2.rectangle(frame, point1, point2, color, line_width)
    cv2.rectangle(frame, point3, point4, color2, thickness)


    if np.argmax(output[out_blob]) == 1:
        cv2.putText(result, "Smile!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        if score < 399:
            score += 1
        thickness = -1
        color = (0,255, 0)
        point1, point2 = (1, INPUT_HEIGHT-2), (score, INPUT_HEIGHT-49)
        cv2.rectangle(frame, point1, point2, color, thickness)
    
    return image, score


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
    net = ie.read_network(model="face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml")
    exec_net = ie.load_network(network=net, device_name="CPU")
    out_blob = next(iter(net.outputs))
    input_blob = next(iter(net.inputs))

    net2 = ie.read_network(model="emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml")
    exec_net2 = ie.load_network(network=net2, device_name="CPU")
    out_blob2 = next(iter(net2.outputs))
    input_blob2 = next(iter(net2.inputs))

#sound
    name_sound_model = "aclnet/FP16/aclnet.xml"
    net3 = ie.read_network(name_sound_model, name_sound_model[:-4] + ".bin")
    exec_net3 = ie.load_network(network=net3, device_name="GPU")
    input_blob3 = next(iter(net3.input_info))
    input_shape3 = net3.input_info[input_blob3].input_data.shape
    output_blob3 = next(iter(net3.outputs))

    """ labels = []
    with open("aclnet_53cl.txt", "r") as file:
        labels = [line.rstrip() for line in file.readlines()] """

    batch_size, channels, one, length = input_shape3
    samlerate = 16000
    prepare_audio_input(samlerate, channels)

    score = 0
    while True:
        stream = sd.InputStream(callback=audio_callback)
        with stream:

            start_time = perf_counter()

            ret, frame = cap.read()

            try:
                face, frame = detect_face(frame, exec_net, input_blob, out_blob)
                face , score = recognize_smile(face, exec_net2, input_blob2, out_blob2, frame, score)
                #print(audio.qsize())
                if audio.qsize() >= 8000:
                    res = detect_laugh(exec_net3, input_blob3, output_blob3, batch_size, channels, length, input_shape3)
                    if res:
                        cv2.putText(frame, "Laugh!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255)) 
            except:
                pass 
            end_time = perf_counter()
            log.info("score is  {} sec".format(score))


            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    sys.exit(main() or 0)