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
    sd.default.blocksize = 16000

def chunks(batch_size, channels, length):
    chunk = np.zeros((batch_size, channels, length),dtype=np.float32)
    n = 0
    while n < batch_size:
        data = audio.get()
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
            #log.warn(label)
            if data[label] > 0.8:
                result = label
    if result == 26:
        return True
    else:
        return False

INPUT_HEIGHT, INPUT_WIDTH = 384, 672


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


def recognize_smile(image, exec_net, input_blob, out_blob):
    input = cv2.resize(image, (64, 64))
    input = input.transpose(2, 0, 1)
    
    
    output = exec_net.infer(inputs={input_blob : input})

    #emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    flag = 0

    if np.argmax(output[out_blob]) == 1:
        flag = 1

    return flag


class Score:
    def __init__(self):
        self.happy = 0
        self.neutral = 0
    
    def _display_score(self, frame):
        BAR_WIDTH = 30
        PADDING = 1
        line_width = -1
        color = (0, 0, 255)
        color2 = (0, 255, 0)
        point1, point2 = (INPUT_WIDTH-BAR_WIDTH, (INPUT_HEIGHT-self.neutral)), (INPUT_WIDTH-PADDING, INPUT_HEIGHT)
        point3, point4 = (INPUT_WIDTH-2*BAR_WIDTH, (INPUT_HEIGHT-self.happy)), (INPUT_WIDTH-PADDING-BAR_WIDTH, INPUT_HEIGHT)
        
        cv2.rectangle(frame, point1, point2, color, line_width)
        cv2.rectangle(frame, point3, point4, color2, line_width)

        return frame

    def update_score(self, flag, frame):
        if flag:
            self.happy += 2
            self.neutral -= 1
            log.warn("Happiness score is {}".format(self.happy))
            log.warn("Neutral score is {}".format(self.neutral))
        else:

            self.neutral += 1
            self.happy -= 1
            log.warn("Happiness score is {}".format(self.happy))
            log.warn("Neutral score is {}".format(self.neutral))

        self._check_score()
        return self._display_score(frame)

    def _check_score(self):
        if self.happy < 0:
            self.happy = 0    
        if self.neutral < 0:
            self.neutral = 0
        if self.happy > INPUT_HEIGHT:
            self.happy = INPUT_HEIGHT
        if self.neutral > INPUT_HEIGHT:
            self.neutral = INPUT_HEIGHT

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    
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

    score = Score()

    with sd.InputStream(callback=audio_callback):
        while True:

            ret, frame = cap.read()
            start_time = perf_counter()
            try:
                face, frame = detect_face(frame, exec_net, input_blob, out_blob)
                flag = recognize_smile(face, exec_net2, input_blob2, out_blob2)
                
                frame = score.update_score(flag, frame)
                if audio.qsize() >= 1:
                #log.info(audio.qsize())
                    
                    res = detect_laugh(exec_net3, input_blob3, output_blob3, batch_size, channels, length, input_shape3)
                    #log.info(res)
                    if res:
                        log.info("Laugh!")
                        #zcv2.putText(frame, , (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            except:
                pass

            end_time = perf_counter()
            FPS_VALUE = int(1/(end_time-start_time))
            cv2.putText(frame, str(FPS_VALUE) + ' FPS', (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            #log.info("score is  {} sec".format(score))

            cv2.imshow('OH THAT IS GAME', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == "__main__":
    sys.exit(main() or 0)