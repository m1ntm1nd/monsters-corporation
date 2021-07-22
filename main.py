import sys
import cv2
import numpy as np
import argparse
import logging as log
from pathlib import Path
import sounddevice as sd
import queue
from openvino.inference_engine import IECore
from time import perf_counter

from src.interface import Interface

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
    flag = False

    if np.argmax(output[out_blob]) == 1:
        flag = True

    return flag

def build_argparser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', type=str, required=True, help="Required. Path to folder with images")
    return parser.parse_args()


def main():
    args = build_argparser()
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
    
    batch_size, channels, one, length = input_shape3
    samlerate = 16000
    prepare_audio_input(samlerate, channels)

    memes_dir = Path(args.i).iterdir()

    with sd.InputStream(callback=audio_callback):
        interface = Interface()
        is_quit = False
        FPS_VALUE = 0
        for meme_file in memes_dir:
            meme = cv2.imread(meme_file.as_posix())

            start_meme_time = perf_counter()
            counter = 0
            start_time = 0
            end_time = 0
            while True:
                is_laughing = False
                is_happy = False
                ret, frame = cap.read()
                
                if counter == 0:
                    start_time = perf_counter()
                try:
                    face, frame = detect_face(frame, exec_net, input_blob, out_blob)
                    is_happy = recognize_smile(face, exec_net2, input_blob2, out_blob2)
                    
                    if audio.qsize() >= 1:
                        
                        is_laughing = detect_laugh(exec_net3, input_blob3, output_blob3, batch_size, channels, length, input_shape3)
                        if is_laughing:
                            log.info("Laugh!")
                    interface.update_score(is_happy, is_laughing)
                except:
                    pass
                
                counter += 1
                if counter == 10:
                    end_time = perf_counter()
                    FPS_VALUE = int((counter + 1)/(end_time-start_time))
                    counter = 0
                cv2.putText(frame, str(FPS_VALUE) + ' FPS', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                cv2.imshow('OH THAT IS GAME', interface.draw_window(meme, frame, 0))
                meme_time = end_time - start_meme_time
                key = cv2.waitKey(1)
                if key == ord('q'):
                    is_quit = True
                    break
                elif key == ord('n'):
                    break
                if interface.score >= 2500:
                    cv2.imshow('OH THAT IS GAME', interface.show_results())
                    is_correct = False
                    while not is_correct:
                        key = cv2.waitKey()
                        if key == ord('y'):
                            is_correct = True
                            interface = Interface()
                            cap.read()
                        if key == ord('q'):
                            is_correct = True
                            is_quit = True
                if is_quit:
                    break
            if is_quit:
                break
        if not is_quit:
            cv2.imshow('OH THAT IS GAME', interface.show_results())
            cv2.waitKey()

if __name__ == "__main__":
    sys.exit(main() or 0)