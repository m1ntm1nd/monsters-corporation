import sys
import cv2
import numpy as np
import argparse
import logging as log
from pathlib import Path
import sounddevice as sd
from openvino.inference_engine import IECore
from time import perf_counter

from src.interface import Interface
from src.laugh_detector import LaughDetector
from src.happy_detector import HappyDetector



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

    memes_dir = Path(args.i).iterdir()

    happy_detector = HappyDetector(ie)
    laugh_detector = LaughDetector(ie)

    with laugh_detector.start_record():
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
                    face, frame = happy_detector.detect_face(frame)
                    is_happy = happy_detector.recognize_smile(face)
                    
                    if not laugh_detector.is_empty():
                        is_laughing = laugh_detector.detect_laugh()
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