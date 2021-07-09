import sys
import cv2
import argparse
import logging as log

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

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    sys.exit(main() or 0)