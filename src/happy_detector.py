import cv2
import numpy as np

class HappyDetector:
    def __init__(self, ie):
        # face detection net load
        self._fd_net = ie.read_network(model="face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml")
        self._fd_exec_net = ie.load_network(network=self._fd_net, device_name="CPU")
        self._fd_out_blob = next(iter(self._fd_net.outputs))
        self._fd_input_blob = next(iter(self._fd_net.inputs))

        # read input shape for face detection model
        _, _, self._fd_h, self._fd_w = self._fd_net.input_info[self._fd_input_blob].input_data.shape

        # emotions recognition net load
        self._er_net = ie.read_network(model="emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml")
        self._er_exec_net = ie.load_network(network=self._er_net, device_name="CPU")
        self._er_out_blob = next(iter(self._er_net.outputs))
        self._er_input_blob = next(iter(self._er_net.inputs))

        _, _, self._er_h, self._er_w = self._er_net.input_info[self._er_input_blob].input_data.shape    

        

    def detect_face(self, image):
        image = cv2.resize(image, (self._fd_w, self._fd_h))
        input = image.copy()
        res = image.copy()

        input = input.transpose(2, 0, 1)

        output = self._fd_exec_net.infer(inputs={self._fd_input_blob : input})

        output = output[self._fd_out_blob]
        output = np.squeeze(output)

        threshold = 0.5
        flag = 0

        for detection in output:
            confidence = detection[2]        
            if  confidence > threshold:
                xmin, ymin, xmax, ymax = (int(detection[3]*self._fd_w), int(detection[4]*self._fd_h), 
                    int(detection[5]*self._fd_w), int(detection[6]*self._fd_h))
                res = res[ymin:ymax+1, xmin:xmax+1]

            if flag > 5:
                break
        
        return res, image

    def recognize_smile(self, image):
        input = cv2.resize(image, (64, 64))
        input = input.transpose(2, 0, 1)
        
        output = self._er_exec_net.infer(inputs={self._er_input_blob : input})

        #emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
        flag = False

        if np.argmax(output[out_blob]) == 1:
            flag = True

        return flag
    