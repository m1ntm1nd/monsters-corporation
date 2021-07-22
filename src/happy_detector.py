import cv2
import numpy as np

class HappyDetector:
    def __init__(self, ie):
        # face detection net load
        self._fd_net = ie.read_network(model="models/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml")
        self._fd_exec_net = ie.load_network(network=self._fd_net, device_name="CPU")
        self._fd_out_blob = next(iter(self._fd_net.outputs))
        self._fd_input_blob = next(iter(self._fd_net.inputs))

        # read input shape for face detection model
        _, _, self._fd_h, self._fd_w = self._fd_net.input_info[self._fd_input_blob].input_data.shape

        # emotions recognition net load
        self._er_net = ie.read_network(model="models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml")
        self._er_exec_net = ie.load_network(network=self._er_net, device_name="CPU")
        self._er_out_blob = next(iter(self._er_net.outputs))
        self._er_input_blob = next(iter(self._er_net.inputs))

        _, _, self._er_h, self._er_w = self._er_net.input_info[self._er_input_blob].input_data.shape    

        

    def _detect_face(self, image):
        image = cv2.resize(image, (self._fd_w, self._fd_h))
        face = []

        input = image.transpose(2, 0, 1)

        output = self._fd_exec_net.infer(inputs={self._fd_input_blob : input})

        output = output[self._fd_out_blob]
        output = np.squeeze(output)

        threshold = 0.5

        for detection in output:
            confidence = detection[2]        
            if  confidence > threshold:
                xmin, ymin, xmax, ymax = (int(detection[3]*self._fd_w), int(detection[4]*self._fd_h), 
                    int(detection[5]*self._fd_w), int(detection[6]*self._fd_h))
                face = image[ymin:ymax+1, xmin:xmax+1]
        
        return face

    def recognize_smile(self, image):
        face = self._detect_face(image)
        
        if len(face) == 0:
            return False
        
        face = cv2.resize(face, (self._er_w, self._er_h))
        face = face.transpose(2, 0, 1)
        
        output = self._er_exec_net.infer(inputs={self._er_input_blob : face})

        # emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
        if np.argmax(output[self._er_out_blob]) == 1:
            return True
        return False
    