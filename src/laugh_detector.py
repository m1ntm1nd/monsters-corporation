import queue
import sys
import sounddevice as sd
import numpy as np
from openvino.inference_engine import IECore

class LaughDetector:
    def __init__(self, ie):
        self._audio = queue.Queue()
        self._name_sound_model = "models/aclnet/FP16/aclnet.xml"
        self._net = ie.read_network(self._name_sound_model, self._name_sound_model[:-4] + ".bin")
        self._exec_net = ie.load_network(network=self._net, device_name="GPU")
        self._input_blob = next(iter(self._net.input_info))
        self._output_blob = next(iter(self._net.outputs))
        self._batch_size, self._channels, _, self._length = self._net.input_info[self._input_blob].input_data.shape
    
    def _chunks(self):
        chunk = np.zeros((self._batch_size, self._channels, self._length),dtype=np.float32)
        n = 0
        while n < self._batch_size:
            data = self._audio.get()
            data = data.T
            chunk[n, :, :] = data[:, :]
            n += 1
        yield chunk

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self._audio.put(indata.copy())

    def detect_laugh(self):
        result = -1
        for chunk in self._chunks():
            output = self._exec_net.infer(inputs={self._input_blob: chunk})
            output = output[self._output_blob]
            for data in output:
                label = np.argmax(data)
                if data[label] > 0.8:
                    result = label
        if result == 26:
            return True
        else:
            return False
    
    def start_record(self):
        sd.default.samplerate = self._length
        sd.default.channels = self._channels
        sd.default.blocksize = self._length

        return sd.InputStream(callback=self._audio_callback)
    
    def is_empty(self):
        return self._audio.empty()