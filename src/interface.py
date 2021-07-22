import cv2
import numpy as np

class Interface: 

    def __init__(self):
        self._main_window = cv2.resize(cv2.imread("images/Background.png"), (1350, 800))
        cv2.rectangle(self._main_window, (50, 700), (1300, 750), (0, 0, 255), 5)
        self.score = 0
    
    def draw_window(self, meme, face, score):
        meme = cv2.resize(meme, (600, 600))
        face = cv2.resize(face, (600, 600))
        
        self._main_window[50 : 650, 50 : 650] = meme
        self._main_window[50 : 650, 700 : 1300] = face
        if self.score <= 2500:
            self._main_window[700:750, 50: 50 + self.score // 2, 0] = 0
            self._main_window[700:750, 50: 50 + self.score // 2, 1] = 255
            self._main_window[700:750, 50: 50 + self.score // 2, 2] = 0
        else:
            self._main_window[700:750, 50: 1300, 0] = 0
            self._main_window[700:750, 50: 1300, 1] = 255
            self._main_window[700:750, 50: 1300, 2] = 0

        return self._main_window
    
    def update_score(self, is_happy, is_laugh):
        if is_happy:
            self.score += 1
        if is_laugh:
            self.score += 25
    
    def show_results(self):
        image = []
        if self.score <= 0:
            image = cv2.imread("images/bad_finish.jpg")
        elif self.score < 1250:
            image = cv2.imread("images/bellow_average_finish.jpg")
        elif self.score >= 1250:
            image = cv2.imread("images/under_average_finish.jpg")
        elif self.score >= 2500:
            image = cv2.imread("images/good_finish.jpg")
        image = cv2.resize(image, (650, 650))
        self._main_window[0 : 650, 350 : 1000] = image
        return self._main_window