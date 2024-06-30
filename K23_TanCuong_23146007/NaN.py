import socket
import cv2
import numpy as np
import torch
import time
import json
import base64
global sendBack_angle, sendBack_Speed, current_speed, current_angle, radius
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 54321
s.connect(('127.0.0.1', PORT))
def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed
error_arr = np.zeros(5)
pre_t = time.time()
MAX_SPEED = 60
def PID(error, p, i, d):
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)
def detect_yellow_line(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    yellow_detected = cv2.bitwise_and(image, image, mask=yellow_mask)
    gray_yellow = cv2.cvtColor(yellow_detected, cv2.COLOR_BGR2GRAY)
    _, binary_yellow = cv2.threshold(gray_yellow, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        yellow_line_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(yellow_line_contour)
        center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        return center_x
    else:
        return image.shape[1] // 2
if __name__ == "__main__":
    try:
        while True:
            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """
            message = bytes(f"{sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)
            data_recv = json.loads(data)
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            center_of_road = detect_yellow_line(image)
            deviation = center_of_road - (image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.1, i=0.01, d=0.05)
            print(current_speed, current_angle)
            print(image.shape)            
            cv2.imshow('Image Original', image)
            Control(angle_setpoint, sendBack_Speed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print('closing socket')
        s.close()