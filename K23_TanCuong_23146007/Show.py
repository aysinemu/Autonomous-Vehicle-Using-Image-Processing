import socket
import cv2
import numpy as np
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
    global pre_t, error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error * p
    delta_t = time.time() - pre_t
    pre_t = time.time()
    if delta_t != 0:
        D = (error - error_arr[1]) / delta_t * d
    else:
        D = 0
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    if abs(angle) > 25:
        angle = np.sign(angle) * 25
    return int(angle)

def remove_shadow(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv_image0', hsv_image)
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([180, 255, 40])
    shadow_mask = cv2.inRange(hsv_image, lower_shadow, upper_shadow)
    cv2.imshow('shadow_mask', shadow_mask)
    image_no_shadow = cv2.addWeighted(image, 1, cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR), 1, 0)
    cv2.imshow('image_no_shadow', image_no_shadow)
    return image_no_shadow

def detect_yellow_line(image):
    image_without_shadow = remove_shadow(image)
    hsv_image = cv2.cvtColor(image_without_shadow, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv_image', hsv_image)
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    cv2.imshow('yellow_mask', yellow_mask)
    yellow_detected = cv2.bitwise_and(image_without_shadow, image_without_shadow, mask=yellow_mask)
    cv2.imshow('yellow_detected', yellow_detected)
    gray_yellow = cv2.cvtColor(yellow_detected, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_yellow', gray_yellow)
    _, binary_yellow = cv2.threshold(gray_yellow, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary_yellow', binary_yellow)
    contours, _ = cv2.findContours(binary_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        yellow_line_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(yellow_line_contour)
        center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        return center_x
    else:
        return image.shape[1] // 2
    
def process_image(image):
    image_without_shadow = remove_shadow(image)
    center_of_road = detect_yellow_line(image_without_shadow)
    deviation = center_of_road - (image.shape[1] // 2)
    angle_setpoint = PID(deviation, p=0.1, i=0.01, d=0.05)
    if abs(angle_setpoint) < 10:
        adjusted_speed = 70
    else:
        adjusted_speed = 50
    return adjusted_speed

if __name__ == "__main__":
    try:
        while True:
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
            speed = process_image(image)
            deviation = center_of_road - (image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.1, i=0.01, d=0.05)
            print(current_speed, current_angle)
            print(image.shape)            
            cv2.imshow('Image Original', image)
            Control(angle_setpoint, speed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print('closing socket')
        s.close()
