# Code Made By Nguyễn Châu Tấn Cường MSSV-23146007 Cơ Điện Tử 
# Gọi thư viện để chạy
import socket
import cv2
import numpy as np
import time
import json
import base64
# Khai báo biến cục bộ để có thể sử dụng ở bất cứ nơi nào
global sendBack_angle, sendBack_Speed, current_speed, current_angle, radius
# Cho các biến đã khai báo giá trị bằng 0 và có thể thay đổi được
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 54321
s.connect(('127.0.0.1', PORT))
# Hàm điều khiển
def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed
# Sử dụng mảng array tạo ra 1 mảng có 5 phần tử để chứa các thông tin về mấy cái trong global (sự sai số)
error_arr = np.zeros(5)
# Gía trị thời gian hiện tại được gắn vào để theo dõi thời gian
pre_t = time.time()
# Speed tối đa được sử dụng trong hệ thống là 60
MAX_SPEED = 60
# Hàm tính toán góc
# Bộ điều khiểu PID sẽ tính toán giá trị "sai số" là hiệu số giữa giá trị đo thông số biến đổi và giá trị 
# đặt mong muốn ,bộ điều khiển sẽ thực hiện giảm tối đa sai số bằng cách điều chỉnh giá trị điều khiển 
# đầu vào
# Link : https://via.makerviet.org/en/docs/1_about-via/3_phan-cung-via/
# Link : https://www.youtube.com/@MATLAB
# Link : https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjJvKjeleaCAxXPQt4KHQtQCTQQFnoECAoQAQ&url=https%3A%2F%2Fvi.wikipedia.org%2Fwiki%2FB%25E1%25BB%2599_%25C4%2591i%25E1%25BB%2581u_khi%25E1%25BB%2583n_PID&usg=AOvVaw1IA0-qqR3eCdMP4c_QVZln&opi=89978449
def PID(error, p, i, d):
    # Cho nó là biến cục bộ để có thể gọi nó ra ở bất cứ đâu trong chương trình
    global pre_t, error_arr
    # 2 Dòng error này giúp shift hàng phần tử sang bên phải 
        # Khởi tạo mảng error_arr với 5 phần tử
        # error_arr = np.array([1, 2, 3, 4, 5])
        # Hiển thị trước khi di chuyển
        # print("Before shift:", error_arr)
        # # Di chuyển tất cả các phần tử về phải một vị trí
        # error_arr[1:] = error_arr[0:-1]
        # Hiển thị sau khi di chuyển
        # print("After shift:", error_arr)
        # Before shift: [1 2 3 4 5]
        # After shift: [1 1 2 3 4]
    error_arr[1:] = error_arr[0:-1]
    # Gán giá trị sai số mới vào đầu mảng
    error_arr[0] = error
    # Gía trị sai số mới nhân với hằng số sai số Kp để có được giá trị thành phần tỷ lệ thuận
    P = error * p
    # Tính thời gian chênh lệch như trong vật lý học time.time() gọi ra thời gian hiện tại
    # dưới dạng second còn pre_t nãy mình khai báo ở trên thì nó là thời gian lúc trước thì để
    # tính delta_t thì ta lấy thời gian sau trừ thời gian trước
    delta_t = time.time() - pre_t
    # tính xong delta_t tại thời điểm đó rùi thì tiếp tục gọi pre_t và đặt nó thành thời gian
    # thực lúc minh tính delta_t để tính mình lại lấy chính cái thời gian thực mình đặt này để tính 
    # delta_t tiếp theo
    pre_t = time.time()
    # Tránh việc chia cho 0 để bớt xuất hiện lỗi NaN
    if delta_t != 0:
        # Nếu delta_t khác 0 thì lấy giá trị sai số mới trừ cho giá trị sai số cữ tất cả chia cho
        # delta_t nhân với hằng số sai số Kd để tính đạo hàm
        D = (error - error_arr[1]) / delta_t * d
    else:
        # Nếu delta_t = 0 thì D = 0 mà delta_t = 0 khi và chỉ khi thời gian thực trừ thời gian trước đó
        # lớn hơn 0 1 tí vì mình làm tròn nên sấp bất kì số gì mà lớn hơn 0 1 tí là làm tròn thành 0 nên
        # delta_t mới có thể = 0
        # Lần lặp 1 - Thời gian delta (delta_t): 0.0
        # Lần lặp 2 - Thời gian delta (delta_t): 0.001  
        # Giả sử có một khoảng thời gian nhỏ giữa lần lặp trước và sau
        # Lần lặp 3 - Thời gian delta (delta_t): 0.002
        # Lần lặp 4 - Thời gian delta (delta_t): 0.0
        # Lần lặp 5 - Thời gian delta (delta_t): 0.003
        D = 0
    # Để tính tích phân ta dùng tổng các thành phần sai số trong mảng cũ nhấn với chênh lệch thời gian
    # và nhân với hằng số hệ số Ki
    I = np.sum(error_arr) * delta_t * i
    # Góc sẽ bằng P I D cộng lại với nhau , mình vẫn chưa hiểu rõ lắm nhưng góc ở đây đối với mình chính
    # là khoảng chênh lệch giữa tâm hình ảnh với tọa độ của đường màu vàng có nghĩa là đồ dài của tâm hình
    # ảnh - cho độ dài hiện tại của từ phía hình ảnh tới đường vàng nếu - nhau mà ra âm thì là vạch vàng 
    # đang ở bên trái còn + thì bên phải, nên khi có chiều dài giữa tâm và đường vàng r thì ta có thể điều 
    # chỉnh độ dài hiện tại về con số 0 đề tâm vs đường vàng trùng nhau (Mình vẫn đang tìm hiểu rõ xem mình
    # có hiểu đúng về nó không :3 )
    angle = P + I + D
    # Đưa về giá trị tuyết đối
    # Dòng lệnh if để giá trị tuyệt đối nên cho chúng ta biết khoang ngưỡng là từ 0 đến 25
    # nếu vượt qua 25 thì sẽ tự động đặt angle = 25
    if abs(angle) > 25:
        angle = np.sign(angle) * 25
    # Trả giá trị góc dưới dạng số integer
    return int(angle)
# Hàm loại bỏ bóng bằng cách chuyển các hệ số thấp phân bóng xuống dạnh hệ số thập phân của
# ánh sáng trắng
def remove_shadow(image):
    # Chuyển đồi thành màu HSV
    # HSV (Hue, Saturation, Value): Giá trị (Value) đo lường độ sáng của màu sắc và 
    # thường được sử dụng trong ngữ cảnh của không gian màu sắc.Thường được sử dụng 
    # trong ngữ cảnh của xử lý ảnh và đồ họa máy tính.Thường được sử dụng trong ngữ 
    # cảnh của xử lý ảnh và đồ họa máy tính.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Xác định khoảng giá trị của cái bóng (Đen + Trắng)
    # H: từ 0 đến 180, không giới hạn màu sắc cụ thể (0 và 180 đều tương đương màu 
    # đỏ trong không gian màu HSV).
    # S: từ 0 đến 255, không giới hạn độ bão hòa (màu sắc đậm hay nhạt).
    # V: từ 0 đến 40, giới hạn giá trị độ sáng, chỉ chấp nhận các màu tối.
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([180, 255, 40])
    # Tạo một lớp số nhị phân để chèn lên phần bóng , giữ lại khoảng vùng màu trong 2 mảng trên
    shadow_mask = cv2.inRange(hsv_image, lower_shadow, upper_shadow)
    # Thay đổi giá trị của shadow thành giá trị của màu trắng bằng cách dùng 2 ảnh chồng lên nhau
    # có thế hiểu nôn la là kết hợp fusion =)))
    # Link : https://pythonexamples.org/python-opencv-add-blend-two-images/
    image_no_shadow = cv2.addWeighted(image, 1, cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR), 1, 0)
    return image_no_shadow
# Hàm dò vạch kẻ đường
def detect_yellow_line(image):
    # Loại bỏ bóng
    image_without_shadow = remove_shadow(image)
    # Chuyển đổi lại thành màu HSV lần nữa
    hsv_image = cv2.cvtColor(image_without_shadow, cv2.COLOR_BGR2HSV)
    # Tìm khoảng vùng có màu vàng
    # H (Hue): 20 đến 40, đại diện cho các gam màu từ cam đến vàng.
    # S (Saturation): 50 đến 255, đại diện cho độ bão hòa, với giá 
    # trị thấp hơn có thể là màu trắng hoặc xám.
    # V (Value): 50 đến 255, đại diện cho độ sáng, với giá trị thấp 
    # hơn có thể là màu tối
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    # Xác định giá trị của khoảng vàng , giữ lại khoảng vùng màu trong 2 mảng trên
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    # Tách phần vàng ra khỏi image nhận được từ data
    # Link : https://pyimagesearch.com/2021/01/19/opencv-bitwise-and-or-xor-and-not/
    yellow_detected = cv2.bitwise_and(image_without_shadow, image_without_shadow, mask=yellow_mask)
    # Chuyển đổi thành màu xám
    gray_yellow = cv2.cvtColor(yellow_detected, cv2.COLOR_BGR2GRAY)
    # Sử dụng threshhold để biến bức ảnh đó thành bức ảnh nhị phân
    # Link : https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
    _, binary_yellow = cv2.threshold(gray_yellow, 128, 255, cv2.THRESH_BINARY)
    # Tìm giá trị của phần màu vàng trong bức ảnh nhị phân
    contours, _ = cv2.findContours(binary_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Tìm tâm của bức ảnh
    # Link học : https://viblo.asia/p/su-dung-opencv-de-tim-diem-chinh-giua-cua-1-hinh-4dbZN8Lm5YM
    # Dòng này kiểm tra xem danh sách contours (contours) có rỗng hay không. 
    # Nếu có ít nhất một contour, điều kiện sẽ là True.
    if contours:
        # Dòng này sử dụng hàm max để xác định contour lớn nhất trong danh sách contours 
        # dựa trên diện tích contour (cv2.contourArea). Contour này được gán cho biến 
        # yellow_line_contour.
        yellow_line_contour = max(contours, key=cv2.contourArea)
        # Dòng này sử dụng hàm cv2.moments để tính các moment của contour (M). Sau đó, 
        # trung tâm x của contour được tính dựa trên các moment này. Điều kiện if M["m00"] != 0 
        # được sử dụng để tránh chia cho 0 (để tránh lỗi) trong trường hợp mẫu moment (M["m00"]) bằng 0.
        M = cv2.moments(yellow_line_contour)
        center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        # Nếu có contours, hàm trả về trung tâm x của contour màu vàng.
        return center_x
    else:
        # Nếu không có contours, tức là không tìm thấy đường contour màu vàng, hàm sẽ trả về giá trị 
        # trung tâm mặc định, được tính bằng cách lấy chiều rộng của hình ảnh (image.shape[1]) và 
        # chia cho 2. Điều này giả sử trung tâm mặc định nằm ở giữa hình ảnh theo chiều ngang.
        return image.shape[1] // 2
# Hàm điều chỉnh tốc độ thông qua ảnh    
def process_image(image):
    # Gọi hàm xóa bóng xong gọi hàm tìm tâm đường xong gọi hàm pid để tìm góc
    image_without_shadow = remove_shadow(image)
    center_of_road = detect_yellow_line(image_without_shadow)
    deviation = center_of_road - (image.shape[1] // 2)
    angle_setpoint = PID(deviation, p=0.1, i=0.01, d=0.05)
    # Dựa theo góc để tính toán tốc độ hợp lí
    # Mặc dù max speed mà đề tài cho là 60 nhưng max speed của đề tài ở đây không quan trọng cái max
    # speed khai báo ở trên chỉ cho mọi người biết là không nên đi quá tốt độ 60 thôi vì trên 60 hệ thống
    # PID sẽ không ổn định và khó điều khiển nên mình vẫn có thể chỉnh thông số speed khác mà mình mong
    # muốn ở đây mình để thông số là 90 và 70 nhưng chiếm 99% là nó không chạy speed 90 vì góc 10 ít khi
    # xảy ra và cộng thêm các ảnh hưởng khác của map sẽ làm cho speed giảm nên đa phần speed sẽ chạy theo
    # tốc độ 70 nhưng vì speed 70 + với ảnh hưởng của các tính chất của map mà speed ở đây sẽ tụt xuống
    # là speed 50+ < 60
    if abs(angle_setpoint) < 10:
        # Nếu đi thẳng thì speed = 90
        adjusted_speed = 70
    else:
        # Nếu quẹo thì speed = 70
        adjusted_speed = 50
    return adjusted_speed
# Kiểm tra xem phần này có phải chương trình chính hay không
if __name__ == "__main__":
    try:
        # Vòng lặp để thu thập dư liệu và phân tích thông số dư liệu liên tục
        while True:
            # Không hiểu
            message = bytes(f"{sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            # Nhận dữ liệu từ server socket
            data = s.recv(100000)
            # Chuyển đổi data thành json
            data_recv = json.loads(data)
            # Thu thập dữ liệu về góc
            current_angle = data_recv["Angle"]
            # Thu thập dữ liệu về tốc độ
            current_speed = data_recv["Speed"]
            # Tạo ra thu thập ảnh bằng base64 :?
            jpg_original = base64.b64decode(data_recv["Img"])
            # Chuyển đổi ảnh thành 1 mảng
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            # Decode lại ảnh bằng cv2
            image = cv2.imdecode(jpg_as_np, flags=1)
            # Gọi hàm tính tọa độ hiện tại
            center_of_road = detect_yellow_line(image)
            # Gọi hàm điều chỉnh tốc độ
            speed = process_image(image)
            # Lấy tọa độ trung tâm ảnh trừ cho tọa độ hiện tại ra tọa độ đang lệch
            deviation = center_of_road - (image.shape[1] // 2)
            # Điều chỉnh lại tọa đồ đang lệch về vị trí trung tâm bằng hàm PID
            angle_setpoint = PID(deviation, p=0.1, i=0.01, d=0.05)
            print(current_speed, current_angle)
            print(image.shape)            
            # Gọi hàm điều khiển và gắn giá trị đã điều chỉnh lại độ lệch và gắn thêm giá trị điều
            # chỉnh tốc độ 
            Control(angle_setpoint, speed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print('closing socket')
        s.close()
