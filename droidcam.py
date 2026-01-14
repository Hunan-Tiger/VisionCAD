import cv2
# 打开默认摄像头 (0 表示默认摄像头)
cap = cv2.VideoCapture(0)

# 设置分辨率为1920*1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取一帧
    ret, frame = cap.read()
    
    # 如果正确读取帧,ret为True
    if not ret:
        print("无法接收帧 (流结束?)")
        break
    
    # 显示帧
    cv2.imshow('Camera', frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
