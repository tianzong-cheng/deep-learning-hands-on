import cv2

video = cv2.VideoCapture('../../yolov5/runs/detect/exp/video.mp4')

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame or video has ended. Exiting.")
        break

    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
