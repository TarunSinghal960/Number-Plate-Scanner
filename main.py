import cv2
import datetime

def get_time():
    x = datetime.datetime.now()
    current_time = str(x.date()) + "_" + str(x.hour) + "_" + str(x.minute) + str(x.second)
    return current_time

plate_number_cascade = cv2.CascadeClassifier("resources/haarcascade_russian_plate_number.xml")
min_area = 500
color = (255, 0, 255)

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ROI = img.copy()

    plate_numbers = plate_number_cascade.detectMultiScale(gray_img, 1.1, 4)

    for (x, y, w, h) in plate_numbers:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            img_ROI = img[y:y+h, x:x+w]
            cv2.imshow("ROI", img_ROI)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("resources/scanned images/NoPlate_" + get_time() + ".jpg", img_ROI)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Saved!!", (150, 265), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Webcam", img)
        cv2.waitKey(1000)