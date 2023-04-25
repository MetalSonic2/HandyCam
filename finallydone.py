import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from flask import Flask, render_template, Response


app = Flask(__name__)


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
classifier2 = Classifier("Model/keras_model.h5", "ModelLeft/labels.txt")

offset = 20
imgSize = 300

# folder = "Data/C" used for test
# counter = 0

labels = ["A", "B", "C"] #index/ get values
letter_counts = {label: 0 for label in labels} #how many times each label appears
last_prediction = {}
current_prediction = {label: 0 for label in labels}


#the page the camera goes to
@app.route('/')
def index():
    return render_template('ProjectWebsite.html')


def gen():
    global last_prediction
    while True:
        success, img = cap.read() #

        if not success or img is None:  #fixes the error when hand is out (i hope)
            continue
        imgOutput = img.copy() #copy of our image
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox'] #bounding box coordinate x,y,w,h

            # size of image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w
#fixes heigth
            if aspectRatio > 1:
                k = imgSize / h #constant = 300/height
                wCal = math.ceil(k * w) #calculated width
                if imgCrop.size > 0:
                    ImgResize = cv2.resize(imgCrop, (wCal, imgSize))
                else:
                    continue
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2) #push gap to center=image size-width/2
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False) #give prediction and index
                letter_counts[labels[index]] += 1

            else:
                #fixes width
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED) #size of rectangle
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2) #put label text

            # Display the percentage on camera
            total_frames = sum(letter_counts.values())
            if total_frames > 0:
                percentage = letter_counts[labels[index]] / total_frames * 100
            else:
                percentage = 0

            cv2.putText(imgOutput, f"{labels[index]}: {percentage:.2f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2) #shows label and percent on screen

            if percentage >= 70 and 80 > percentage:
                cv2.putText(imgOutput, f"Almost there Analyzing... {labels[index]}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            if percentage >= 80:
                cv2.putText(imgOutput, f"you are correct: {labels[index]}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            for label, percentage in current_prediction.items():
                if percentage >= 80:
                    print(f"You are correct: {label}")

            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', imgOutput)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Left')
def left():
    return render_template('ProjectWebsiteLeft.html')
def gen2():
    global last_prediction
    while True:
        success, img = cap.read() #

        if not success or img is None:  #fixes the error when hand is out (i hope)
            continue
        imgOutput = img.copy() #copy of our image
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox'] #bounding box coordinate x,y,w,h

            # size of image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w
#fixes heigth
            if aspectRatio > 1:
                k = imgSize / h #constant = 300/height
                wCal = math.ceil(k * w) #calculated width
                if imgCrop.size > 0:
                    ImgResize = cv2.resize(imgCrop, (wCal, imgSize))
                else:
                    continue
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2) #push gap to center=image size-width/2
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier2.getPrediction(imgWhite, draw=False) #give prediction and index
                letter_counts[labels[index]] += 1

            else:
                #fixes width
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier2.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED) #size of rectangle
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2) #put label text

            # Display the percentage on camera
            total_frames = sum(letter_counts.values())
            if total_frames > 0:
                percentage = letter_counts[labels[index]] / total_frames * 100
            else:
                percentage = 0

            cv2.putText(imgOutput, f"{labels[index]}: {percentage:.2f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2) #shows label and percent on screen

            if percentage >= 70 and 80 > percentage:
                cv2.putText(imgOutput, f"Almost there Analyzing... {labels[index]}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            if percentage >= 80:
                cv2.putText(imgOutput, f"you are correct: {labels[index]}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            for label, percentage in current_prediction.items():
                if percentage >= 80:
                    print(f"You are correct: {label}")

            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', imgOutput)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/quiz')
def quiz():
    return render_template('Quizpage01.html')



@app.route('/quiz1')
def quiz1():
    return render_template('quiz.html')



if __name__ == '__main__':
    app.run(debug=True,port=5000, host='0.0.0.0')
