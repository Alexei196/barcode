import cv2 as cv
import os, sys
from cv2 import barcode

import keras_ocr

def fetch_images(path: str):
    pathList = [] 
    with open(path, "r") as file:
        for line in file.readlines()[1:]:
            split = line.rstrip("\n").split(",")
            pathList.append((split[0], split[1]))
    return pathList

def get_roi_of_image(image, points):
    x, y, w, h = cv.boundingRect(points)
    x = max(x-200, 0)
    y = max(y-200, 0)
    w = w + 400
    h = h + 400
    # cv.imshow("subset", cv.resize(image[y:y+h, x:x+w], (512, 512)))
    # cv.waitKey(0)
    return image[y:y+h, x:x+w]

def detect_text_in_image(path, pipeline):
    image = keras_ocr.tools.read(path)
    # data is of type (word, boundbox) where boundingbox is 4 connected segments
    data = pipeline.recognize([image])[0]
    print(f"The word is {data[0]}")
    image = cv.line(image, data[1][0], data[1][1])
    image = cv.line(image, data[1][1], data[1][2])
    image = cv.line(image, data[1][2], data[1][3])
    image = cv.line(image, data[1][3], data[1][0])

def main() :
    image_paths = fetch_images("barcodes.csv")

    if not os.path.exists("output"):
        os.mkdir("output")
    
    detector = barcode.BarcodeDetector()
    pipeline = keras_ocr.pipeline.Pipeline()

    for id, path in image_paths:
        image = cv.imread(path)
        value, points = detector.detectMulti((image))
        print(f"I report {value} for {path}!")
        if value is True:
            image = get_roi_of_image(image, points)  
            detect_text_in_image(path, pipeline)
            

if __name__ == "__main__":
    main()