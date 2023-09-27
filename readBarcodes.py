import cv2 as cv
import os, sys
from cv2 import barcode

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

def main() :
    image_paths = fetch_images("barcodes.csv")

    if not os.path.exists("output"):
        os.mkdir("output")
    
    detector = barcode.BarcodeDetector()

    for id, path in image_paths:
        print(id, f"\"{path}\"")
        image = cv.imread(path)
        value, points = detector.detectMulti((image))
        print(f"I report {value} for {path}!")
        if value is True:
            image = get_roi_of_image(image, points)
            

if __name__ == "__main__":
    main()