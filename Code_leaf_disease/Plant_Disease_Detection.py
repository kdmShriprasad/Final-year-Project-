from robotModule import Robot
import WebcamModule
import cv2
import utlis
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import tcp_listener as srv
from time import sleep
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO

#### Change this for leaf green color ####
lowerGreen = np.array([30, 40, 20])
upperGreen = np.array([90, 215, 191])
#####################################

mot = 10
GPIO.setup(mot,GPIO.OUT)
GPIO.output(mot,GPIO.HIGH)

robo = Robot(2,4,17,3,22,27)
# MOT1: Gpio 4 = IN1,  GPIO 17 = IN2
# MO2: Gpio 27 = IN1,  GPIO 22 = IN2
#motor= Motor(23,4,17,24,27,22)		# EnaA,In1A,In2A,EnaB,In1B,In2B
robo.stop(2)

# Define host and port
host = '0.0.0.0'  # Listen on all available interfaces
#host = '192.168.30.55'  # Listen on all available interfaces
port = 8080  # Choose any available port (above 1024 for non-privileged ports)

def motor_on():
    GPIO.output(mot,GPIO.LOW)
def motor_off():
    GPIO.output(mot,GPIO.HIGH)

def LeafDetection(img):

    mask = utlis.thresholding(img, lowerGreen, upperGreen)      # Color detection, extract only desired colour
    
    kernel = np.ones((10,10),np.uint8)

    dilate = cv2.dilate(mask,kernel,iterations = 1)
    #cv2.imshow('Dilate', dilate)
    erode = cv2.erode(dilate, kernel, iterations = 1)
    #cv2.imshow('Erode', erode)
    opened = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('Open', opened)
 
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('Close', closed)
    
    final = closed
    result = cv2.bitwise_and(img, img, mask=final)
    #cv2.imshow('Masked', result)

    # Find contours in the image
    contours = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    print('Total objects', len(contours))
    
    ## Get the properties of each contour
    #props = cv2.moments(contours[0])
    ## Print the properties
    #print(props)

    # Draw the contours on the image
    #cv2.drawContours(img, contours, -1, (255, 255, 0), 3)
    #cv2.imshow('Leaf Contours', img)

    #imgStack = utlis.stackImages(0.6, ([dilate, erode], [opened, closed], [img, result]))
    #cv2.imshow("Stacked Images", imgStack)

    

    #cv2.imwrite('Result.jpg', imgStack)
    return final


def FindDiseaseArea(binImg):
    # Invert image
    inv = cv2.bitwise_not(binImg)

    # Copy the thresholded image.
    im_floodfill = binImg.copy()
 
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = binImg.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
 
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    disease = cv2.bitwise_not(im_floodfill)
    #cv2.imshow('Disease', disease)

    # Combine the two images to get the foreground.
    wholeLeaf = binImg | disease

    cv2.bitwise_and(wholeLeaf, wholeLeaf, mask=inv)
    

    return wholeLeaf, disease

def NormalizeImg(img):
    # Read the image
    #img = cv2.imread('image.jpg')

    # Get the EXIF metadata
    #exif = cv2.getEXIFData(img)
    #cv2.normalize(img, img,)

    ## Get the aperture and ISO
    #aperture = exif.get('ApertureValue')
    #iso = exif.get('ISOSpeedRatings')

    ## Print the aperture and ISO
    #print('Aperture:', aperture)
    #print('ISO:', iso)

    normalizedimage = cv2.normalize(img, img, 0, 100, cv2.NORM_MINMAX)
    return normalizedimage

def main(cam = False, keyInt = False):
    curpath = os.getcwd()
    #print(curpath)
    folderName = '/Result/'
    resultPath = curpath + folderName
    print(resultPath)

    if os.path.exists(resultPath) == False:
        print('Folder created')
        os.mkdir(folderName)
    else:
        print('Folder present')
        
    root = tk.Tk()
    root.withdraw()
        
    if cam == True:
        img = WebcamModule.getImg()
    else:
         # Open file dialog, get file path   
        file_path = filedialog.askopenfilename()
        if not file_path:
            return -1
        # From file path read image
        img = cv2.imread(file_path)
    
    
    img = cv2.resize(img, (480, 240), interpolation = cv2.INTER_LINEAR)
    white = [255,255,255]     #---Color of the border---
    # Draw border around image
    img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=white )
    #cv2.imshow('Orig Img', img)
    # Save image
    cv2.imwrite(resultPath + 'Orig_Img.jpg', img)

    img = NormalizeImg(img)

    #cv2.imshow('Norm Img', img)
    
    # Make a Copy original image
    origImg = img.copy()

    # Detect color of leaf & create leaf mask
    final = LeafDetection(img)

    # Detect disease area & create mask
    leaf, dis = FindDiseaseArea(final)

    #cv2.imshow('Leaf Mask', leaf)
    cv2.imwrite(resultPath + 'Leaf_Mask.jpg', leaf)
    #cv2.imshow('Disease Mask', dis)
    cv2.imwrite(resultPath + 'Disease_Mask.jpg', dis)

    # Find contours in the leaf
    contoursLeaf = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Draw the contours on the image
    cv2.drawContours(img, contoursLeaf, -1, (255, 255, 0), 2)

    # Find contours in the disease
    contoursDis = cv2.findContours(dis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Draw the contours on the image
    cv2.drawContours(img, contoursDis, -1, (0, 0, 255), 2)
    #cv2.imshow('Contours', img)
    cv2.imwrite(resultPath + 'Contours.jpg', img)
 
    # Calculate the area of the mask
    leafArea = np.sum(leaf == 255)
    disArea = np.sum(dis == 255)

    disPer = disArea / leafArea * 100

    print(f"Total leaf area: {leafArea}\nDisease area: {disArea}\nPercent Disease: {disPer}%")

    imgStack = utlis.stackImages(0.6, ([origImg, img], [leaf, dis]))
    cv2.imshow("Plant Disease Detection", imgStack)
    
    cv2.imwrite(resultPath + 'Final_Result.jpg', imgStack)
    '''
    plt.imshow(imgStack)
    plt.title('Final Output')
    plt.axis('off')
    plt.show()
    '''

    
    if disPer > 0:
        print('************ Disease Detected!! ************')
    else:
        print('************ No Disease ************')
    
    if keyInt == True:
        # if keyboard interrupt required
        key = 0
        while key != 27:		# wait for escape key
            key = cv2.waitKey(1)

            if(key == 27):
                break
    else:
        key = cv2.waitKey(2000)
    
    return disPer

if __name__ == '__main__':
    server_socket, serverIp = srv.listnerStartSocket(host, port)
    print("Server Ip: ", serverIp)
    print(f"Listening for connections on {host}:{port}...")
    
    while True:
        client_socket, client_address = srv.listnerAcceptConnection(server_socket)
        
        while True:
            data = srv.listnerRecData(client_socket)
            if not data: 
                    break # Exit loop if no data is received
            
            cmd, hdr = srv.listnerParsInputHeader(data)

            cmd2 = str(cmd)

            print("Commad:", cmd2)
            
            if cmd2.find('State=F') > 0:
                print("Robot Forward")
                robo.stop(0.5)
                robo.forward()		# Forward
            if cmd2.find('State=S') > 0:
                print("Robot Stop")
                robo.stop()
            
            if cmd2.find('State=W') > 0:
                print("Detect command")
                robo.stop(0.5)
                dis = main(True, False)		# Camera = True, Wait for key = Flase
                if dis > 0:
                  print("Starting spray")
                  motor_on()
                  sleep(3)
                  print("Spray off")
                  motor_off()

            if cmd2.find('State=V') > 0:
                print("Detect command")
                robo.stop(0.5)
                dis = main(False, False)		# Camera = False, Wait for key = Flase
                if dis > 0:
                  print("Starting spray")
                  motor_on()
                  sleep(3)
                  print("Spray off")
                  motor_off()

            if cmd2.find('State=B') > 0:
                print("Robot Backward")
                robo.stop(0.5)
                robo.backward()		# Backward
            if cmd2.find('State=L') > 0:
                print("Robot Left")
                robo.left()			# Left
            if cmd2.find('State=R') > 0:
                print("Robot Right")
                robo.right()		# Right

            client_socket.send("Received data..\n".encode())
        
        



