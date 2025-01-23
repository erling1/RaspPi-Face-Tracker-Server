import imutils
import cv2


def ObjCenter(rects, frameCenter):


    if len(rects)>0:
               (x, y, w, h) = rects[0] 
               faceX = int(x + (w / 2.0)) 
               faceY = int(y + (h / 2.0))

               return ((faceX,faceY), rects[0])
          

    return (frameCenter, None)





