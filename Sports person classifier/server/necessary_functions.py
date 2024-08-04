import pywt
import cv2
import numpy as np
import base64


# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")


def get_cropped_image_for_2_eyes(image_array):

    img_main = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img_main is None:
        print('Not an image')
        return None
    else:
        try:
            gray_img = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
        except:
            print('Gray image is not available')
            return None
        else:
            faces_gray = face_cascade.detectMultiScale(gray_img, 1.1, 5)
            for (x,y,w,h) in faces_gray:
                roi_color = img_main[y:y+h, x:x+w]
                eyes_gray = eye_cascade.detectMultiScale(gray_img, 1.1, 5)
                if len(eyes_gray) >=1:
                    _, buffer = cv2.imencode('.png', roi_color)
                    return buffer.tobytes()
    
    return None


# I got this function from stack overflow
def w2d(img, mode='db1', level=5):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


def get_the_features(cropped_array):
    X = []
    cropped_main = cv2.imdecode(cropped_array, cv2.IMREAD_COLOR)
    if cropped_main is not None:  
        scalled_img = cv2.resize(cropped_main, (32, 32)) 
        wavelet_img = w2d(cropped_main, 'db1', 5 )
        scalled_wavelet_img = cv2.resize(wavelet_img, (32, 32)) 

        combined_image = np.vstack((scalled_img.reshape(32*32*3,1), scalled_wavelet_img.reshape(32*32,1))) # I'm stacking 
        X.append(combined_image)
        return X