import cv2
import numpy as np
import itertools
import time
import os
from os import listdir
from os.path import isfile, join
from model.yolo_model_architecture import *
from model.keras_ocr_model_architecture import *


# define some parameters that OCR will need
CHAR_VECTOR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
letters = [letter for letter in CHAR_VECTOR]
num_classes = len(letters) + 1


def get_lp(input_image_filename, 
           input_image_folder,
		   yolo_model,
		   keras_ocr_model):
    
    # read in the input image from the specified folder
    original_image = cv2.imread(input_image_folder + input_image_filename)
    
    # set up a dummy array for using when predicting
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    
    # preprocess the image a little bit
    image = original_image.copy()
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)

    # predict
    netout = yolo_model.predict([input_image, dummy_array])

    # define the boxes from the prediction
    boxes = decode_netout(netout[0], 
                          obj_threshold=OBJ_THRESHOLD,
                          nms_threshold=NMS_THRESHOLD,
                          anchors=ANCHORS, 
                          nb_class=CLASS)
    
    # now use the boxes to crop down to the detected lp within the image
    if len(boxes) == 0:
        print("No license plate detected for", input_image_filename)
    elif len(boxes) == 1:
        for box in boxes:
            # save the height and width of the original image 
            image_h, image_w, _ = original_image.shape

            # get x,y min,max for cropping down the original image to the lp that the model predicted/detected
            xmin = int(box.xmin*image_w)
            ymin = int(box.ymin*image_h)
            xmax = int(box.xmax*image_w)
            ymax = int(box.ymax*image_h)

            # crop down to just the lp that the model detected
            detected_lp = original_image[ymin:ymax, xmin:xmax]
    else:
        print(len(boxes), "boxes were detected for", input_image_filename)
        # identify the box with the highest score and use that one for output
        scores = []
        for box in boxes:
            scores.append(box.get_score())
        box = boxes[np.argmax(scores)]
        output_score = box.get_score()    
        # save the height and width of the original image 
        image_h, image_w, _ = original_image.shape

        # get x,y min,max for cropping down the original image to the lp that the model predicted/detected
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        # crop down to just the lp that the model detected
        detected_lp = original_image[ymin:ymax, xmin:xmax]
    
    # TODO: remove this after figuring out how to reshape
    cv2.imwrite("outputs/output_" + input_image_filename, detected_lp[:,:,::-1])
    #return detected_lp
    # read back in the image.... there has to be away around this...
    img = cv2.imread("outputs/output_" + input_image_filename, cv2.IMREAD_GRAYSCALE)
    get_ocr_prediction(input_image=detected_lp,
                       input_image_grayscale=img,
                       input_image_filename=input_image_filename,
					   keras_ocr_model=keras_ocr_model)


def decode_label(out):

    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr


def get_ocr_prediction(input_image,
                       input_image_grayscale,
                       input_image_filename,
					   keras_ocr_model):
    
    # initialize some things
    total = 0
    acc = 0
    letter_total = 0
    letter_acc = 0
    start = time.time()

    img_pred = input_image_grayscale.astype(np.float32)
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)

    net_out_value = keras_ocr_model.predict(img_pred)
    #pred_texts = decode_label(net_out_value)

    #for i in range(min(len(pred_texts), len(input_image_filename[0:-7]))):
    #    if pred_texts[i] == input_image_grayscale[i]:
    #        letter_acc += 1
    #letter_total += max(len(pred_texts), len(input_image_filename[0:-7]))

    #if pred_texts == input_image_filename[0:-7]:
    #    acc += 1
    #total += 1
    
    #print('Predicted LP: %s | True LP (from filename): %s' % (pred_texts, input_image_filename[0:-7]))
    
    # bk edits
    pred_texts = decode_label(net_out_value).strip('_').split('-')[0]
    
    for i in range(min(len(pred_texts), len(input_image_filename.strip('_').split('-')[0]))):
        if pred_texts[i] == input_image_filename[i]:
            letter_acc += 1
    letter_total += max(len(pred_texts), len(input_image_filename.strip('_').split('-')[0]))

    if pred_texts == input_image_filename.strip('_').split('-')[0]:
        acc += 1
    else:
        print('Predicted: %s  /  True: %s' % (pred_texts, input_image_filename.strip('_').split('-')[0]))
    total += 1
    
    end = time.time()
    total_time = (end - start)
#     print("Time : ",total_time / total)
#     print("ACC : ", acc / total)
#     print("letter ACC : ", letter_acc / letter_total)

    # TODO: add some plotting!
    image_h, image_w, _ = input_image.shape
    cv2.putText(input_image, 
                pred_texts, 
                (int(image_w * 0.40), # horizontal
                 int(image_h - (image_h * 0.05))), #vertical
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.005 * image_h, #1e-3 * image_h, 
                (0,255,0), 2)

    plt.figure(figsize=(10,10))
    plt.imshow(input_image[:,:,::-1]); plt.show()
  