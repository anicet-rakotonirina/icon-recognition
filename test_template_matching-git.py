"""-----------------------------------------------------------
11/12/2018
Template matching for icon recognition
Anicet Rakotonirina

Python program using the openCV library
and normalized template matching methods
to detect a predetermined icon in an image

TESTED ON:
-Windows 10
NOT TESTED ON:
-Linux OS
-Mac OS

PYTHON VERSION: 3.7.1

call from the command line:

python test_template_matching-git.py [template image] [test image]

source:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

------------------------------------------------------------"""

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


img = cv2.imread(sys.argv[2],0)
img2 = img.copy()
template = cv2.imread(sys.argv[1],0)
w, h = template.shape[::-1]

"""
# All the 6 methods for comparison are:
methods = ['cv2.TM_CCOEFF',
           'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 
           'cv2.TM_SQDIFF', 
           'cv2.TM_SQDIFF_NORMED']
"""

#CHANGE METHODS HERE
methods = ['cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF_NORMED']


for meth in methods:

    img = img2.copy()
    method = eval(meth)

    t0=time.time()
    res = cv2.matchTemplate(img,template,method)
    t1= time.time()-t0
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method==cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
        val= min_val
    else:
        top_left = max_loc
        val = max_val

    #ORIGINAL THRESHOLD IS 99%. CHANGE THRESHOLD HERE
    if method in [cv2.TM_CCOEFF_NORMED,cv2.TM_CCORR_NORMED]:
        if val>0.99:
            overlay="MATCH"
        else:
            overlay="NO MATCH"

    if method==cv2.TM_SQDIFF_NORMED:
        if val<0.01:
            overlay="MATCH"
        else:
            overlay="NO MATCH"

    bottom_right = (top_left[0] + w, top_left[1] + h)
    center= (top_left[0] + w/2, top_left[1] + h/2)

    #CHANGE COMMAND LINE OUTPUT HERE
    print(meth + ":" 
          + overlay 
          + ", center: x=" + str(center[0]) + "; y="+ str(center[1]) 
          + ", time = " + str('%.3f'%(t1)) + "s")

    #ORIGINAL RECTANGLE THICKNESS IS 10. CHANGE RECTANGLE THICKNESS HERE
    cv2.rectangle(img,top_left, bottom_right, 255, 10)

    #CHANGE PLOT OUTPUT HERE
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth +"\n"
                 + "time :" + str('%.3f'%(t1)) + " seconds" +"\n"  
                 + "value = " + str(val) +"\n" 
                 + overlay)

    plt.show() 