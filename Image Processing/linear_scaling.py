import cv2
import numpy as np
import sys

# read arguments
if(len ( sys.argv ) != 3) :
    print ( sys.argv [0] , ": takes 2 arguments .Not ", len( sys.argv ) -1)
    print (" Expecting arguments : ImageIn ImageOut .")
    print (" Example :", sys.argv [0] , " fruits .jpg out.png")
    sys.exit ()

name_input = sys.argv [1]
name_output = sys.argv [2]

# read image
inputImage = cv2.imread ( name_input , cv2.IMREAD_COLOR )
if( inputImage is None ) :
    print ( sys.argv [0] , ": Failed to read image from : ", name_input )
    sys.exit ()
cv2.imshow (" input image : " + name_input , inputImage )

rows , cols , bands = inputImage . shape
if( bands != 3) :
    print (" Input image is not a standard color image :", inputImage )
    sys.exit ()

img_Luv = cv2.cvtColor (inputImage , cv2.COLOR_BGR2LUV )
l = list(np.concatenate(img_Luv[:,:,0]).flat)
l_max = max(l)
l_min = min(l)
print(l_max)
print(l_min)
cv2.imshow (" Luv ", img_Luv )
for i in range(0, rows):
    for j in range(0, cols):
        L, u, v = img_Luv[i, j]
        new_L = 255 * (L - l_min) / (l_max - l_min)
        img_Luv [i , j ] = [ new_L , u , v ]

new_l = list(np.concatenate(img_Luv[:,:,0]).flat)
print(max(new_l))
print(min(new_l))

cv2.imshow (" Modified Luv ", img_Luv )
outputImage = cv2 . cvtColor (img_Luv , cv2.COLOR_LUV2BGR )
cv2.imshow (" output image ", outputImage )

# saving the output
cv2.imwrite ( name_output , outputImage )

# wait for key to exit
cv2.waitKey (0)
cv2.destroyAllWindows ()