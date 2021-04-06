import sys 
import cv2
from datetime import datetime
from webcam import Webcam

import sys
sys.path.append("../")

from timer import Timer
import os
# CHESSBOARD SIZE
#chessboard_size = (11,7)
chessboard_size = (5,8)
#chessboard_size = (10,6)

# grab an image every 
kSaveImageDeltaTime = 1  # second

if __name__ == "__main__":
    bp = sys.argv[1]
    do_save = True
    for im_path in os.listdir(sys.argv[1]):

        
        image = cv2.imread('{}/{}'.format(bp, im_path))#webcam.get_current_frame()
        if image is not None: 

            # check if pattern found
            ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), chessboard_size, None)
            print(ret)
            print(image.shape)
        
            if ret == True:     
                print('found chessboard')
                # save image
                #filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.bmp'
                filename = im_path
                image_path="./calib_images/" + filename
                
                if do_save:
                    print('saving file ', image_path)
                    cv2.imwrite(image_path, image)

                # draw the corners
                image = cv2.drawChessboardCorners(image, chessboard_size, corners, ret)                       

            cv2.imshow('camera', image)                

        else: 
            pass
            #print('empty image')                
                            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
   
