## Wrapper written by Peter Lee

import sys
import os

# get the path to THIS file.
basepath = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, '{}/../'.format(basepath))
from rfnet.model.rf_des import HardNetNeiMask
from rfnet.model.rf_det_so import RFDetSO
from rfnet.model.rf_net_so import RFNetSO
import torch
import cv2

# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, size=1): 
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        #kps = [ cv2.KeyPoint(float(p[1]), float(p[2]), _size = 1.0,  _response=scores[0,p[1],p[2],0]) for i,p in enumerate(pts) ]
        kps = [ cv2.KeyPoint(float(p[2]), float(p[1]), _size = 1.0,  _response=scores[0,p[1],p[2],0]) for i,p in enumerate(pts) ]
    return kps         


# interface for pySLAM
class RFNetFeature2D:
    def __init__(self, num_features = 1000):
        # Initialize detector

        # Prepare argumentsL just default arguments to RFNet
        BATCH_SIZE = 1
        EPOCH_NUM = 201
        LOG_INTERVAL = 5
        WEIGHT_DECAY = 1e-4
        DET_LR = 0.1
        DES_LR = 10
        DET_OPTIMIZER = "adam"
        DET_LR_SCHEDULE = "exp"
        DET_WD = 0
        DES_OPTIMIZER = "adam"
        DES_LR_SCHEDULE = "sgd"
        DES_WD = 0
        LR_DECAY_EPOCH = 5
        LR_BASE = 0.0001
        score_com_strength = 100.0
        scale_com_strength = 100.0,
        NMS_THRESH = 0.0
        NMS_KSIZE = 5
        TOPK = 512
        SCORE = 1000
        PAIR = 1
        PATCH_SIZE = 32
        HARDNET_MARGIN = 1.0
        COO_THRSH = 5.0
        scale_list = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
        padding = 1
        dilation = 1
        GAUSSIAN_KSIZE = 15
        KSIZE = 3
        # gaussian kernel sigma
        GAUSSIAN_SIGMA = 0.5

        
        # Initialize the model.
        det = RFDetSO(
            score_com_strength,
            scale_com_strength,
            NMS_THRESH,
            NMS_KSIZE,
            TOPK,
            GAUSSIAN_KSIZE,
            GAUSSIAN_SIGMA,
            KSIZE,
            padding,
            dilation,
            scale_list,
        )
        des = HardNetNeiMask(HARDNET_MARGIN, COO_THRSH)
        model = RFNetSO(
            det, des, SCORE, PAIR, PATCH_SIZE, TOPK
        )

        # load the nodel
        #device = torch.device("cuda")
        device = torch.device("cuda")
        self.device = device
        model = model.to(device)

        # Load the model
        resume = '{}/../runs/10_24_09_25/model/e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar'.format(basepath)

        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        self.model = model

    def detectAndCompute(self, frame, mask = None):
        # Call RFNet model.
        output_size = frame.shape
        kp, des, img, scores = self.model.detectAndCompute_fromimg(frame, self.device, output_size)
        kp = kp.detach().cpu().numpy()
        des = des.detach().cpu().numpy()

        kps  = convert_pts_to_keypoints(kp, scores.detach().cpu().numpy(), size=kp.shape)
        
        self.kps = kps
        self.des = des


    
        return kps, des

    def detect(self, frame, mask = None):
        self.detectAndCompute(frame)
        return self.kps


    def compute(self, frame, kps=None, mask=None): 
        self.detectAndCompute(frame)
        return self.kps, self.des   
    
