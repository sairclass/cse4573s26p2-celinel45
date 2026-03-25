'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    
    imgsTensors = list(imgs.values())
    num = len(imgs)
    for i in range(num):
        imgsTensors[i] = imgsTensors[i].float() / 255.0

    # Extract a set of key points for each image. Extract features from each key point.
    keypoints = {}
    descriptors = {}
    for i, img in enumerate(imgsTensors):
        img = imgsTensors[i].unsqueeze(0) # (1, C, H, W)
        gray = K.color.rgb_to_grayscale(img)
        sift = K.feature.SIFTFeature(1000, rootsift=True) # use sift for feature extraction
        lafs, responses, descs = sift(gray) # descriptors: (1, N, 128), lafs: (1, N, 2, 3)
        keypoints[i] = K.feature.get_laf_center(lafs)[0] # [N, 2] N points
        print("keypts shape: ", keypoints[i].shape)
        descriptors[i] = descs[0] # [N, 128]
        print("descts shape: ", descriptors[i].shape)
    

    matching = K.feature.DescriptorMatcher("smnn", 0.85) # sift feature matching
    distances, indexs = matching(descriptors[0], descriptors[1])
    print("matched points: ", len(indexs))
    if len(indexs) < 4:
        print("at least 4 points are needed.")
    
    pts1 = keypoints[0][indexs[:, 0]] # indexs[:,0] first column -> matched index of first image
    pts2 = keypoints[1][indexs[:, 1]] # indexs[:,1] second column -> matched index of second image
    ransac = K.geometry.ransac.RANSAC()
    h, m = ransac(pts1, pts2) # img1 -> img2
    inliers = m.sum().item()
    if inliers < 20:
        print("Not enough inliers.")
    
    c1, h1, w1 = imgsTensors[0].shape
    corners1 = torch.tensor([[0, 0, 1], 
                            [w1, 0, 1],
                            [w1, h1, 1], 
                            [0, h1, 1] ], dtype=torch.float32)
    homoPts1 = (h @ corners1.T).T # 4x3 points in img i -> baseImg - img 2
    homoPts1 = homoPts1[:, :2] / homoPts1[:, 2:3] # (x, y) -> (x/w, y/w) 4x2 

    c2, h2, w2 = imgsTensors[1].shape
    xMin, yMin = int(min(0, homoPts1[:, 0].min())), int(min(0, homoPts1[:, 1].min()))
    xMax, yMax = int(max(w2, homoPts1[:, 0].max())), int(max(h2, homoPts1[:, 1].max()))
    width = xMax - xMin
    height = yMax - yMin
    size = (height, width)
    # translate value, if less than 0, than translate
    if xMin < 0:
        X = -xMin
    else: 
        X = 0
    if yMin < 0:
        Y = -yMin
    else: 
        Y = 0
    translate = torch.tensor([[1, 0, X],
                            [0, 1, Y],
                            [0, 0, 1]], dtype=torch.float32)

    h = translate @ h # translate if partial pts on the left side

    # img 1 after warp
    res = K.geometry.transform.warp_perspective(imgsTensors[0].unsqueeze(0), h.unsqueeze(0), size)
    baseNew = torch.zeros(1, c1, height, width)
    baseNew[0, :, Y:Y+h2, X:X+w2] = imgsTensors[1].unsqueeze(0)
    print("res shape: ", res.shape)
    print("baseNew shape: ", baseNew.shape)

    img = 0.5 * res + 0.5 * baseNew
    # mask1 = (res > 0).float()
    # mask2 = (baseNew > 0).float()
    # dtrans1 = K.contrib.distance_transform(mask1)
    # dtrans2 = K.contrib.distance_transform(mask2)

    # alpha = dtrans1 / (dtrans1 + dtrans2 + 0.01) # center weighting

    # blending = alpha * res + (1 - alpha) * baseNew

    # overlapRegion = (mask1 == 1) & (mask2 == 1) # overlap region of img i and baseImg
    # img = overlapRegion.float() * blending + (1 - overlapRegion.float()) * (res + baseNew)
    img = (img[0] * 255).to(torch.uint8)
    # show_image(img)
    
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    imgsTensors = list(imgs.values())
    num = len(imgs)
    for i in range(num):
        imgsTensors[i] = imgsTensors[i].float() / 255.0

    keypoints, descriptors = computeKeypoints(imgsTensors)

    H = {}
    overlap = torch.eye(num) # each image overlap with itself

    for i in range(num):
        for j in range(i+1, num):
            matching = K.feature.DescriptorMatcher("smnn", 0.85) # sift feature matching
            distances, indexs = matching(descriptors[i], descriptors[j])
            if len(indexs) < 4:
                print("at least 4 points are needed.")
                continue
            
            H = findHomography(indexs,keypoints, imgsTensors, overlap, H, i, j)
            
    
    # get the maximum overlapped image as base image
    baseIndex = torch.argmax(overlap.sum(dim=1)).item()
    baseImg = imgsTensors[baseIndex].clone() 
    t = torch.eye(3) # total translate
    hgraphy = torch.eye(3)
    for i in range(num):
        if overlap[i, baseIndex] == 0: # Images that do not overlap with any other image can be ignored.
            continue
        if (i, baseIndex) in H:
            hgraphy = H[(i, baseIndex)] # homograpgy from img i -> baseImg
        elif (baseIndex, i) in H:
            hgraphy = torch.inverse(H[(baseIndex, i)]) # inverse to get homograpgy from img i -> baseImg
        else:
            print("cannot find H.")
            continue
        
        hgraphy = t @ hgraphy
        # print(imgsTensors[i].shape) # C, H, W
        c1, h1, w1 = imgsTensors[i].shape
        pts1 = torch.tensor([[0, 0, 1], 
                            [w1, 0, 1],
                            [w1, h1, 1], 
                            [0, h1, 1] ], dtype=torch.float32)
        homoPts1 = (hgraphy @ pts1.T).T # 4x3 points in img i -> baseImg
        homoPts1 = homoPts1[:, :2] / homoPts1[:, 2:3] # (x, y) -> (x/w, y/w) 4x2 

        # print(baseImg.shape) # C, H, W
        c1, baseH, baseW = baseImg.shape
        

        # find pts boundry after homography
        xMin, yMin = int(min(0, homoPts1[:, 0].min())), int(min(0, homoPts1[:, 1].min()))
        xMax, yMax = int(max(baseW, homoPts1[:, 0].max())), int(max(baseH, homoPts1[:, 1].max()))
        width = xMax - xMin
        height = yMax - yMin
        size = (height, width)
        previous = t
        # translate value, if less than 0, than translate
        if xMin < 0:
            X = -xMin # move to right
        else: 
            X = 0
        if yMin < 0:
            Y = -yMin # move down
        else: 
            Y = 0
        translate = torch.tensor([[1, 0, X],
                          [0, 1, Y],
                          [0, 0, 1]], dtype=torch.float32)
    
        t = translate @ t
        hgraphy = translate @ hgraphy # translate if partial pts on the left side

        # img i after warp
        res = K.geometry.transform.warp_perspective(imgsTensors[i].unsqueeze(0), hgraphy.unsqueeze(0), size)
        # show_image(res.squeeze(0))

        # base = baseImg.unsqueeze(0) # 1xCxHxW
        baseNew = torch.zeros(1, c1, height, width)
        baseNew[0, :, Y:Y+baseH, X:X+baseW] = baseImg
        # show_image(baseNew.squeeze(0))

        mask1 = (res > 0).float()
        mask2 = (baseNew > 0).float()

        dtrans1 = K.contrib.distance_transform(mask1) 
        dtrans2 = K.contrib.distance_transform(mask2)
    
        alpha = dtrans1 / (dtrans1 + dtrans2 + 0.01) # center weighting
        # blending = alpha * collapse + (1-alpha) * base
        blending = alpha * res + (1 - alpha) * baseNew
        # regionI = (mask1 == 1) & (mask2 == 0)
        overlapRegion = (mask1 == 1) & (mask2 == 1) # overlap region of img i and baseImg
        combine = overlapRegion.float() * blending + (1 - overlapRegion.float()) * (res + baseNew)
        
        baseImg = combine[0]
        
        # baseImg = res[0] # update baseImg

        # t = previous @ t # update translate matrix
        # show_image(baseImg)
    
    img = (baseImg * 255).to(torch.uint8)
    return img, overlap


def computeKeypoints(imgs):
        # for i in range(len(imgs)):
        #     imgs[i] = imgs[i].unsqueeze(0) # (1, C, H, W)
        keypts = {}
        descts = {}
        for i, img in enumerate(imgs):
            img = imgs[i].unsqueeze(0) # (1, C, H, W)
            gray = K.color.rgb_to_grayscale(img)
            sift = K.feature.SIFTFeature(1000, rootsift=True) # use sift for feature extraction
            lafs, responses, descriptors = sift(gray) # descriptors: (1, N, 128), lafs: (1, N, 2, 3)
            keypts[i] = K.feature.get_laf_center(lafs)[0] # [N, 2] N points
            print("keypts shape: ", keypts[i].shape)
            descts[i] = descriptors[0] # [N, 128]
            print("descts shape: ", descts[i].shape)
        return keypts, descts
    
def findHomography(indexs, keypoints, imgsTensors, overlap, H, i, j):


    pts1 = keypoints[i][indexs[:, 0]] # indexs[:,0] first column -> matched index of first image
    pts2 = keypoints[j][indexs[:, 1]] # indexs[:,1] second column -> matched index of second image
    homography = K.geometry.ransac.RANSAC()
    h, m = homography(pts1, pts2) # img1 -> img2
    inliers = m.sum().item()
    print(f"H[{i},{j}]:")
    print(h)
    # if inliers < 50:
    #     continue
    # find the min and max of x, y, and compute overlap area in img i
    xMin1, yMin1 = pts1.min(0).values
    xMax1, yMax1 = pts1.max(0).values
    match1 = (xMax1 - xMin1) * (yMax1 - yMin1)
    c1, h1, w1 = imgsTensors[i].shape
    area1 = h1 * w1
    overlapRatio1 = match1 / area1
    # find the min and max of x, y, and compute overlap area in img j
    xMin2, yMin2 = pts2.min(0).values
    xMax2, yMax2 = pts2.max(0).values
    match2 = (xMax2 - xMin2) * (yMax2 - yMin2)
    c2, h2, w2 = imgsTensors[j].shape
    area2 = h2 * w2
    # area2 = imgsTensors[j].shape[1] * imgsTensors[j].shape[2]
    overlapRatio2 = match2 / area2
    overlapRatio = max(overlapRatio1, overlapRatio2)
    if overlapRatio >= 0.2: #  an image is to be part of the panorama, it will overlap at least one other image by at least 20%.
        overlap[i, j] = 1
        overlap[j, i] = 1
    H[(i, j)] = h
    return H
