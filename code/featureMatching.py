import cv2

def featureMatching(img1,img2):
    orb = cv2.ORB_create(nfeatures=100)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # sift = cv2.xfeatures2d.SIFT_create()	 
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None) 

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches,kp1,kp2,des1,des2

if __name__=='__main__':
    img1 = cv2.imread('./Oxford_dataset/stereo/centre/1399381447017075.png')
    img2 = cv2.imread('./Oxford_dataset/stereo/centre/1399381447079564.png')
    matches,kp1,kp2,des1,des2 = featureMatching(img1,img2)

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv2.imshow('Matches', match_img)
    cv2.waitKey()
      

