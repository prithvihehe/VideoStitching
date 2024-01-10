import numpy as np
import cv2 as cv

def stitch_frames(frame1, frame2):
    gray_frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray_frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_frame1, None)
    kp2, des2 = sift.detectAndCompute(gray_frame2, None)

    flann = cv.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    kp1_coordinates = [(kp.pt[0], kp.pt[1]) for kp in kp1]
    kp2_coordinates = [(kp.pt[0], kp.pt[1]) for kp in kp2]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10.0)
    
    num_inliers = np.sum(_)
    total_matches = len(good_matches)
    accuracy = num_inliers / total_matches
    print(f"Accuracy: {accuracy:.2%}")

    h, w = gray_frame1.shape
    warped_frame1 = cv.warpPerspective(frame1, M, (w * 2, h * 2))

    result = warped_frame1.copy()
    result[0:frame2.shape[0], 0:frame2.shape[1]] = frame2

    return result, kp1_coordinates, kp2_coordinates, accuracy

# Open video capture objects
cap1 = cv.VideoCapture(r'C:\Users\prith\OneDrive\Desktop\Team Averera\Autonomy code\8vid.mp4')
cap2 = cv.VideoCapture(r'C:\Users\prith\OneDrive\Desktop\Team Averera\Autonomy code\7vid.mp4')

# Define the codec and create VideoWriter object for the output
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output_video.avi', fourcc, 30.0, (1280, 720))  # Adjust the resolution as needed
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break  # Break the loop if one of the videos ends

    stitched_frame, _, _, acc = stitch_frames(frame1, frame2)

    out.write(stitched_frame)

    if acc > 0.60: 
        cv.imshow("Stitched Video", stitched_frame)
    out.write(stitched_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
out.release()
cv.destroyAllWindows()