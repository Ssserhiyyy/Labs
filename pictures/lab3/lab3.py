import cv2
import numpy as np
import matplotlib.pyplot as plt

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read()
        return cv2.imdecode(np.frombuffer(chunk, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Помилка читання: {e}")
        return None

path_l = 'G:/allLabs/обробка зображень/lab3/photo_2026-03-09_16-30-33.jpg'
path_r = 'G:/allLabs/обробка зображень/lab3/photo_2026-03-09_16-30-26.jpg'

img_left = imread_unicode(path_l)
img_right = imread_unicode(path_r)

if img_left is None or img_right is None:
    print(" Файли не знайдено!")
    exit()

img_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_l, des_l = sift.detectAndCompute(gray_l, None)
kp_r, des_r = sift.detectAndCompute(gray_r, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des_l, des_r, k=2) 
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

if len(good_matches) > 4:
    src_pts = np.float32([kp_l[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_r[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h_l, w_l = img_l.shape[:2]
    h_r, w_r = img_r.shape[:2]
    
    corners_l = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
    corners_l_transformed = cv2.perspectiveTransform(corners_l, H)
    corners_r = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
    
    all_corners = np.concatenate((corners_l_transformed, corners_r), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    w_pano = x_max - x_min
    h_pano = y_max - y_min
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    pano_l = cv2.warpPerspective(img_l, H_translation.dot(H), (w_pano, h_pano))
    pano_r = np.zeros((h_pano, w_pano, 3), dtype=np.uint8)
    pano_r[translation_dist[1]:translation_dist[1]+h_r, 
           translation_dist[0]:translation_dist[0]+w_r] = img_r

    mask_l_img = cv2.warpPerspective(np.ones((h_l, w_l), dtype=np.uint8)*255, H_translation.dot(H), (w_pano, h_pano))
    mask_l = mask_l_img > 0
    mask_r = np.zeros((h_pano, w_pano), dtype=bool)
    mask_r[translation_dist[1]:translation_dist[1]+h_r, translation_dist[0]:translation_dist[0]+w_r] = True

    overlap = mask_l & mask_r
    coords = np.where(overlap)
    if len(coords[1]) > 0:
        mid_x = (np.min(coords[1]) + np.max(coords[1])) // 2
    else:
        mid_x = w_pano // 2

    panorama = np.zeros_like(pano_l)
    _, x_idx = np.indices((h_pano, w_pano))
    
    left_side = x_idx < mid_x
    right_side = x_idx >= mid_x

    panorama[left_side & mask_l] = pano_l[left_side & mask_l]
    panorama[left_side & (~mask_l) & mask_r] = pano_r[left_side & (~mask_l) & mask_r]
    
    panorama[right_side & mask_r] = pano_r[right_side & mask_r]
    panorama[right_side & (~mask_r) & mask_l] = pano_l[right_side & (~mask_r) & mask_l]

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    img_matches = cv2.drawMatches(img_l, kp_l, img_r, kp_r, good_matches[:30], None, flags=2)
    plt.imshow(img_matches)
    plt.title(f"Знайдено збігів: {len(good_matches)}")
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(panorama)
    plt.title("Результат зшивання (розумний шов по центру)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Недостатньо спільних точок для зшивання!")