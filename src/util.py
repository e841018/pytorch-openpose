import numpy as np, cv2

# 18 keypoints, zero-indexed
keypoint_colors = [
    [255,   0,   0],                                    # 0: nose
    [255,  85,   0],                                    # 1: shoulder center, mean of 2 and 5
    [255, 170,   0], [255, 255,   0], [170, 255,   0],  # 2-4: right arm
    [ 85, 255,   0], [  0, 255,   0], [  0, 255,  85],  # 5-7: left arm
    [  0, 255, 170], [  0, 255, 255], [  0, 170, 255],  # 8-10: right leg
    [  0,  85, 255], [  0,   0, 255], [ 85,   0, 255],  # 11-13: left leg
    [170,   0, 255], [255,   0, 255],                   # 14-15: right/left eyes
    [255,   0, 170], [255,   0,  85],                   # 16-17: right/left ears
]
n_keypoint = len(keypoint_colors)

# limb index to keypoint index, zero-indexed
l2k = np.array([
                        [ 1,  2],   [ 1,  5],                       # right/left shoulders
              [ 2,  3], [ 3,  4],   [ 5,  6], [ 6,  7],             # right/left arms
    [ 1,  8], [ 8,  9], [ 9, 10],   [ 1, 11], [11, 12], [12, 13],   # right/left legs
                              [ 1,  0],                             # neck
              [ 0, 14], [14, 16],   [ 0, 15], [15, 17],             # right/left face
                        [ 2, 16],   [ 5, 17]                        # right/left shoulder-ear connections
])
n_limb = len(l2k)

# limb index to heatmap index, zero-indexed
l2m = [
                        [12, 13],   [20, 21],                       # right/left shoulders
              [14, 15], [16, 17],   [22, 23], [24, 25],             # right/left arms
    [ 0,  1], [ 2,  3], [ 4,  5],   [ 6,  7], [ 8,  9], [10, 11],   # right/left legs
                              [28, 29],                             # neck
              [30, 31], [34, 35],   [32, 33], [36, 37],             # right/left face
                        [18, 19],   [26, 27],                       # right/left shoulder-ear connections
]

# draw the body keypoints and limbs
def draw_bodypose(canvas, peak_list, persons):
    assert canvas.shape[2] == 3
    canvas = canvas[:, :, ::-1] # RGB -> BGR
    persons = persons[:, :n_keypoint].astype(int) # using only the indices

    # draw limbs, using keypoint_colors[:n_limb-2]
    canvas_tmp = canvas.copy()
    for person in persons:
        for l in range(n_limb-2): # not drawing the last 2 limbs
            color = keypoint_colors[l][::-1] # BGR
            k0, k1 = l2k[l]
            index0, index1 = person[k0], person[k1]
            if index0 != -1 and index1 != -1:
                x0, y0 = peak_list[index0, :2]
                x1, y1 = peak_list[index1, :2]
                xm, ym = int((x0+x1)/2), int((y0+y1)/2)
                xd, yd = x0-x1, y0-y1
                length = int((xd ** 2 + yd ** 2) ** 0.5 / 2)
                angle = int(np.degrees(np.arctan2(yd, xd)))
                polygon = cv2.ellipse2Poly((xm, ym), (length, 4), angle, 0, 360, 1)
                cv2.fillConvexPoly(canvas_tmp, polygon, color)
    canvas = cv2.addWeighted(canvas, 0.4, canvas_tmp, 0.6, 0)

    # draw keypoints
    canvas_tmp = canvas.copy()
    for person in persons:
        for k in range(n_keypoint):
            color = keypoint_colors[k][::-1] # BGR
            index = person[k]
            if index != -1:
                x, y = peak_list[index][0:2].astype(int)
                cv2.circle(canvas_tmp, (x, y), 4, color, thickness=-1)
    canvas = cv2.addWeighted(canvas, 0.4, canvas_tmp, 0.6, 0)

    canvas = canvas[:, :, ::-1] # BGR -> RGB
    return canvas
