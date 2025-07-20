import cv2
import mediapipe as mp
import copy
import numpy as np
from collections import deque


class Video_preprocessing:
    def __init__(self, options):
        self.options = options

    @staticmethod
    def median_filter(lndmrks, win_len=5, lnd=True):
        windowed_sample = deque()
        sample_count = 0
        temporal_length = win_len
        temporal_stride = 1
        samples_data = []

        for i in range(0, len(lndmrks)):
            windowed_sample.append(lndmrks[i])
            sample_count += 1
            if len(windowed_sample) == temporal_length:
                final_windowed_sample = np.median(np.asarray(list(copy.deepcopy(windowed_sample))), axis=0)
                if lnd:
                    final_windowed_sample = [[int(v) for v in l] for l in final_windowed_sample]
                for t in range(temporal_stride):
                    windowed_sample.popleft()
                samples_data.append(final_windowed_sample)

        final_landmarks = [*lndmrks[0:int(win_len // 2)], *np.asarray(samples_data[:]), *lndmrks[-int(win_len // 2):]]
        return final_landmarks

    @staticmethod
    def merge_add_mask(img_1, mask):
        assert mask is not None
        mask = mask.astype('uint8')
        mask = mask * 255
        b_channel, g_channel, r_channel = cv2.split(img_1)
        r_channel = cv2.bitwise_and(r_channel, r_channel, mask=mask)
        g_channel = cv2.bitwise_and(g_channel, g_channel, mask=mask)
        b_channel = cv2.bitwise_and(b_channel, b_channel, mask=mask)
        res_img = cv2.merge((b_channel, g_channel, r_channel))
        return res_img

    @staticmethod
    def landmarks_transform(image, landmark_list, region):
        lndmrks = []
        lnd_list = np.asarray(landmark_list)[region]
        for landmark in lnd_list:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            shape = image.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            relative_z = int(z * shape[0])
            lndmrks.append([relative_x, relative_y])
        return np.asarray(lndmrks)

    @staticmethod
    def poly2mask(landmarks, img_shape, val=1, b_val=0):
        if b_val == 0:
            hull_mask = np.zeros(img_shape[0:2] + (1,), dtype=np.float32)
        else:
            hull_mask = np.ones(img_shape[0:2] + (1,), dtype=np.float32)
        cv2.fillPoly(hull_mask, [landmarks], (val,))
        return hull_mask

    def landmarks_extraction(self, video):

        mp_face_mesh = mp.solutions.face_mesh
        lnds_video = []
        lnd_video_z = []
        prev_face_landmarks = None
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5) as face_mesh:
            for frame in video:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results = face_mesh.process(frame)
                try:
                    face_landmarks = results.multi_face_landmarks[0]
                except:
                    face_landmarks = prev_face_landmarks
                    if not face_landmarks:
                        results = face_mesh.process(video[30])
                        face_landmarks = results.multi_face_landmarks[0]

                pred_landmarks = face_landmarks.landmark
                pred_landmarks, z_pos = self.landmarks_transform(frame, pred_landmarks, range(0, 468))
                lnds_video.append(pred_landmarks)
                lnd_video_z.append(z_pos)
                prev_face_landmarks = face_landmarks
        return lnds_video, lnd_video_z

    def ROI_extraction(self, video, pred_landmarks, method, img_size, color='BGR', mask=True, use_larger_box=True,
                       larger_box_coef=1.5, flag_write=False):

        leftEyeUpper = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]  # 463
        rightEyeUpper = [243, 173, 246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133]
        face_countours = [54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
                          365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103]
        face_in = [139, 21, 143, 116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 199, 428, 262, 431, 430, 434, 416,
                   433, 376, 352, 345, 372, 368, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54]
        face_right = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 151,
                      9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 37, 39, 40, 185, 57, 146, 91, 181, 84, 17, 18,
                      200, 199, 175]
        face_left = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 151,
                     9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 267, 269, 270, 409, 287, 375, 321, 405, 314,
                     17, 18, 200, 199, 175]
        lips = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306, 307, 320, 404, 315, 16, 85, 180, 90, 96, 62]

        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        crop_res = (self.options['W'], self.options['H'])
        writer = cv2.VideoWriter(self.options['sample_path'] + '/' + self.options['sample_name'], fourcc,
                                 self.options['sample_rate'], crop_res)
        if method == 'static':
            x = np.min(np.min(pred_landmarks, axis=0), axis=0)[0]
            y = np.min(np.min(pred_landmarks, axis=0), axis=0)[1]
            width = np.max(np.max(pred_landmarks, axis=0), axis=0)[0] - np.min(np.min(pred_landmarks, axis=0), axis=0)[
                0]
            height = np.max(np.max(pred_landmarks, axis=0), axis=0)[1] - np.min(np.min(pred_landmarks, axis=0), axis=0)[
                1]
            center_x = x + width // 2
            center_y = y + height // 2
            square_size = max(width, height)
            new_x = center_x - (square_size // 2)
            new_y = center_y - (square_size // 2)
            new_width = square_size
            new_height = square_size

            if use_larger_box:
                new_x = int(max(0, new_x - (larger_box_coef - 1.0) / 2 * new_width))
                new_y = int(max(0, new_y - (larger_box_coef - 1.0) / 2 * new_height))
                new_width = int(larger_box_coef * new_width)
                new_height = int(larger_box_coef * new_height)
            video_crop = [frm[new_y:new_y + new_height, new_x:new_x + new_width, :] for frm in video]

        elif method == 'tracking':
            pred_landmarks = np.asarray(self.median_filter(pred_landmarks, 5))
            video_crop = []
            x_0, y_0, width, height = cv2.boundingRect(pred_landmarks[0])
            square_size = max(width, height)
            mov_video = np.sum(np.abs(np.diff(np.asarray([pred[4][0] for pred in pred_landmarks])))) / len(
                pred_landmarks)

            for i in range(0, len(pred_landmarks)):
                try:
                    x, y, _, _ = cv2.boundingRect(pred_landmarks[i])
                except:
                    print("ERROR: No Face Detected")
                new_center_x = x + width // 2
                new_center_y = y + height // 2
                new_x = max(0, new_center_x - (square_size // 2))
                new_y = max(0, new_center_y - (square_size // 2))
                if mov_video <= 0.09:
                    new_center_x = x_0 + width // 2
                    new_center_y = y_0 + height // 2
                    new_x = max(0, new_center_x - (square_size // 2))
                    new_y = max(0, new_center_y - (square_size // 2))
                new_width = square_size
                new_height = square_size
                if use_larger_box:
                    new_x = int(max(0, new_x - (larger_box_coef - 1.0) / 2 * new_width))
                    new_y = int(max(0, new_y - (larger_box_coef - 1.0) / 2 * new_height))
                    new_width = int(larger_box_coef * new_width)
                    new_height = int(larger_box_coef * new_height)
                    if (new_x + new_width) > video[0].shape[1]:
                        new_x = max(0, new_x - ((new_x + new_width) - video[0].shape[1]))
                    if (new_y + new_height) > video[0].shape[0]:
                        new_y = max(0, new_y - ((new_y + new_height) - video[0].shape[0]))
                video_crop.append(video[i][new_y:new_y + new_height, new_x:new_x + new_width, :])
        if mask:
            new_pred_landmarks, _ = self.landmarks_extraction(video_crop)
            face_countour = [pred[face_countours] for pred in new_pred_landmarks]
            face_inside = [pred[face_in] for pred in new_pred_landmarks]
            face_left_c = [pred[face_left] for pred in new_pred_landmarks]
            face_right_c = [pred[face_right] for pred in new_pred_landmarks]
            lips_countour = [pred[lips] for pred in new_pred_landmarks]
            leye_countour = [pred[leftEyeUpper] for pred in new_pred_landmarks]
            reye_countour = [pred[rightEyeUpper] for pred in new_pred_landmarks]
        for i, frame in enumerate(video_crop):
            if mask:
                face_mask = Video_preprocessing.poly2mask(face_countour[i], frame.shape, val=1, b_val=0)
                face_in_mask = self.poly2mask(face_inside[i], frame.shape, val=1, b_val=0)
                face_l_mask = self.poly2mask(face_left_c[i], frame.shape, val=1, b_val=0)
                face_r_mask = self.poly2mask(face_right_c[i], frame.shape, val=1, b_val=0)
                lips_mask = Video_preprocessing.poly2mask(lips_countour[i], frame.shape, val=0, b_val=1)
                leye_mask = Video_preprocessing.poly2mask(leye_countour[i], frame.shape, val=0, b_val=1)
                reye_mask = Video_preprocessing.poly2mask(reye_countour[i], frame.shape, val=0, b_val=1)
                face_roi = np.logical_or.reduce([face_mask, face_in_mask, face_l_mask, face_r_mask])
                roi_mask = np.logical_and.reduce([face_roi, lips_mask, leye_mask, reye_mask])
                frame = Video_preprocessing.merge_add_mask(frame, roi_mask)
            if flag_write:
                if color == 'RGB':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(cv2.resize(frame, (img_size, img_size)))
        writer.release()
