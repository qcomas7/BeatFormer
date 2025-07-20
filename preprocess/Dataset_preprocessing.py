import json
import pandas as pd
import glob
from preprocess.landmarks_extraction import *
from scipy import signal
import re
import os
import itertools
from dataloading.utils_data import load_video


def low_pass_filter(x, fs, fc):
    fc = fc
    w = fc / (fs / 2)
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, x)
    return output


def convert_timestamp(current_timestamp, first_timestamp):
    return float(current_timestamp - first_timestamp) * 1e-9


def load_signal_file(filename):
    bvp = []
    hr = []
    timestamp = []
    with open(filename) as json_file:
        json_data = json.load(json_file)
        for p in json_data['/FullPackage']:
            timestamp.append(p['Timestamp'])
            bvp.append(p['Value']['waveform'])
            hr.append(p['Value']['pulseRate'])
    bvp = np.array(bvp, dtype=np.float32).reshape([-1])
    hr = np.array(hr, dtype=np.float32).reshape([-1])
    return bvp, hr, timestamp


def PURE_PPG_preprocessing(dataset_path, dest_dir):
    json_name_list = [[_ for _ in os.listdir(os.path.join(dataset_path, folder)) if '.json' in _] for folder in os.listdir(dataset_path)]
    json_name_list = list(itertools.chain(*json_name_list))
    json_name_list.sort()

    for user in json_name_list:
        trial = user.split('.')[0]
        user_path = os.path.join(dataset_path, trial, user)
        output_dir = dest_dir + trial
        output_phys_path = output_dir + '/phys.csv'
        video_data = output_dir + '/video_mask.avi'
        ppg_raw_data, hr_raw_data, timestamp_raw_data = load_signal_file(user_path)
        first_timestamp = timestamp_raw_data[0]
        timestamp_phys = [convert_timestamp(time, first_timestamp) for time in timestamp_raw_data]
        end = timestamp_phys[-1]
        video = load_video(video_data, type_color='BGR', img_size=32)
        fs = len(video) / end
        ppg_raw_data = low_pass_filter(ppg_raw_data, fs, 10)
        phys_data = {}
        phys_data['timestamp'] = timestamp_phys
        phys_data['ppg'] = ppg_raw_data
        phys_data['ppg_heart_rate'] = hr_raw_data
        df = pd.DataFrame(phys_data)
        if not os.path.exists(output_phys_path):
            df.to_csv(output_phys_path)
        print(user)


def read_video(video_file):
    frames = list()
    all_png = sorted(glob.glob(video_file + '/*.png'))
    for png_path in all_png:
        img = cv2.imread(png_path)
        frames.append(img)
    return frames


def Facial_preprocessing(path_source, path_out, img_size, fs=30, sample_name='video_mask', method='tracking', mask=True,
                         coef=1.20):
    filelist = os.listdir(path_source)
    filelist.sort(key=lambda f: int(re.sub('\D', '', f)))
    for user in filelist:
        user_path = os.path.join(path_source, user)
        output_video_dir = os.path.join(path_out, user)
        os.makedirs(output_video_dir, exist_ok=True)
        options = {'W': img_size, 'H': img_size, 'sample_path': output_video_dir, 'sample_name': sample_name+'.avi',
                   'sample_rate': fs}
        preprocessing = Video_preprocessing(options)
        video = read_video(user_path+'/' + user)
        pred_landmarks = preprocessing.landmarks_extraction(video)
        preprocessing.ROI_extraction(video, pred_landmarks, method=method, img_size=img_size, color='BGR', mask=mask,
                                     use_larger_box=True, larger_box_coef=coef, flag_write=True)
        del video
        print(output_video_dir)


if __name__ == "__main__":
    path_source = 'path_source'
    path_dest = 'path_dest'
    img_size = 96
    Facial_preprocessing(path_source, path_dest, img_size=img_size, sample_name='video_mask', method='tracking',
                         mask=True, coef=1.20)
    PURE_PPG_preprocessing(path_source, path_dest)