from os import listdir, path
import json
from scipy import io
import numpy as np
import h5py
from pathlib import Path

class Evaluation:
    def __init__(self, frames_path, summ_path, gt_path, model, method):
        self.summ_path = summ_path
        self.frames_path = frames_path
        self.gt_path = gt_path
        self.model = model 
        self.method = method
    
    def evaluate(self, video):
        with open(self.summ_path + '/' + video + '/' + self.method + '.txt') as f:
            results = [int(x[:x.rfind(".")]) for x in f] # getting de frames selected
            PACKAGEDIR = Path(__file__).parent.absolute()
            concat = 'mat-files/' + video + '.mat'
            gt_file = PACKAGEDIR / concat
            gt_data = io.loadmat(gt_file) 
            video_duration = gt_data.get('video_duration')[0][0] # getting video duration
            user_summary, n_frames, shotbound = self.get_h5_info(video, PACKAGEDIR) # getting user summary and n frames
            if n_frames == 0: 
                n_frames = int(gt_data.get('nFrames')[0][0])
            
            # sb_summary = self.find_corresponding_supersesg(n_frames, video_duration, results, shotbound)
            sb_summary = self.find_corresponding_frames(n_frames, video_duration, results)
            
            f_score = self.f_score(sb_summary, user_summary, 'max')
            mean_cusa, mean_cuse, cov_value = self.eval_methods(sb_summary, user_summary)
            print(f'{video} Fscore: {f_score}')
            print(f'{video} CUSa: {mean_cusa}')
            print(f'{video} CUSe: {mean_cuse}')
            print(f'{video} Cov: {cov_value}')
            return f_score, mean_cusa, mean_cuse, cov_value

    def get_h5_info(self, video_name, PACKAGEDIR): 
        dataset_path = PACKAGEDIR / 'eccv16_dataset_summe_google_pool5.h5'
        remove_ = ['Saving_dolphines',
                    'Bike_Polo', 
                    'Fire_Domino', 
                    'Eiffel_Tower', 
                    'Statue_of_Liberty', 
                    'Excavators_river_crossing', 
                    'St_Maarten_Landing', 
                    'Base_jumping']
        if video_name in remove_: 
            video_name = video_name.replace('_', ' ')
    
        with h5py.File(dataset_path, 'r') as hdf:
            for video_index in hdf:
                h5_video_name = str(np.array(hdf.get(video_index + '/video_name')))[2:-1]
                if video_name == h5_video_name:
                    user_summary = np.array(hdf.get(video_index + '/user_summary')) 
                    n_frames = np.array(hdf.get(video_index + '/n_frames'))
                    sb = np.array(hdf.get(video_index + '/change_points'))
                    return user_summary, n_frames, sb
            return [], 0, []

    def find_corresponding_frames(self, n_frames, video_duration, results): 
        frames_per_sec = round(n_frames / video_duration)
        equivalent_array = np.arange(3, 4 * round(video_duration), 4)

        try: 
            original_shots = np.arange(0, n_frames, frames_per_sec)
        except: 
            print('n_frames or video_duration is missing assuming 25 frames per sec')
            frames_per_sec = 25 
            original_shots = np.arange(0, n_frames, frames_per_sec)

        if original_shots[-1] != n_frames:
            original_shots = np.concatenate([original_shots, [n_frames]])

        final_shot = n_frames
        summary = np.zeros(final_shot + 1, dtype=np.int8)
        results.sort()

        # print(f'selected frames -> {results}')
        # print(f'equivalent array -> {equivalent_array}')
        # print(f'original shots -> {original_shots}')

        for i in range(len(results)):
            current_frame = results[i]
            for j in range(0,4):
                nxt_frame = current_frame + j
                if np.isin(equivalent_array, nxt_frame).any():
                    index = np.where(equivalent_array == nxt_frame)[0][0]
                    break
            range_ = original_shots[index:index+1][0]
            summary[range_:range_+25] = 1
        return summary 

    def find_corresponding_supersesg(self, n_frames, video_duration, results, shotbound): 
        frames_per_sec = round(n_frames / video_duration)
        equivalent_array = np.arange(3, 4 * round(video_duration), 4)

        try: 
            original_shots = np.arange(0, n_frames, frames_per_sec)
        except: 
            print('n_frames or video_duration is missing assuming 25 frames per sec')
            frames_per_sec = 25 
            original_shots = np.arange(0, n_frames, frames_per_sec)

        if original_shots[-1] != n_frames:
            original_shots = np.concatenate([original_shots, [n_frames]])

        final_shot = n_frames
        summary = np.zeros(final_shot + 1, dtype=np.int8)
        results.sort()

        # print(f'selected frames -> {results}')
        # print(f'equivalent array -> {equivalent_array}')
        # print(f'original shots -> {original_shots}')

        for i in range(len(results)):
            current_frame = results[i]
            for j in range(0,4):
                nxt_frame = current_frame + j
                if np.isin(equivalent_array, nxt_frame).any():
                    index = np.where(equivalent_array == nxt_frame)[0][0]
                    break
            range_ = original_shots[index:index+1][0]

            for sf in shotbound:
                if range_ >= sf[0] and range_ <= sf[1]:
                    summary[sf[0]:sf[1]] = 1
                    break
        return summary 

    def f_score(self, predicted_summary, user_summary, eval_method):
        max_len = max(len(predicted_summary), user_summary.shape[1])
        S = np.zeros(max_len, dtype=int)
        G = np.zeros(max_len, dtype=int)
        
        S[:len(predicted_summary)] = predicted_summary

        f_scores = []
        for user in range(user_summary.shape[0]):
            G[:user_summary.shape[1]] = user_summary[user]
            overlapped = S & G
            
            # Compute precision, recall, f-score
            precision = sum(overlapped)/sum(S)
            recall = sum(overlapped)/sum(G)
            if precision+recall == 0:
                f_scores.append(0)
            else:
                f_scores.append(2 * precision * recall * 100 / (precision + recall))

        if eval_method == 'max':
            return max(f_scores)
        else:
            return sum(f_scores)/len(f_scores)
    
    def eval_methods(self, predicted_summary, user_summary):
        max_len = max(len(predicted_summary), user_summary.shape[1])
        S = np.zeros(max_len, dtype=int)
        G = np.zeros(max_len, dtype=int)
        total_user_frames = np.zeros(max_len, dtype=int)

        S[:len(predicted_summary)] = predicted_summary

        cusa_values = []
        cuse_values = []
        for user in range(user_summary.shape[0]):
            G[:user_summary.shape[1]] = user_summary[user]
            overlapped = S & G
            current_user_frame = total_user_frames
            total_user_frames = current_user_frame | G

            cusa_sum = sum(overlapped)
            cuse_sum = sum(S) - sum(overlapped)
            user_sum = sum(G)
            print('--------------------------------')
            print(f'USER {user}')
            print(f'quantidade de frames escolhidos errado: {cuse_sum}')
            print(f'quantidade de frames escolhidos pelo usuario: {user_sum}')
            print(f'quantidade de frames acertados: {cusa_sum}')
            print(f'total de frames escolhidos pelo hieta: {sum(S)}')
            cusa_value = cusa_sum / user_sum 
            cuse_value = cuse_sum / user_sum 

            cusa_values.append(cusa_value)
            cuse_values.append(cuse_value)
        
        overlapped_cov = S & total_user_frames
        cov_value = sum(overlapped_cov) / sum(total_user_frames)
        mean_cusa = sum(cusa_values) / len(cusa_values)
        mean_cuse = sum(cuse_values) / len(cuse_values)

        return mean_cusa, mean_cuse, cov_value