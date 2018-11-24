import torch.utils.data as data
import pdb 
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pickle
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
class Utter_Record(object):
    def __init__(self, row, video, utter, visual_root_path, audio_root_path, text_root_path, label_name):
        self._data = row
        self.video = video
        self.utterance = utter
        self._label_name = label_name
        if not label_name in row.keys():
            print('Wrong label name')
            os.exit()
        self.visual_root_path = visual_root_path
        self.audio_root_path = audio_root_path
        self.text_root_path = text_root_path
    @property
    def vipath(self):
        if os.path.exists( os.path.join(self.visual_root_path, self.video+'_'+self.utterance+'_feature.pkl')):
            return os.path.join(self.visual_root_path, self.video+'_'+self.utterance+'_feature.pkl')
        else:
            return None
    @property
    def aupath(self):
        if os.path.exists( os.path.join(self.audio_root_path, self.video+'_'+self.utterance+'.csv')):
            
            return os.path.join(self.audio_root_path,self.video+'_'+self.utterance+'.csv')
        else:
            return None
    @property
    def tepath(self):
        if os.path.exists( os.path.join(self.text_root_path, self.video+'_'+self.utterance+'.txt')):
            
            return os.path.join(self.text_root_path, self.video+'_'+self.utterance+'.txt')
        else:
            return None
    @property
    def label(self):
        return self._data[self._label_name]
    
class VAT_DataSet(Dataset):
    def __init__(self, visual_root_path, audio_root_path, text_root_path, dict_file , label_name,test_mode=False):
        
        self.visual_root_path = visual_root_path
        self.audio_root_path = audio_root_path
        self.text_root_path = text_root_path
        self.dict_file = dict_file
        self.label_name = label_name
        self.test_mode = test_mode
        
        self._parse_dict()
        
    def _parse_dict(self):
        data_dict = pickle.load(open(self.dict_file,'rb'))
        self.utterance_list = list()
        for video in data_dict.keys():
            for utter in data_dict[video].keys():
                item = data_dict[video][utter]
                ur = Utter_Record(item, video, utter, self.visual_root_path, self.audio_root_path, self.text_root_path, self.label_name)
                if (ur.vipath is not None) and (ur.tepath is not None) and (ur.aupath is not None):
                    self.utterance_list.append(ur)
    def _parse_visual_feature(self, path):
        with open(path ,'rb') as f:
            features = pickle.load(f) # uniformly sampled only face features
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        maximum = np.max(features, axis=0)
        return np.concatenate([mean, std, maximum ], axis=0)
    def _parse_audio_feature(self, path):
        df = pd.read_csv(path, skipinitialspace = True, sep=';')
        features = df[df.keys()[2:]].values
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        maximum = np.max(features, axis=0)
        return np.concatenate([mean, std, maximum ], axis=0)
    def _parse_text_feature(self, path):
        file = open (path,'r')
        feature = file.readline().split(' ')
        feature = [float(f) for f in feature]
        return np.asarray(feature)
    def get(self, record):
        vis_f = self._parse_visual_feature(record.vipath)
        aud_f = self._parse_audio_feature(record.aupath)
        text_f = self._parse_text_feature(record.tepath)
        feature = np.concatenate([vis_f, aud_f, text_f])
        return feature, record.label
    def __getitem__(self, index):
        record = self.utterance_list[index] 
        if not self.test_mode:
            return self.get(record)
    def __len__(self):
        return len(self.utterance_list)

class VAT_video_Dataset(Dataset):
    def __init__(self, visual_root_path, audio_root_path, text_root_path, dict_file , label_name,test_mode=False, feature_selection_from_model = None):
        
        self.visual_root_path = visual_root_path
        self.audio_root_path = audio_root_path
        self.text_root_path = text_root_path
        self.dict_file = dict_file
        self.label_name = label_name
        self.test_mode = test_mode
        self.mask = None
        self._parse_dict()
        self.feature_selection_from_model = feature_selection_from_model
        if self.feature_selection_from_model is not None:
            model = pickle.load(open(feature_selection_from_model, 'rb'))
            self.meta  = model.meta
            self.mask= model.feature_importances_!=0.0
    def _parse_dict(self):
        data_dict = pickle.load(open(self.dict_file,'rb'))
        self.video_list = list()
        self.max_utter_length = 0
        for video in data_dict.keys():
            utterance_list = []
            for utter in sorted(data_dict[video].keys()):
                item = data_dict[video][utter]
                ur = Utter_Record(item, video, utter, self.visual_root_path, self.audio_root_path, self.text_root_path, self.label_name)
                if (ur.vipath is not None) and (ur.tepath is not None) and (ur.aupath is not None):
                    utterance_list.append(ur)
                if len(utterance_list) > self.max_utter_length:
                    self.max_utter_length = len(utterance_list)
            self.video_list.append(utterance_list)
        self.video_list = sorted(self.video_list, key=lambda x:len(x), reverse=True)
    def _parse_visual_feature(self, path):
        with open(path ,'rb') as f:
            features = pickle.load(f) # uniformly sampled only face features
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        maximum = np.max(features, axis=0)
        return np.concatenate([mean, std, maximum ], axis=0)
    def _parse_audio_feature(self, path):
        df = pd.read_csv(path, skipinitialspace = True, sep=';')
        features = df[df.keys()[2:]].values
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        maximum = np.max(features, axis=0)
        return np.concatenate([mean, std, maximum ], axis=0)
    def _parse_text_feature(self, path):
        file = open (path,'r')
        feature = file.readline().split(' ')
        feature = [float(f) for f in feature]
        return np.asarray(feature)
    def get(self, record):
        vis_f = self._parse_visual_feature(record.vipath)
        aud_f = self._parse_audio_feature(record.aupath)
        text_f = self._parse_text_feature(record.tepath)
        feature = np.concatenate([vis_f, aud_f, text_f])
        # normalize
        if self.feature_selection_from_model is not None:
            mean = self.meta['mean']
            std = self.meta['std']
            assert self.meta['input_size'] == feature.shape
            feature = (feature - mean)/std
        if self.mask is not None:
            feature = feature[self.mask]
        return feature, record.label
    def __len__(self):
        return len(self.video_list)
    def __getitem__(self, index):
        record_list = self.video_list[index] 
        if not self.test_mode:
            data_list= []
            label_list= []
            for record in record_list:
                feature, label = self.get(record)
                n_feature = feature.shape[0]
                data_list.append(feature)
                label_list.append(label)
            original_length = len(data_list)
            # padding to self.max_utter_length
            for _ in range(self.max_utter_length - len(data_list)):
                data_list.append(np.zeros(n_feature))
                label_list.append(100.0) # dummy label
            return np.asarray(data_list), np.asarray(label_list), original_length
    
if __name__ == "__main__":
    visual_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/vgg_fer_features_fps=15_fc7'
    audio_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/egemaps_VAD'
    text_root_path = '/newdisk/OMGEmotionChallenge/fur_experiment/transfer_learning/Extracted_features/MPQA' 
    
    #dataset = VAT_DataSet(visual_root_path, audio_root_path, text_root_path, '../train_dict.pkl',  label_name='arousal')
    dataset = VAT_video_Dataset(visual_root_path, audio_root_path, text_root_path, '../train_dict.pkl',  
                                label_name='arousal', feature_selection_from_model= 'xgb_ccc:0.2234_label:arousal.pkl')
    sample = dataset[0]
    sample = dataset[10]
    dl = DataLoader(dataset=dataset, batch_size=3) 
    print('Total samples', len(dl))
    
    it = iter(dl)
    xs,ys,lens =  next(it)
    print(type(xs))
    print(xs)
        