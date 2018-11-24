from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
import numpy as np
import pickle
class Record(object):
    def __init__(self, row, video, utter, audio_root_path, label_name):
        self._data = row
        self.video = video

        self._label_name = label_name
        if not label_name in row.keys():
            print('Wrong label name')
            os.exit()
        self.audio_root_path = audio_root_path
    @property
    def aupath(self):
        chunks_paths = glob.glob(os.path.join(self.audio_root_path, self.video+'_'+self.utterance+'*'))
        if len(chunks_paths)!=0:            
            return chunks_paths
        else:
            return None
    @property
    def label(self):
        return self._data[self._label_name]
def clean_data(data):
    length = len(data)
    new_data = np.zeros(length)
    for i, item in enumerate(data):
        new_data[i] = float(item)
    return new_data    
def read_arff(input_arff):
    attributes = list()
    file=open(input_arff,'r')
    data = None
    while True:
        line = file.readline()
        if line:
            if line.startswith("@attribute"):
                attribute = line.split(' ')[1]
                attributes.append(attribute)
            if line.startswith('@data'):
                line = file.readline()# skip one line
                line = file.readline()
                if line: 
                    data = clean_data(line)
                    
                break
        else:
            break
    if data is not None:
        d = dict((attr, value) for (attr, value) in zip(attributes, data))
        data = pd.DataFrame.from_dict(d)
    return data
class emolarge_dataset(Dataset):
    def __init__(self, audio_root_path, dict_file, label_name,test_mode=False):
        self.audio_root_path = audio_root_path
        self.label_name = label_name
        self.test_mode = test_mode
        
        self._parse_dict()
        # get attributes frist
        _, self.attributes = self._parse_audio_feature(self.utterance_list[0])
    def _parse_dict(self):
        data_dict = pickle.load(open(self.dict_file,'rb'))
        self.utterance_list = list()
        for video in data_dict.keys():
            for utter in data_dict[video].keys():
                item = data_dict[video][utter]
                ur = Record(item, video, utter, self.audio_root_path, self.label_name)
                if (ur.aupath is not None):
                    self.utterance_list.append(ur)
    def _parse_audio_feature(self, arff_path_list):
        mean_values = []
        for arff_path in arff_path_list:
            data = read_arff(arff_path)
            data = data[data.keys()[:-1]]
            attributes = data.keys()
            values = data.values
            mean_values.append(values)
        mean_values = np.mean(np.asarray(mean_values), axis=0)
        
        return mean_values, attributes
    def get(self, record):
        aud_f, _ = self._parse_audio_feature(record.aupath)
        return aud_f, record.label
    def __getitem__(self, index):
        record = self.utterance_list[index] 
        if not self.test_mode:
            return self.get(record)
    def __len__(self):
        return len(self.utterance_list)
            
        
        