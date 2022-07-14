import os
import json
import pandas as pd
import numpy as np

'''
Labeler
    #label from directory?
    #label from path?
    #label from json / txt / excel / etc... ?
    #\
    #label type
    #one-hot / ordinal / regression
    # per-patient / per-image label (CT, MR)
    
    #MC label / ML label / BC label
'''
class base_labeler():
    def __init__(self, file_name, task_type, label_type, label_name, label_source):
        '''
        task_type: (str) BC, MC, ML
        label_type: (str) one-hot(float), ordinal(int)
        label_name: (list: str) ['label1', 'label2', ...]
                    ex) ['Normal', 'Disease']
        label_source: (str) label_from_path, label_from_json, label_from_df
        '''
        self.task_type = task_type
        self.label_type = label_type
        
        self.label_name = label_name
        self.label_source = label_source
        
        self.n_classes = len(self.label_name)
    
        if self.label_type =='one-hot':
            self.label_type = np.eye(self.n_classes)    
        else:
            self.label_type = np.arange(self.n_classes)
            
    def _label_from_path(self, path):
        r = [self.label_type[idx] for idx, l in enumerate(self.label_name) if l in path]
            
    def _label_from_json(self):
        pass
    def _label_from_df(self):
        pass