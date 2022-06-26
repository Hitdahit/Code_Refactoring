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
    def __init__(self, task_type, label_type):
        '''
        task_type: BC, MC, ML
        label_type: one-hot(float), ordinal(int)
        '''
        self.task_type = task_type
        self.label_type = label_type
    
    
    
class label_from_dir(base_labeler):
    def __init___(self, task_type, label_type):
        super.__init___(task_type, label_type)
        pass
    
class label_from_path(base_labeler):
    def __init___(self, task_type, label_type):
        super.__init___(task_type, label_type)
        pass
    
class label_from_json(base_labeler):
    def __init___(self, task_type, label_type):
        super.__init___(task_type, label_type)
        pass
    
class label_from_df(base_labeler):
    def __init___(self, task_type, label_type):
        super.__init___(task_type, label_type)
        pass