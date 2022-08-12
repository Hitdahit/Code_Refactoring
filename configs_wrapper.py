import os
import configs
import sys
import models
from utils.family import runtime, etc, datautil, modelutil

class setting():
    def __init__(self, set_dict):
        self.settings = set_dict
        
    def parse(self):
        for i in self.settings.keys():
            
            
            if type(self.settings[i]) is str or type(self.settings[i]) is int or type(self.settings[i]) is bool:
                setattr(self, i, self.settings[i])
        
            elif type(self.settings[i]) is configs.ConfigDict:
                
                values = list(self.settings[i].values())
                
                family = values[0]
                lib = values[1]
                attr = values[2]
                param = tuple(values[3:])
                
                if lib is None and 'augmentation' in attr:
                    setattr(self, i, values[3])
                    
                elif 'model' in i:
                    setattr(self, i, getattr(sys.modules[lib], attr)(*param))
                    
                elif 'optim' in i:
                    
                    setattr(self, i, getattr(sys.modules[lib], attr)(self.model.parameters(), *param))
                    
                elif 'scheduler' in i:
                    setattr(self, i, getattr(sys.modules[lib], attr)(self.optimizer, *param))
                    
                else:    
                    setattr(self, i, getattr(sys.modules[lib], attr)(*param))
        
            elif type(self.settings[i]) is list:
                item_lst = []
                for item in self.settings[i]:
                    if type(item) is str or type(item) is int or type(item) is float:
                        item_lst.append(item)
                        
                    elif type(item) is configs.ConfigDict:
                        
                        values = list(item.values())
                
                        family = values[0]
                        lib = values[1]
                        attr = values[2]
                        param = tuple(values[3:])
                        

                        item_lst.append(getattr(sys.modules[lib], attr)(*param))
                        
                setattr(self, i, item_lst)
            
            elif type(self.settings[i]) is None:
                setattr(self, i, None)