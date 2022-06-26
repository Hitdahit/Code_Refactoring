import os
import configs
import sys

class setting():
    def __init__(self, set_dict):
        self.settings = set_dict
        self.parse()
        
    def parse(self):
        for i in self.settings.keys():
            
            
            if type(self.settings[i]) is str:
                setattr(self, i, self.settings[i])
        
            elif type(self.settings[i]) is configs.ConfigDict:
                print(i)
                
                values = list(self.settings[i].values())
                
                family = values[0]
                lib = values[1]
                attr = values[2]
                param = tuple(values[3:])
                
                if 'model' in lib:
                    setattr(self, i, getattr(sys.modules[lib], attr))
                    print(self.model)
                    
                    self.net = getattr(self.model, *param)()
                elif 'optim' in lib:
                    setattr(self, i, getattr(sys.modules[lib], attr)(self.model.parameters(), *param))
                    
                elif 'scheduler' in lib:
                    setattr(self, i, getattr(sys.modules[lib], attr)(self.optimizer, *param))
                    
                else:    
                    setattr(self, i, getattr(sys.modules[lib], attr)(*param))
        
            elif type(self.settings[i])is list:
                setattr(self, i, None)