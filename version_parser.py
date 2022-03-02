import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
import albumentations.pytorch
import models

def get_version_text(version, path):
    
    # Read experiment files
    with open(path, 'r') as file:
        lines = file.readlines()

    # Get 'ver' location to split each version
    versions = []
    for idx, i in enumerate(lines):
        if '#' in i:
            continue
        if 'ver' in i:
            versions.append(idx)
    
    # Get selected version block and append to selected_version_list
    selected_version_list=[]
    for idx, i in enumerate(versions):        
        # if version matched
        if int(lines[i].strip().split(' ')[-1]) == version:  
            # set the version block's end point
            if idx != len(versions)-1:   
                end = versions[idx+1]
            else:
                end = len(lines)-1
    
            for j in range(i, end):
                selected_version_list.append(lines[j].strip())    
                
            # no more iterations are required
            break
    
    result = []
    for i in selected_version_list:
        if '#' in i:
            continue
        if '=' in i:
            result.append(i)

    return result

class Version_Dictionary:          
    def __init__(self, version, selected_version_list_to_dictionary):
        keys=[]
        values=[]
        for i in selected_version_list_to_dictionary:
            keys.append(self._key_value_extractor_key(i))
            values.append(self._key_value_extractor_value(i))
        dic = dict(zip(keys, values))
        
        self.ver = version
        self.GPU_number = dic['GPU_number']
        self.GPU_id = dic['GPU_id']
        self.num_worker = dic['num_worker']
        
        self.seed = dic['seed']
        self.batch_size = dic['batch_size']
        self.img_size = dic['img_size']
        self.epochs = dic['epochs']
        self.learning_rate = dic['learning_rate']
        
        self.modality = dic['modality']
        self.dimension = dic['dimension']
        
        self.checkpoint_dir = dic['checkpoint_dir']
        self.logs_dir = dic['logs_dir']
        self.data_dir = dic['data_dir']
        self.shuffle = dic['shuffle']
        self.drop_last = dic['drop_last']
        
        self.model = dic['model']
        self.model_size = dic['model_size']
        self.n_classes = dic['n_classes']
        self.task_type = dic['task_type']
        self.loss = dic['loss']
        
        self.optimizer = dic['optimizer']
        self.optimizer_params = dic['optimizer_params']
        
        self.scheduler = dic['scheduler']
        self.scheduler_params = dic['scheduler_params']
        
        self.augmentation = None
        self.augmentation_kind = dic['augmentation_kind']
        self.augmentation_settings = dic['augmentation_settings']
        
    def _key_value_extractor_key(self, list_name):
        key = list_name.replace(" ","").split('=')[0]   #replace:공백제거, split('=', 기준분류)
        return key
    def _key_value_extractor_value(self, list_name):
        value = list_name.replace(" ","").split('=')[-1]
        return value
    def _long_param_parser(self, string, additional=None):
        tmp = []
        if additional is not None:
            tmp.append(additional)
        for i in string.split(','):
            if '.' in i or 'e-' in i:
                tmp.append(float(i))
            elif 'True' in i or 'False' in i:
                tmp.append(i=='True')
            else:
                tmp.append(int(i))
        result = tuple(tmp)
        
        return result
    def _aug_parser(self, string):
        setting_split = string.split('/')
        tmp = []
        for i in setting_split:
            x = i.split(',')
            tmp_set = []
            for j in x:
                if '.' in j or 'e-' in j:
                    tmp_set.append(float(j))
                elif 'True' in j or 'False' in j:
                    tmp_set.append(j == 'True')
                elif 'None' in j:
                    tmp_set.append(None)
                else:
                    tmp_set.append(int(j))
            tmp.append(tuple(tmp_set))
        result = tuple(tmp)
        return result
    
    def set_value(self):  # with getattr (str2attribute), locals, globals (str2namespace function)funciton
        os.environ["CUDA_VISIBLE_DEVICES"]=self.GPU_number
        self.n_classes = int(self.n_classes)
        self.device = torch.device(f'cuda:{int(self.GPU_id)}' if torch.cuda.is_available() else 'cpu') # CPU/GPU에서 돌아갈지 정하는 device flag
        self.seed = int(self.seed)
        self.batch_size = int(self.batch_size)
        self.img_size = (int(self.img_size),int(self.img_size))
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)
        self.num_worker  = int(self.num_worker)

        '''
        eps (float) -> for Adam family

        lr_decay (float), weight_decay (float), initial_accumulator_value (int) -> for Adagrad

        betas (tuple, float), weight_decay (float) -> for Adam

        dampening (int) -> for SGD

        '''
        self.model = getattr(models, self.model)(self.n_classes, int(self.model_size))
        self.loss = getattr(nn, self.loss)

        '''
        반드시 모든 파라미터들에 대한 값을 지정하게 해야할 것인가?
        일단은 txt에 모든 파라미터 값을 지정해준다는 가정 (단 한줄에 다 때려박아서)하에 작성.
            init에서 tuple로 만들어 줬다고 쳤을 때
        만약 아니라면 파라미터값들을 하나씩 받게 하고 tuple로 만들어주는게 좋을 듯
        ex. 
            self.lr_decay = float(self.lr_decay) if self.lr_decay is not None else None
            self.weight_decay = float(self.weight_decay) if self.weight_decay is not None else None
            self.initial_accumulator_value = int(self.accumulator_value) if self.accumulator_value is not None else None
            self.eps = float(self.eps) if self.eps is not None else None
            self.betas = (float(self.betas1), float(self.betas2)) if self.betas1 is not None else None
            self.dampening = int(self.dampening) if self.dampening is not None else None

        '''
        # optimizer TODO
        '''
        CLASS: torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad

        CLASS torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

        CLASS torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW

        CLASS torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        if want to optimize two different network with one optimizer, then use itertools.chain().
        '''
        
        self.optimizer_params = self._long_param_parser(self.optimizer_params, self.model.parameters())
        self.optimizer = getattr(optim, self.optimizer)(*self.optimizer_params)

        # scheduler TODO scheduler도 일단 optimizer와 같은 상황이라는 가정하에 작성.
        '''
        CLASS torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=- 1, verbose=False)
        
        CLASS torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.3333333333333333, total_iters=5, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        '''
        self.scheduler_params = self._long_param_parser(self.scheduler_params, self.optimizer)
        self.scheduler = getattr(lr_scheduler, self.scheduler)(*self.scheduler_params)

        # augmentation TODO
        '''
        일단 augmentation은 getattr로 사용한다 해도 그 세팅값 전달을 어케할지가 고민..
        일단 2key dictionary로 -> aug 종류 ('kind')(type:strings), 세팅 값('settings')(type:tuple)
        이러면 아까와 또 마찬가지로 augmentation의 모든 인자값에 대한 parameter 를 txt에서 다 적어야함.
        '''
        aug_list = []
        self.augmentation_kind = self.augmentation_kind.split(',')
        self.augmentation_settings = self._aug_parser(self.augmentation_settings)
        for idx, i in enumerate(self.augmentation_kind[:-1]):
            aug_list.append(getattr(A, i)(*self.augmentation_settings[idx]))
        aug_list.append(getattr(A.pytorch, self.augmentation_kind[-1]))
        self.augmentations = A.Compose(aug_list)

        self.modality = self.modality   # TODO
        self.dimension = int(self.dimension) #TODO

        self.ckpt_dir = self.checkpoint_dir
        self.logs_dir = self.logs_dir
        self.data_dir = self.data_dir
        self.shuffle = self.shuffle == True
        self.drop_last = self.drop_last==True

        self.task_type = self.task_type
        self.epochs = self.epochs
 