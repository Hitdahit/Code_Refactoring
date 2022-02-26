import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
import albumentations.pytorch


def get_version_text(version, path):
    # f.close 자동으로 해줌
    with open(path, 'r') as file:
        # 모든 글자들을 하나의 문자열로 읽기
        # a = file.read()
        # 줄별로 리스팅
        lines = file.readlines()

    versions = []   #versions에 version들 차곡차곡 쌓기
    for idx, i in enumerate(lines):
        if 'ver' in i:
            versions.append(idx)

    selected_version_list=[]
    for idx, i in enumerate(versions):
        #print (idx, i)
        #print (lines[i].strip().split(' ')[-1])   #strip():\n과 같은 특수기호 날리기, split(''): 괄호안 기준 쪼개
        if int(lines[i].strip().split(' ')[-1]) == version:  #위에서 선택한 version
            if idx != len(versions)-1:   #중간이면 다음버전 한줄 위에 까지가 end이다.
                end = versions[idx+1]
            else:
                end = len(lines)-1        #마지막 version이면 맨 밑줄이 end이다.
            for j in range(i, end):       #i부터 end까지 list로 append
                selected_version_list.append(lines[j].strip())    
    #list에서 '='표시가 있는 의미 있는 자료들을 dictionary하기 위해서 거름.
    
    result = []
    for i in selected_version_list:
        if '=' in i:
            result.append(i)

    return result

class Version_Dictionary:          #class만들어서 하나하나 넣자  # 날리고,
    def __init__(self, selected_version_list_to_dictionary):
        #self.GPU_number = dic['GPU_num']
        keys=[]
        values=[]
        for i in selected_version_list_to_dictionary:
            keys.append(self._key_value_extractor_key(i))
            values.append(self._key_value_extractor_value(i))
        dictionary = dict(zip(keys, values))
        
        self.ver = version
        self.GPU_number = dictionary['GPU_number']
        self.GPU_id = dictionary['GPU_id']
        self.num_worker = dictionary['num_worker']
        self.seed = dictionary['seed']
        self.batch_size = dictionary['batch_size']
        self.img_size = dictionary['img_size']
        self.epochs = dictionary['epochs']
        self.learning_rate = dictionary['learning_rate']
        self.modality = dictionary['modality']
        self.dimiension = dictionary['dimiension']
        self.is_time_series = dictionary['is_time_series']
        self.checkpoint_dir = dictionary['checkpoint_dir']
        self.logs_dir = dictionary['logs_dir']
        self.augmentation = dictionary['augmentation']
        self.model = dictionary['model']
        self.task_type = dictionary['task_type']
        self.optimizer = dictionary['optimizer']
        self.scheduler = dictionary['scheduler']
        
    def _key_value_extractor_key(self, list_name):
        key = list_name.replace(" ","").split('=')[0]   #replace:공백제거, split('=', 기준분류)
        return key
    def _key_value_extractor_value(self, list_name):
        value = list_name.replace(" ","").split('=')[-1]
        return value

    def set_value(self):
        os.environ["CUDA_VISIBLE_DEVICES"]=self.GPU_number
        self.n_classes = int(self.num_class)
        self.device = torch.device(f'cuda:{int(self.GPU_id)}' if torch.cuda.is_available() else 'cpu') # CPU/GPU에서 돌아갈지 정하는 device flag
        self.seed = int(self.seed)
        self.batch_size = int(self.batch_size)
        self.img_size = int(self.img_size)
        self.epochs = int(self.epochs)
        self.lr = float(self.leraning_rate)

        '''
        eps (float) -> for Adam family

        lr_decay (float), weight_decay (float), initial_accumulator_value (int) -> for Adagrad

        betas (tuple, float), weight_decay (float) -> for Adam

        dampening (int) -> for SGD

        '''
        self.lr_decay = float(self.lr_decay) if self.lr_decay is not None else None
        self.weight_decay = float(self.weight_decay) if self.weight_decay is not None else None
        self.initial_accumulator_value = int(self.accumulator_value) if self.accumulator_value is not None else None
        self.eps = float(self.eps) if self.eps is not None else None
        self.betas = (float(self.betas1), float(self.betas2)) if self.betas1 is not None else None
        self.dampening = int(self.dampening) if self.dampening is not None else None

        self.modality = self.modality   # TODO
        self.dimension = int(self.dimension) #TODO

        self.ckpt_dir = self.checkpoint_dir
        self.log_dir = self.logs_dir
        self.data_dir = self.data_dir

        # augmentation TOCO
        self.task_type = self.task_type


        self.epochs = self.epochs
    
        losses = {'BCE':nn.BCEWithLogitsLoss(),'CE':nn.CrossEntropyLoss(), 'F':}  # 추후 추가
        loss = None
        for key, value in losses.items():
            if key in self.loss:
                loss = value
        self.loss = loss

        models = {'vgg':model.VGG_Classifier(n_classes, int(self.model_size)),\
                'resnet':model.ResNet_Classifier(n_classes, int(self.model_size)),\
                'densenet':model.DenseNet_Classifier(n_classes, int(self.model_size)),\
                'efficientnet': model.EfficientNet_Classifier(n_classes, int(self.model_size))}
        self.model = None
        for key, value in models.items():
            if key in self.model:
                self.model = value

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

        optimizers = {'Adagrad':optim.Adagrad(params = self.model.parameters(), lr=self.lr, lr_decay=self.lr_decay, weight_decay=self.weight_decay, initial_accumulator_value=self.initial_accumulator_value, eps=self.eps),\
                    'Adam':optim.Adam(params = self.model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay), \
                    'AdamW':optim.AdamW(params = self.model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay),\
                    'SGD':optim.SGD(params = self.model.parameters(), lr=self.lr, dampening=self.dampening, weight_decay=self.weight_decay, nesterov=False)}
        optimizer = None
        for key, value in optimizers.items():
            if key in self.optimizer:
                optimizer = value
        self.optimizer = optimizer

        # scheduler TODO
        '''
        CLASS torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=- 1, verbose=False)
        
        CLASS torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.3333333333333333, total_iters=5, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False)

        CLASS torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        '''
        schedulers = {'StepLR': lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=self.gamma, last_epoch=self.last_epoch, verbose=self.verbose),\
                    'MStepLR': lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.milestones, gamma=self.gamma, last_epoch=self.last_epoch, verboas=self.verbose),\
                    'ConstLR': lr_scheduler.ConstantLR(optimizer=self.optimizer, factor=self.factor, total_iters=self.total_iters, last_epoch=self.last_epoch, verbose=self.verboase),\
                    'LinearLR': lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=self.start_factor, end_factor=self.end_factor, total_iters=self.total_iters, last_epoch=self.last_epoch, verbose=self.verbose), \
                    'CosAnnealLR': lr_scheduler.cosineAnnealingLR(optimizer=self.optimizer, T_max=self.T_max, eta_min=self.eta_min, last_epoch=self.last_epoch, verbose=self.verbose), \
                    'ReduceLR': lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode=self.mode, factor=self.factor, patience=self.patience, threshold=self.threshold, threshold_mode=self.threshold_mode, min_lr=self.min_lr, eps=self.eps, verbose=self.verbose)
        }

        
def set_value2(self):  # with getattr (str2attribute), locals, globals (str2namespace function)funciton
        os.environ["CUDA_VISIBLE_DEVICES"]=self.GPU_number
        self.n_classes = int(self.num_class)
        self.device = torch.device(f'cuda:{int(self.GPU_id)}' if torch.cuda.is_available() else 'cpu') # CPU/GPU에서 돌아갈지 정하는 device flag
        self.seed = int(self.seed)
        self.batch_size = int(self.batch_size)
        self.img_size = int(self.img_size)
        self.epochs = int(self.epochs)
        self.lr = float(self.leraning_rate)

        '''
        eps (float) -> for Adam family

        lr_decay (float), weight_decay (float), initial_accumulator_value (int) -> for Adagrad

        betas (tuple, float), weight_decay (float) -> for Adam

        dampening (int) -> for SGD

        '''
        self.model = getattr(model, self.model)(self.n_classes, int(self.model_size))
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
        self.optimizer_params = tuple(list(self.optimizer_params).insert(0, self.model.parameters()))
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
        self.scheduler_params = tuple(list(self.scheduler_params).insert(0, self.optimizer))
        self.scheduler = getattr(lr_scheduler, self.scheduler)(*self.scheduler_params)

        # augmentation TODO
        '''
        일단 augmentation은 getattr로 사용한다 해도 그 세팅값 전달을 어케할지가 고민..
        일단 2key dictionary로 -> aug 종류 ('kind')(type:strings), 세팅 값('settings')(type:tuple)
        이러면 아까와 또 마찬가지로 augmentation의 모든 인자값에 대한 parameter 를 txt에서 다 적어야함.
        '''
        aug_list = []
        for i in self.augmentation_params:
            aug_list.append(getattr(A, i['kind'])(*self.i['settings']))
        self.augmentations = A.Compose(aug_list)

        self.modality = self.modality   # TODO
        self.dimension = int(self.dimension) #TODO

        self.ckpt_dir = self.checkpoint_dir
        self.log_dir = self.logs_dir
        self.data_dir = self.data_dir

        
        self.task_type = self.task_type


        self.epochs = self.epochs
    
        
        
        
 

 
 