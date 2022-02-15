from concurrent.futures.process import _ResultItem


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

    def _set_value(self):
        n_classes = int(self.num_class)
    
        losses = {'BCE':nn.BCEWithLogitsLoss(),'CE':nn.CrossEntropyLoss(), 'F':}  # 추후 추가
        loss = None
        for key, value in losses.items():
            if key in self.loss:
                loss = value

        models = {'vgg':model.VGG_Classifier(n_classes, int(self.model_size)),\
                'resnet':model.ResNet_Classifier(n_classes, int(self.model_size)),\
                'densenet':model.DenseNet_Classifier(n_classes, int(self.model_size)),\
                'efficientnet': model.EfficientNet_Classifier(n_classes, int(self.model_size))}
        model = None
        for key, value in models.items():
            if key in self.model:
                model = value

        # optimizer TODO

        # scheduler TODO


        os.environ["CUDA_VISIBLE_DEVICES"]=self.GPU_number
        device = torch.device(f'cuda:{int(self.GPU_id)}' if torch.cuda.is_available() else 'cpu') # CPU/GPU에서 돌아갈지 정하는 device flag
        seed = int(self.seed)
        batch_size = int(self.batch_size)
        img_size = int(self.img_size)
        epochs = int(self.epochs)
        lr = float(self.leraning_rate)
        modality = self.modality   # TODO
        dimension = int(self.dimension) #TODO

        ckpt_dir = self.checkpoint_dir
        log_dir = self.logs_dir
        data_dir = self.data_dir

        # augmentation TOCO
        task_type = self.task_type


        epochs = self.epochs

def param_parser(self):

    
    n_classes = int(self.num_class)
    
    losses = {'BCE':nn.BCEWithLogitsLoss(),'CE':nn.CrossEntropyLoss(), 'F':}  # 추후 추가
    loss = None
    for key, value in losses.items():
        if key in self.loss:
            loss = value
    
    models = {'vgg':model.VGG_Classifier(n_classes, int(self.model_size)),\
            'resnet':model.ResNet_Classifier(n_classes, int(self.model_size)),\
            'densenet':model.DenseNet_Classifier(n_classes, int(self.model_size)),\
            'efficientnet': model.EfficientNet_Classifier(n_classes, int(self.model_size))}
    model = None
    for key, value in models.items():
        if key in self.model:
            model = value

    # optimizer TODO

    # scheduler TODO
        

    os.environ["CUDA_VISIBLE_DEVICES"]=self.GPU_number
    device = torch.device(f'cuda:{int(self.GPU_id)}' if torch.cuda.is_available() else 'cpu') # CPU/GPU에서 돌아갈지 정하는 device flag
    seed = int(self.seed)
    batch_size = int(self.batch_size)
    img_size = int(self.img_size)
    epochs = int(self.epochs)
    lr = float(self.leraning_rate)
    modality = self.modality   # TODO
    dimension = int(self.dimension) #TODO

    ckpt_dir = self.checkpoint_dir
    log_dir = self.logs_dir
    data_dir = self.data_dir

    # augmentation TOCO
    task_type = self.task_type

    
    epochs = self.epochs
    
    
    return device, seed, batch_size, img_size, epochs, lr, modality, \
        dimension, ckpt_dir, log_dir, data_dir, augmentation, task_type, loss, model
 
 