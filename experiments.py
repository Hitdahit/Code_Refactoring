_base_ = ['experiments.py']
import albumentations as A
import albumentations.pytorch
'''
MODEL FAMILY
'''
    # model (model caller 구현해야할듯...)
model = dict(family='model', lib='models', type='ResNet', 
             n_classes=2, model_size=50, pretrained = True)

    # optimizer
optimizer = dict(family='model', lib='torch.optim', type='Adam', 
                 lr=1e-3)

    # loss
loss = dict(family='model', lib='torch.nn', type='CrossEntropyLoss',
           weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    # if you use your custom loss
    # dict(family='runtime', lib='utils.family.runtime', type='custom_Loss, "put your hyperparams of your custom Loss

  # scheduler
scheduler = dict(family='model', lib='torch.optim.lr_scheduler', type='StepLR',
                 step_size=0.1, gamma=0.1, last_epoch=-1, verbose=False)
    # if you use your custom lr scheduler
    # dict(family='runtime', lib='utils.family.runtime', type='custom_LRScheduler', "put your hyperparams of your custom LRscheduler")
    # if you don't use lr scheduler then set this variable as None
    # scheduler = None


'''
RUNTIME FAMILY
    - use_amp: True or False

    - print_freq: (int) 프롬프트에 학습 로그를 띄울 주기

    - experiment_name: (str) 현 세팅의 실험 명.

    - ckpt_directory: (str) 학습 가중치 저장할 경로

    - log_directory: (str) 학습 로그 저장할 경로
'''

    # runtime settings
use_amp = False
print_freq = 1
epoch = 140
batch_size = 16
experiment_name = 'experiments'
ckpt_directory = './runs/{}/ckpt'.format(experiment_name)
log_directory = './runs/{}/log'.format(experiment_name)

'''
save_config

    log_dir: 학습 로그를 저장할 경로
    ckpt_dir: 학습된 모델 가중치 저장할 경로
    experiment_name: 해당 실험 명칭 폴더로 통합하여 위의 두 경로를 묶어줌.
    log_library: 'wandb' or 'tensorboard'
'''
save_config = dict(family='runtime', lib='utils.family.runtime', type='Saver',
                         log_dir=log_directory, ckpt_dir=ckpt_directory, experiment_name=experiment_name, log_library='tensorboard')

'''
save_config

    activation: 모델의 최종 logit의 activation function 결정
    threshold: None or (float) None이면 input 그대로를, (float)이면 그 값을 기준으로 크면 1, 작으면 0 리턴.
    name: (list(str)) 보고자하는 metric. 'accuracy', 'f_score', 'precision', 'recall'
    eps: (float) underflow 방지용 실수 값.
    beta: (int) f_score 만을 위한 값. f_score 사용 안할 경우 None.
'''
evaluation = dict(family='runtime', lib='utils.family.runtime', type='Metrics',
         batch_size=batch_size, activation='custom', threshold=0.5, name=['f_score', 'accuracy'], eps=1e-7, beta=1)


'''
DATASET FAMILY

    - modality: (str) CT, MR, EN, XRAY(?)

    - task_type:  (str) BC, MC, ML 
                !!! BC and MC will work without annotation_file

    - label_type: (str) one-hot(float), ordinal(int)

    - label_name: (list: str) ['label1', 'label2', ...]
                ex) ['Normal', 'Disease']

    - annotation_file: (str) None, ~.json, ~.csv, ~.xlsx
'''
modality = 'EN'

data_root = '/mnt/nas125_vol2/jisup/DATA/CD_TB/new_data/20220613_new/'

task_type = 'BC'

img_size = 512

classes = ['CD','TB']

'''

'''

labeler = dict(family='datautil', lib='utils.family.datautil', type='Source',
               task_type='BC', label_type='ordinal', label_name=classes)

prep_config = dict(family='datautil', lib='utils.family.datautil', type='Endo_preprocessor',
                   image_size=img_size, normalize_range='1', mode='default')


    # augmentation
    # param은 다시 확인
train_augmentations = dict(family='datautil', lib=None, type='augmentation',
        items = A.Compose([                
                A.OneOf([
                    A.MedianBlur(blur_limit=5, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5)
                    ], p=0.6),
                
    
                A.RandomBrightnessContrast(brightness_limit=(0.1, 0.5), contrast_limit=(0.1, 0.5), p=0.6),
                A.HueSaturationValue(hue_shift_limit=(0.1, 0.3), sat_shift_limit=(0.1, 0.3),
                                        val_shift_limit=(0.1, 0.3), p=0.3),
                
                A.OpticalDistortion(distort_limit=0.05, p=0.2),
                A.pytorch.transforms.ToTensorV2()
                ])
    )

valid_augmentations = dict(family='datautil', lib=None, type='augmentation',
        items = A.Compose([A.pytorch.transforms.ToTensorV2()]))

'''
dataset은 train에서 정의후 쓰는걸로.
'''