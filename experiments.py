_base_ = ['experiments.py']
import albumentations as A
import albumentations.pytorch
'''
MODEL FAMILY
'''
    # model (model caller 구현해야할듯...)
model = dict(family='model', lib='models', type='ResNet', 
             n_classes=2, model_size=34, pretrained = False)

    # optimizer
optimizer = dict(family='model', lib='torch.optim', type='SGD', 
                 lr=0.01, momentum=0.9, weight_decay=0.0005)

    # scheduler
scheduler = dict(family='model', lib='torch.optim.lr_scheduler', type='StepLR',
                 step_size=0.1, gamma=0.1, last_epoch=-1, verbose=False)

    # loss
loss = dict(family='model', lib='torch.nn', type='BCEWithLogitsLoss',
           weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

'''
RUNTIME FAMILY
'''
    # runtime settings
experiment_name = 'wow'
ckpt_directory = './runs/{}/ckpt'.format(experiment_name)
log_directory = './runs/{}/log'.format(experiment_name)

    #기본적으론 torch.save 쓰되 함수로 묶어서 다른 부가기능도 끼워 넣기.
save_config = dict(family='runtime', lib='utils.family.runtime', type='Saver',
                         log_dir=log_directory, ckpt_dir=ckpt_directory, experiment_name=experiment_name, log_library='tensorboard')

evaluation = [
    dict(family='runtime', lib='utils.family.runtime', type='Metrics',
         activation=None, threshold=0.5, name='f_score', eps=1e-7, beta=1),
    dict(family='runtime', lib='utils.family.runtime', type='Metrics',
        activation=None, threshold=0.5, name='f_score', eps=1e-7, beta=1)
]

'''
DATASET FAMILY
'''
modality = 'EN'

data_root = '/~~~/datav1'

task_type = 'BC'

img_size = 512


classes = ['EoE', 'Normal']
labeler = dict(family='datautil', lib='utils.family.datautil', type='base_labeler',
               task_type='BC', label_type='one-hot', label_name=classes, label_source='from_path')

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