_base_ = ['custom_param.py']
'''
MODEL FAMILY
'''
    # model (model caller 구현해야할듯...)
model = dict(family='model', lib='model', type='resnet', 
             name='resnet34')

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
ckpt_directory = '/~~~/train_v1/ckpt_v1'
log_directory = '/~~~/train_v1/log_v1'

    #기본적으론 torch.save 쓰되 함수로 묶어서 다른 부가기능도 끼워 넣기.
checkpoint_config = dict(family='runtime', lib='utils', type='save',
                         interval=1, ckpt_dir=ckpt_directory)
log_config = dict(family='runtime', lib='utils', type='save',
                  sub_lib='wandb', log_dir=log_directory)


    # augmentation
    # param은 다시 확인
train_augmentations = [
    dict(family='runtime', lib='albumentations',type='Resize', 
         scale=(224, 224), keep_ratio=False),
    dict(family='runtime', lib='albumentations', type='Flip',
         flip_ratio=0.5),
    dict(family='runtime', lib='albumentations', type='FormatShape',
         input_format='NCHW'),
    dict(family='runtime', lib='albumentations', type='Collect', 
         keys=['imgs', 'label'], meta_keys=[]),
    dict(family='runtime', lib='albumentations', type='ToTensor', 
         keys=['imgs', 'label'])
]

valid_augmentations = [
    dict(family='runtime', lib='albumentations', type='Collect', 
         keys=['imgs', 'label'], meta_keys=[]),
    dict(family='runtime', lib='albumentations', type='ToTensor', 
         keys=['imgs', 'label'])
]


evaluation = [
    dict(family='runtime', lib='utils', type='top_k_accuracy',
         k=1, interval=2),
    dict(family='runtime', lib='utils', type='mean_class_accuracy',
        interval=2)
]

'''
DATASET FAMILY
'''
    # dataset settings
modality = 'EN'

data_root = '/~~~/datav1'

task_type = 'BC'

img_size = 512

    #label from directory?
    #label from path?
    #label from json / txt / excel / etc... ?
    #\
    #label type
    #one-hot / ordinal / regression
    # per-patient / per-image label (CT, MR)
classes = ['EoE', 'Normal']
labeler = dict(family='datautil', lib='utils', task_type='BC', label_type='one-hot', label_name=classes, label_source='from_path')

prep_config = dict(family='datautil', lib='utils', type='None',
                  is_rgb=True, modality=modality, img_dir=data_root, 
                  img_sz=(img_size, img_size), WW=None, WL=None, drop_percentile=1.0,
                  norm_range=(0, 1), norm_method='minmax')

'''
dataset은 train에서 정의후 쓰는걸로.
'''

'''
부가기능 list:
    1. utils/resume   -  지영         etc-family
    2. utils/transfer learning - 지영 etc-family
    3. utils/log frequency - 강길     runtime-family
    4. utils/ckpt interval - 강길     runtime-family
    5. utils/metric - 강길            runtime-family
    6. utils/labeler - 종준           datautil-family
    7. 
    8. multi GPU - 

'''