import os
import torch
import torch.nn as nn
import torchvision.models as models


'''
Training strategy

    1. Resume
    2. Transfer learning
        2-1. gradual unfreezing
        
'''
fn_tonumpy = lambda x: x.cpu().detach().numpy().transpose(0, 2, 3, 1)

def resume(args, ckpt_dir, experiment_name, epoch, device, is_zipped=False):
    # runtime.py 에서 저장하는 포맷 바꾸면 .format 안의 내용 바꿀 것.
    if is_zipped == True:
        state_dict = torch.load(os.path.join(ckpt_dir, experiment_name, '{}.pth.tar.gz'.format(epoch)), map_location='cpu')
    else:
        state_dict = torch.load(os.path.join(ckpt_dir, experiment_name, '{}.pth'.format(epoch)), map_location='cpu')

    for i in state_dict.keys():
        getattr(args, i).load_state_dict(state_dict[i])  
    
    # DP
    # pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    
def transfer(args, pretrain):
    
    if pretrain == 'imagenet':
        net = models.resnet50(pretrained=True)
        
    elif pretrain == 'downstream':
        net = args.net(args.num_classes)
    
        pretrained_weight = 'path of pretrained weight'  
        resnet_model_pretrained = pretrained_weight
        
        if resnet_model_pretrained is not None:
            if os.path.isfile(resnet_model_pretrained):
                print("=> loading checkpoint '{}'".format(resnet_model_pretrained))
                checkpoint = torch.load(resnet_model_pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
        #             print(k)
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = net.load_state_dict(state_dict, strict=False)
                print(msg.missing_keys)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                print("=> loaded pre-trained model '{}'".format(resnet_model_pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(resnet_model_pretrained))

            ###freeze all layers but the last fc
            # for name, param in model.named_parameters():
            #     if name not in ['fc.weight', 'fc.bias']:
            #         param.requires_grad = False
    net.fc = nn.Linear(2048, args.num_classes)
                

'''
Gradual Unfreezing
    Use Only When your model is fixed and well examined.
'''
def freeze_params(model: torch.nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model: torch.nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True
        
def gradual_unfreezing(net, epoch, start, interval):
    
    # interval 씩 one stage block 풀기, start epoch까지는 아예 고정
    if epoch >= 0 and epoch <= 100:
        freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
        print("Freeze encoder ...!!")
    elif epoch >= 101 and epoch < 111:
        print("Unfreeze encoder.layer4 ...!")
        unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
    elif epoch >= 111 and epoch < 121:
        print("Unfreeze encoder.layer3 ...!")
        unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
    elif epoch >= 121 and epoch < 131:
        print("Unfreeze encoder.layer2 ...!")
        unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
    elif epoch >= 131 and epoch < 141:
        print("Unfreeze encoder.layer1 ...!")
        unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
    else :
        print("Unfreeze encoder.stem ...!")
        unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
        
    return model