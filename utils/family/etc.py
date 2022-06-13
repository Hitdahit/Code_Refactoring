import os
import torch
import 
'''
Training strategy

    1. Resume
    2. Transfer learning
        2-1. gradual unfreezing
'''
fn_tonumpy = lambda x: x.cpu().detach().numpy().transpose(0, 2, 3, 1)

def resume(net, optimizer, ckpt_dir, experiment_name, epoch, device, scheduler=None):
    # runtime.py 에서 저장하는 포맷 바꾸면 .format 안의 내용 바꿀 것.
    state_dict = torch.load(os.path.join(ckpt_dir, experiment_name, '{}.pth'.format(epoch)), map_location='cpu')
    
    net.load_state_dict(state_dict['net'])
    optimizer.load_state_dict(state_dict['net'])
    if 'scheduler' in state_dict.keys():
        scheduler.load_state_dict(state_dict['scheduler'])
    
    return net, optimizer, scheduler

def transfer():
    pass
                


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