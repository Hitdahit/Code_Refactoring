'''
Metirc
BC: binary class, ML: multi label, MC: multi class
'''
def BC_metric(y, yhat, thresh=0.5):
    cor = 0
    yhat = torch.softmax(yhat, dim=1)
    yhat[yhat>=thresh] = 1
    yhat[yhat<thresh] = 0
    
    y = torch.argmax(y, dim=1)
    yhat = torch.argmax(yhat, dim=1)
    for i, j in zip(y, yhat):
        if i==j:
            cor = cor+1
            
    acc = cor/len(yhat)
    
    return acc

def ML_metric(y_true, y_pred):
    '''
    일반적인 accuracy (example_based_accuracy)
    전체 predicted와 actual label 중에 맞은 predicted label 비율
    
    if y_true = np.array([[0,1,1], [1,1,1]])
       y_pred = np.array([[1,0,1], [0,0,0]])
    numerator = [1,0]
    denominator = [3,3]
    instance_accuracy = [0.333333, 0]
    np.sum(instance_accuracy) : 0.33333
    '''

    # compute true positive using the logical AND operator
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1) 

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    instance_accuracy = numerator/denominator

    return np.sum(instance_accuracy) # accuracy 계산 하려면 data 갯수를 나눠줘야 됨

def MC_metric(y, yhat):

    acc_targets = []
    acc_outputs = []

    y_temp = y
    for t in y_temp.view(-1,1).cpu():
        acc_targets.append(t.item()) 

    _, yhat_temp = torch.max(yhat, 1)
    for u in yhat_temp.view(-1,1).cpu():
        acc_outputs.append(u.item())

    cor = 0
    for i in range(len(acc_targets)):
        if acc_outputs[i] == acc_targets[i]:
            cor += 1

    acc = cor/len(acc_outputs)

    return acc