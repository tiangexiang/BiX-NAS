import numpy as np
from sklearn.metrics import confusion_matrix

def iou(y_true, y_pred, num_class=1, threshold=0.5, eps=1e-7):
    """
    compute IoU
    """
    y_pred = threshold_binarize(y_pred, threshold)
    #print(y_pred.shape, y_true.shape)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    #print(y_pred_f[10:20], y_true_f[10:20])
    intersection = np.sum(y_true_f * y_pred_f)
    #print(intersection)
    return (intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + eps)

def dice_coef(y_true, y_pred, num_class=1, threshold=0.5, eps=1e-7):
    """
    compute DICE
    """
    y_pred = threshold_binarize(y_pred, threshold)
    #print(y_pred.shape, y_true.shape)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    #print(intersection)
    return (2. * intersection) / (
                np.sum(y_true_f) + np.sum(y_pred_f) + eps)

def threshold_binarize(x, threshold=0.5):
    return (x > threshold).astype(np.float32)

def miou(y_true, y_pred, num_class=1, reduce=True, ignore_background=True):
    """
    compute mean IoU
    """
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.argmax(axis=1).flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=range(num_class))
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection + 1e-7
    IoU = intersection / union.astype(np.float32)
    #print('iou shape', IoU.shape)
    if reduce:
        return np.mean(IoU[1:]) if ignore_background else np.mean(IoU)
    else:
        return np.mean(IoU[1:]) if ignore_background else np.mean(IoU), IoU

def mdice(y_true, y_pred, num_class=1, reduce=True, ignore_background=True):
    """
    compute mean DICE
    """
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.argmax(axis=1).flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=range(num_class))
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set + 1e-7
    DICE = 2*intersection / union.astype(np.float32)
    if reduce:
        return np.mean(DICE[1:]) if ignore_background else np.mean(DICE)
    else:
        return np.mean(DICE[1:]) if ignore_background else np.mean(DICE), DICE

def sens_spec(true, pred, threshold=0.3, eps=1e-7):
    """
    compute sensitivity and specificity
    """
    pred = threshold_binarize(pred, threshold).flatten()
    true = true.flatten().astype('int32')
    pred = pred.flatten().astype('int32')
    TP = np.sum(true * pred)
    TN = pred.shape[0] - np.count_nonzero(true + pred)
    FP = np.count_nonzero(pred - true == 1)
    FN = np.count_nonzero(true - pred == 1)
    sens = TP / (TP + FN + eps)
    spec = TN / (TN + FP + eps)
    #cm = confusion_matrix(true, pred)
    #sens = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    #spec = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    return sens, spec

class Metrics3D(object):
    """
    3D metrics
    """
    def __init__(self, gts):
        
        self.gts = [gt.astype('float32') for gt in gts]
        
        self.refresh()
    def refresh(self):
        self.accum_matrix = [np.zeros(self.gts[i].shape, dtype=np.float32) for i in range(len(self.gts))]
        self.predict_matrix = [np.zeros(self.gts[i].shape, dtype=np.float32) for i in range(len(self.gts))]
    def __call__(self, output, ret):
        #print(output.shape)
        #print(ret['coords'])
        c, w, h, d = output.shape
        idx = ret['idx'][0]
        
        coords = ret['coords'] # x y z
        #print(idx)
        #print(self.predict_matrix[idx].shape)
        #print(coords[0]+w, coords[1]+h, coords[2]+d)
        self.predict_matrix[idx][coords[0]:coords[0]+w, coords[1]:coords[1]+h, coords[2]:coords[2]+d] += output[1]
        self.accum_matrix[idx][coords[0]:coords[0]+w, coords[1]:coords[1]+h, coords[2]:coords[2]+d] += 1.

    def compute(self):
        #print('check this!', np.count_nonzero(self.accum_matrix == 0))
        senss = []
        specs = []
        dices = []
        ious = []
        for i in range(len(self.gts)):
            tmp_sens = 0.
            tmp_spec = 0.
            tmp_dice = 0.
            tmp_iou = 0.
            for d in range(self.gts[i].shape[-1]):
                #print(self.gts[i][60:70,60:70,d], self.predict_matrix[i][60:70,60:70,d])
                #print('pred: ', self.predict_matrix[i][60:70,60:70,d])
                #print(self.gts[i].dtype, self.predict_matrix[i].dtype)
                tmp_dice += dice_coef(self.gts[i][:,:,d], self.predict_matrix[i][:,:,d]/self.accum_matrix[i][:,:,d])
                sens, spec = sens_spec(self.gts[i][:,:,d], self.predict_matrix[i][:,:,d]/self.accum_matrix[i][:,:,d])
                tmp_sens += sens
                tmp_spec += spec
                #print(tmp_dice)
                #ioou = miou(self.gts[i][:,:,d], self.predict_matrix[i][:,:,:,d]/self.accum_matrix[i][:,:,:,d], num_class=2, axis=0,ignore_background=False)
                #print(ioou)
                tmp_iou += iou(self.gts[i][:,:,d], self.predict_matrix[i][:,:,d]/self.accum_matrix[i][:,:,d])
                #tmp_dice += dice_coef(self.gts[i][:,:,d], self.predict_matrix[i][:,:,d]/self.accum_matrix[i][:,:,d])
                #tmp_dice += mdice(self.gts[i][:,:,d], self.predict_matrix[i][:, :,:,d]/self.accum_matrix[i][:, :,:,d], num_class=2, axis=0, ignore_background=False)
            #print(tmp_iou)
            specs.append(tmp_spec/self.gts[i].shape[-1])
            senss.append(tmp_sens/self.gts[i].shape[-1])
            dices.append(tmp_dice/self.gts[i].shape[-1])
            ious.append(tmp_iou/self.gts[i].shape[-1])
        self.refresh()
        #print(dices)
        return dices, senss, specs, ious


