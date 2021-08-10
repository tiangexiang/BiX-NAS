from torch.utils.data import DataLoader
import sys
import os
import torch
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
from .metrics import *
from .dataloaders import Nuclei
from .Phase1model import Phase1
from .Phase2model import Phase2, decode
from .Phase2tools import BFS
import pytorch_version.Phase1model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(args):
    """
    load data and create data loaders
    """
    # augmentations
    transforms = iaa.Sequential([
            iaa.Rotate((-5., 5.)),
            iaa.TranslateX(percent=(-0.05,0.05)),
            iaa.TranslateY(percent=(-0.05,0.05)),
            iaa.Affine(shear=(-10, 10)),
            iaa.Affine(scale=(0.8, 1.2)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])

    # load data and create data loaders
    train_set = Nuclei(args.train_data, 'monuseg', batchsize=args.batch_size, transforms=transforms)
    test_set = Nuclei(args.valid_data, args.valid_dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_set, test_set, train_loader, test_loader

def create_model(args):
    """
    BiX-NAS model
    :Phase1: do not have genes
    :Phase2/retraining: decode searched genes and train 
    """
    if args.gene is None: # Phase1 
        model = Phase1(in_channel=args.in_channel,
                   iterations=args.iter,
                   num_classes=args.num_class,
                   multiplier=args.multiplier).to(device).float()
    else: # Phase2 / retraining
        gene = decode(args.gene, args.iter, 4) 
        print(type(args.in_channel))
        model = Phase2(in_channel=args.in_channel,
                iterations=args.iter,
                num_classes=args.num_class,
                multiplier=args.multiplier,
                gene=gene,
                with_att=args.with_att).to(device).float()
    print('generated model:', type(model))
    return model

def compute_metrics(x, y, tot_iou, tot_dice, args):
    """
    compute accuracy metrics: IoU, DICE
    """
    if args.num_class == 1:
        iou_score = iou(y, x)
        dice_score = dice_coef(y, x)
    else:
        iou_score = miou(y, x, args.num_class)
        dice_score = mdice(y, x, args.num_class)
    tot_iou += iou_score
    tot_dice += dice_score
    return tot_iou, tot_dice

def get_flops(args, model, new_network=None):
    """
    compute computations: MACs, #Params
    """
    if new_network is not None:
        model.reset_gene(new_network, BFS(args.iter, new_network))

    from ptflops import get_model_complexity_info
    save_stdout = sys.stdout
    sys.stdout = open('./trash', 'w')
    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
    sys.stdout = save_stdout
    print("macs: %.4f x 10^9, num params: %.4f x 10^6" % (float(macs) * 1e-9, float(params) * 1e-6))
    return macs, params

def save_model(args, model):
    weights = '_weights'
    cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+'_best'+weights+'.pt'
    torch.save({'state_dict':model.state_dict()}, "./checkpoints/"+args.exp+"/"+cpt_name)

def save_metrics(val_loss, val_iou, val_dice, args):
    if not os.path.exists("./checkpoints/"+args.exp+"/outputs"):
        os.mkdir("./checkpoints/"+args.exp)
        os.mkdir("./checkpoints/"+args.exp+"/outputs")
    with open("./checkpoints/"+args.exp+"/outputs/result.txt", 'w+') as f:
        f.write('Validation loss:\t'+str(val_loss)+'\n')
        f.write('Validation  iou:\t'+str(val_iou)+'\n')
        f.write('Validation dice:\t'+str(val_dice)+'\n')
    print('Metrics have been saved to:', "checkpoints/"+args.exp+"/outputs/result.txt")

def save_masks(segmentations, inputs, gts, args):
    """
    save segmentation masks
    :segmentations: predictions
    :inputs: input images
    "gts: ground truth
    """
    results = np.transpose(np.concatenate(segmentations, axis=0), (0, 2, 3, 1))
    inputs = np.transpose(np.concatenate(inputs, axis=0), (0, 2, 3, 1))
    gts = np.concatenate(gts, axis=0)
    if len(gts.shape) == 4:
        gts = np.transpose(gts, (0, 2, 3, 1))
    if args.num_class == 1:
        results = (results > 0.5).astype(np.float32) # Binarization. Comment out this line if you don't want to
    else:
        r_map = {0: 0., 1: 1., 2: 0., 3: 1., 4: 0.}
        g_map = {0: 0., 1: 0., 2: 1., 3: 1., 4: 0.}
        b_map = {0: 0., 1: 0., 2: 1., 3: 0., 4: 1.}
        ph = np.zeros(results.shape[:3] + (3,))
        gt_ph = np.zeros(results.shape[:3] + (3,))
        results = np.argmax(results, axis=-1)
        for b in range(results.shape[0]):
            for i in range(results.shape[1]):
                for j in range(results.shape[2]):
                    if results[b,i,j] > 0:
                        ph[b,i,j,0] = r_map[results[b,i,j]]
                        ph[b,i,j,1] = g_map[results[b,i,j]]
                        ph[b,i,j,2] = b_map[results[b,i,j]]
                    if gts[b,i,j] > 0:
                        gt_ph[b,i,j,0] = r_map[gts[b,i,j]]
                        gt_ph[b,i,j,1] = g_map[gts[b,i,j]]
                        gt_ph[b,i,j,2] = b_map[gts[b,i,j]]
        results = ph

    print('Saving segmentations...')
    if not os.path.exists("./checkpoints/"+args.exp+"/outputs/segmentations"):
        os.mkdir("./checkpoints/"+args.exp+"/outputs/segmentations")

    for i in range(results.shape[0]):
        if args.num_class == 1:
            plt.imsave("./checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+".png",results[i,:,:,0],cmap='gray') # binary segmenation
            plt.imsave("./checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+"_gt.png",gts[i,:,:,0], cmap='gray')
            plt.imsave("./checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+"_input.png",inputs[i,:,:])
        else:
            plt.imsave("./checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+".png",results[i,:,:,:])
            plt.imsave("./checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+"_gt.png",gt_ph[i,:,:,:])
            plt.imsave("./checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+"_input.png",inputs[i,:,:,0], cmap='gray')
    print('A total of '+str(results.shape[0])+' segmentation results have been saved to:', "checkpoints/"+args.exp+"/outputs/segmentations/")

def get_skip(model):
    """
    Get Phase1 searched skips
    """
    encoder = []
    decoder = []
    for name, module in model.named_modules():
        if isinstance(module, pytorch_version.Phase1model.Block):
            # new style!
            topk = torch.topk(module.alpha, 1, dim=0)[1].cpu().numpy()
            topk = sorted(topk.tolist()[0])
            if module.alpha.shape[-1] == 2: # last decoder
                topk = [0] + [v+1 for v in topk]
            
            # old style!
            #topk = torch.topk(module.alpha, 3, dim=-1, sorted=True)[1][0,0,0,0].cpu().numpy()
            #topk = list(topk)
            #if module.alpha.shape[-1] == 4: # last decoder
            #    topk = [0] + [v+1 for v in  topk[:2]]
            #topk = sorted(topk)

            if name[:7] == 'encoder':
                encoder.append(topk)
            else:
                decoder.append(topk)
    return encoder, decoder

def encode(args):
    """
    encode Phase1 searched skips to genes and save them
    """
    if args.model_path is None:
        weights = '_weights'
        cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+'_best'+weights+'.pt'
        model_path = "./checkpoints/"+args.exp+"/"+cpt_name
    else:
        model_path = args.model_path
    print('Restoring model from path: '+model_path)
    
    checkpoint = torch.load(model_path)
    model = create_model(args)
    model.load_state_dict(checkpoint['state_dict'])
    level = 4

    # make a file to save Phase1 genes
    if not args.save_gene:
        # if do not specify the path to save Phase1 genes
        if not os.path.exists("./phase1_genes/"):
            os.mkdir("./phase1_genes/")
        if args.gene is None:
            encoder, decoder = get_skip(model)
            with open('./phase1_genes/'+'phase1_gene_'+args.exp+'.txt', 'w+') as f:
                f.write('%.3f_%f\n' % (0.0, 0.0))
                for it in range(len(encoder) // level):
                    for l in range(level):
                        print('enc', encoder[it * level + l])
                        f.write('enc_'+str(it)+'_'+str(l))
                        for i in encoder[it * level + l]:
                            f.write('_'+str(i))
                        f.write('\n')
                    for l in range(level):
                        print('dec', decoder[it * level + l])
                        f.write('dec_'+str(it)+'_'+str(l))
                        for i in decoder[it * level + l]:
                            f.write('_'+str(i))
                        f.write('\n')
    else:
        if args.gene is None:
            encoder, decoder = get_skip(model)
            with open(args.save_gene+'/'+'phase1_gene_'+args.exp+'.txt', 'w+') as f:
                f.write('%.3f_%f\n' % (0.0, 0.0))
                for it in range(len(encoder) // level):
                    for l in range(level):
                        print('enc', encoder[it * level + l])
                        f.write('enc_'+str(it)+'_'+str(l))
                        for i in encoder[it * level + l]:
                            f.write('_'+str(i))
                        f.write('\n')
                    for l in range(level):
                        print('dec', decoder[it * level + l])
                        f.write('dec_'+str(it)+'_'+str(l))
                        for i in decoder[it * level + l]:
                            f.write('_'+str(i))
                        f.write('\n')