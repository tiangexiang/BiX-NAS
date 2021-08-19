from torch.utils.data import DataLoader
import sys
import os
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch
from tqdm import tqdm
from .metrics import *
from .dataloaders import Nuclei
from .Phase1tools import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:0')

#torch.manual_seed(1)
#np.random.seed(1)
#random.seed(1)
#torch.cuda.manual_seed_all(1)
#torch.cuda.manual_seed(1)
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True

def train(args):
    """
    Phase1 training / retraining
    """
    # load data
    train_set, test_set, train_loader, test_loader = load_data(args)

    # create model
    model = create_model(args)
    # define criterion
    criterion = BCELoss() if args.num_class == 1 else CrossEntropyLoss(weight=train_set.CLASS_WEIGHTS)
    # create optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)  
    # keras lr decay equivalent
    lr_decay = 3e-3
    fcn = lambda step: 1./(1. + lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    print('model successfully built and compiled.')
  
    # make dictionary for saving results
    if not os.path.exists("./checkpoints/"+args.exp+"/outputs"):
        os.mkdir("./checkpoints/"+args.exp)
        os.mkdir("./checkpoints/"+args.exp+"/outputs")

    best_iou = 0.
    print('\nStart training...')
    for epoch in range(args.epochs):
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.
        val_loss = 0.
        val_iou = 0.
        val_dice = 0.

        # training
        model.train()
        for step, ret in enumerate(tqdm(train_loader, desc='[TRAIN] Epoch '+str(epoch+1)+'/'+str(args.epochs), disable=True)):
            if step >= 1:
                break

            x = ret['x'].to(device)
            y = ret['y'].to(device)
            
            optimizer.zero_grad()
            output = model(x)

            # loss
            l = criterion(output, y)
            tot_loss += l.item()
            l.backward()
            optimizer.step()

            # compute metrics
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            tot_iou, tot_dice = compute_metrics(x, y, tot_iou, tot_dice, args)

        scheduler.step()

        print('[TRAIN] Epoch: '+str(epoch+1)+'/'+str(args.epochs),
              'loss:', tot_loss/len(test_loader),
              'iou:', tot_iou/len(test_loader),
              'dice:', tot_dice/len(test_loader))

        # validation
        model.eval()
        with torch.no_grad():
            for step, ret in enumerate(tqdm(test_loader, desc='[VAL] Epoch '+str(epoch+1)+'/'+str(args.epochs), disable=True)):
                x = ret['x'].to(device)
                y = ret['y'].to(device)

                output = model(x)

                # loss
                l = criterion(output, y)
                val_loss += l.item()

                # compute metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)

        val_iou = val_iou/len(test_loader)
        val_dice = val_dice/len(test_loader)
   
        # save model
        if val_iou >= best_iou:
            best_iou = val_iou
            save_model(args, model)

        print('[VAL] Epoch: '+str(epoch+1)+'/'+str(args.epochs),
        'val_loss:', val_loss/len(test_loader),
        'val_iou:', val_iou,
        'val_dice:', val_dice,
        'best val_iou:', best_iou)

    print('\nTraining fininshed!')


def evaluate(args):

    # load data and create data loader
    if args.dataset == 'nuclei':
        test_set = Nuclei(args.valid_data, args.valid_dataset)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        print('Invalid dataset!')

    # load best model
    if args.model_path is None:
        weights = '_weights'
        cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+'_best'+weights+'.pt'
        model_path = "./checkpoints/"+args.exp+"/"+cpt_name
    else:
        model_path = args.model_path
    print('Restoring model from path: '+model_path)

    # create model
    model = create_model(args)

    # checkpoint = torch.load(model_path) # cuda 
    checkpoint = torch.load(model_path, map_location=torch.device('cpu')) #cpu
    
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    criterion = BCELoss() if args.num_class == 1 else CrossEntropyLoss()

    val_loss = 0.
    val_iou = 0.
    val_dice = 0.

    segmentations = []
    inputs = []
    gts = []

    print('\nStart evaluation...')
    model.eval()
    with torch.no_grad():
        for step, ret in enumerate(tqdm(test_loader)):

            x = ret['x'].to(device)
            y = ret['y'].to(device)
            input_x = x.cpu().numpy()
            output = model(x)

            # loss
            l = criterion(output, y)
            val_loss += l.item()

            # compute metrics
            x, y = output.detach().cpu().numpy(), y.cpu().numpy()
            val_iou, val_dice = compute_metrics(x, y, val_iou, val_dice, args)

            # save predictions
            if args.save_result:
                segmentations.append(x)
                inputs.append(input_x)
                gts.append(y)

    val_iou = val_iou/len(test_loader)
    val_dice = val_dice/len(test_loader)
    val_loss = val_loss/len(test_loader)

    print('Validation loss:\t', val_loss)
    print('Validation  iou:\t', val_iou)
    print('Validation dice:\t', val_dice)

    # compute computations
    get_flops(args, model)
    print('\nEvaluation finished!')

    # save results
    if args.save_result:
        save_metrics(val_loss, val_iou, val_dice, args)
        save_masks(segmentations, inputs, gts, args)

    # Phase1 need to save genes
    if args.Phase1:
        encode(args)
