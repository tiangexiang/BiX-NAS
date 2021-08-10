import os
from torch.nn import BCELoss
from torch.optim import Adam
import torch
from tqdm import tqdm
from .metrics import *
import numpy as np
import copy
from .Phase2model import Phase2, decode
import argparse
from .dataloaders import *
from .Phase2tools import *
from .Phase1tools import load_data, get_flops
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def search(args):
    """
    Conduct Phase2 search from the last extraction stage pair to the first
    :new_networks: networks that ready to be searched, i.e., retained networks from the preceding search iteration
    :candidate: number of retained networks at each search iteration
    """
    if args.save_gene:
        if os.path.exists(args.save_gene+'/'+args.exp):
            shutil.rmtree(args.save_gene+'/'+args.exp)
        os.makedirs(args.save_gene+'/'+args.exp)
    else:
        if os.path.exists('./phase2_genes/'+args.exp):
            shutil.rmtree('./phase2_genes/'+args.exp)
        os.makedirs('./phase2_genes/'+args.exp)

    # load data first!!
    _, _, train_loader, test_loader = load_data(args)

    hyper_network = decode(args.gene, args.iter, 4) # Phase1 searched architecture
    new_networks = [hyper_network] # retained networks at each generation
    candidate = [2,2,2,2,2,3] # number of retained networks at each generation

    for iteration in range(2*args.iter-1, 0, -1): # 5th extractio pair to the first
        # sample and search between an extraction stage pair
        new_networks = search_single_iteration(args, hyper_network, new_networks, iteration)
    
        # train
        networks, scores = train_single_iteration(args, new_networks, iteration, train_loader, test_loader)
        # sort based on iou
        sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k][0])[:candidate[iteration]]
        print('Top networks are: ', sorted_idx)
        new_networks = [networks[idx] for idx in sorted_idx]  # ready to be searched at the next search iteration
       

def search_single_iteration(args, hyper_network, networks, iteration):
    """
    Search between an extraction stage pair:
    : hyper_network: Phase1 searched architecture
    : networks: retained architectures from the preceding search iteration
    : random_num: number of sampled architectures s
    """
    if args.save_gene:
        if os.path.exists(args.save_gene+'/'+args.exp+'/iteration_'+str(iteration)):
            shutil.rmtree(args.save_gene+'/'+args.exp+'/iteration_'+str(iteration))
        os.makedirs(args.save_gene+'/'+args.exp+'/iteration_'+str(iteration))
    else:
        if os.path.exists('./phase2_genes/'+args.exp+'/iteration_'+str(iteration)):
            shutil.rmtree('./phase2_genes/'+args.exp+'/iteration_'+str(iteration))
        os.makedirs('./phase2_genes/'+args.exp+'/iteration_'+str(iteration))

    print('\nSample up to '+str(args.random_num)+' different archs at iteration '+str(iteration))
    new_networks = []
    sampled = []
    count = 0 
    # draw samples
    for i in range(args.random_num):
        sampled_skip = sample(args, hyper_network, iteration, sampled)
        if sampled_skip is None:
            print('sampling exhausted!')
            break
        sampled_skip = list(sampled_skip)
        sampled.append(sampled_skip)
        # draw s samples for each retained network
        for network in networks:
            new_network = copy.deepcopy(network)
            new_network[iteration] = sampled_skip
            new_networks.append(new_network)
            print('#'+str(count)+' sample at iteration '+str(iteration)+' :')
            print_network(new_network)
            # save s samples
            if not args.save_gene:
                save_gene(os.path.join('./phase2_genes/', args.exp, 'iteration_'+str(iteration),'samples'),
                        'arch_'+str(count), new_network, 0.0, 0.0)
            else:
                save_gene(os.path.join(args.save_gene, args.exp, 'iteration_'+str(iteration),'samples'),
                        'arch_'+str(count), new_network, 0.0, 0.0)
            count += 1
    print(str(count)+' sampled at iteration '+str(iteration))

    return new_networks

def train_single_iteration(args, new_networks, iteration, train_loader, test_loader):
    """
    train between an extraction stage pair and retain networks at the Pareto front
    """
    print('\n#####begin training '+str(len(new_networks))+' models#####')
    tot_score = train(args, new_networks, iteration, train_loader, test_loader)

    # retain networks at the Pareto front
    print('\n#####iteration '+str(iteration)+' training finished#####')
    print('\n winners:',end='')
    tot_score = np.array(tot_score)
    ans = get_pareto_front(tot_score, False) # get networks at the Pareto front
    networks = []
    return_score = []
    for winner in ans:
        print('#'+str(winner),end='')
        if not args.save_gene:
            save_gene(os.path.join('./phase2_genes/', args.exp, 'iteration_'+str(iteration),'winners'),
                    'arch_'+str(winner), new_networks[winner], 1. - tot_score[winner][0], tot_score[winner][1]) # save winners
        else:
            save_gene(os.path.join(args.save_gene, args.exp, 'iteration_'+str(iteration),'winners'),
                    'arch_'+str(winner), new_networks[winner], 1. - tot_score[winner][0], tot_score[winner][1]) # save winners
        networks.append(new_networks[winner]) 
        return_score.append(list(tot_score[winner]))
    print()

    return networks, return_score


def train(args, new_networks, iterations, train_loader, test_loader):
    """
    Phase2 training
    """
    # create super model
    model = Phase2(iterations=args.iter,
                num_classes=args.num_class,
                multiplier=args.multiplier,
                gene=new_networks[0]).to(device).float()
    
    criterion = BCELoss()
    optimizer = Adam(params=model.parameters(), lr=args.lr / len(new_networks), weight_decay=0.)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,20,30], gamma=0.1)

    best_ious = [0. for _ in range(len(new_networks))]

    for epoch in range(args.epochs):
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.   

        # training
        model.train()
        for step, ret in enumerate(tqdm(train_loader, desc='[TRAIN] Epoch '+str(epoch+1)+'/'+str(args.epochs), disable=True)):
            # if step >= args.steps:
            #     break
            x = ret['x'].to(device).float()
            y = ret['y'].to(device).float()

            optimizer.zero_grad()
            cum_loss = None
            # forward head network
            return_value = model(x, iterations=iterations, profile=False)
            for new_network in new_networks:
                # set tail network topology
                model.reset_gene(new_network)
                # forward tail network
                output = model(x, iterations=iterations, return_value=return_value, profile=False)

                # loss
                l = criterion(output, y)
                if cum_loss is None:
                    cum_loss = l
                else:
                    cum_loss += l
                tot_loss += l.item()

            tot_loss /= len(new_networks)
            cum_loss = cum_loss / len(new_networks)
            cum_loss.backward()
            optimizer.step()

        scheduler.step()

        print('[TRAIN] Epoch: '+str(epoch+1)+'/'+str(args.epochs),
              'loss:', tot_loss/args.steps)

        # validation
        val_losss = [0. for _ in range(len(new_networks))]
        val_ious = [0. for _ in range(len(new_networks))]
        val_dices = [0. for _ in range(len(new_networks))]

        model.eval()
        with torch.no_grad():
            for step, ret in enumerate(tqdm(test_loader, desc='[VAL] Epoch '+str(epoch+1)+'/'+str(args.epochs), disable=True)):
                x = ret['x'].to(device).float()
                y = ret['y'].to(device).float()

                # forward head network
                return_value = model(x, iterations=iterations, profile=False)
                for idx, new_network in enumerate(new_networks):
                    # set tail network topology
                    model.reset_gene(new_network)
                    # forward tail network
                    output = model(x, iterations=iterations, return_value=return_value, profile=False)

                    # metrics
                    output, gt = output.detach().cpu().numpy(), y.cpu().numpy()
                    iou_score = iou(gt, output)
                    dice_score = dice_coef(gt, output)
                    val_ious[idx] += iou_score
                    val_dices[idx] += dice_score

        print('[VAL] Epoch: '+str(epoch+1)+'/'+str(args.epochs))
        for idx, val_iou in enumerate(val_ious):
            if val_iou/len(test_loader) > best_ious[idx]:
                best_ious[idx] = val_ious[idx]/len(test_loader)
            print('#'+str(idx)+': ',
              'val_iou:', val_ious[idx]/len(test_loader),
              'val_dice:', val_dices[idx]/len(test_loader),
              'best val_iou:', best_ious[idx])
    
    # compute metrics and flops
    scores = []
    for idx, new_network in enumerate(new_networks):
        print()
        macs, params = get_flops(args, model, new_network)
        scores.append([1. - best_ious[idx], macs]) # for getting the Pareto front
        print('model #'+str(idx)+': iou '+str(best_ious[idx])+', macs '+str(macs)+', params '+str(params))
    return scores
