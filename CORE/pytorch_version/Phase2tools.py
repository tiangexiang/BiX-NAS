import os
from .metrics import *
import numpy as np
from .dataloaders import *

# gene: [types, iterations, layers, skip_list]
def decode(file, tot_iteration, tot_level=4):
    """
    decode Phase1 searched genes from txt file
    :tp: type of block, encoder/decoder
    :it: iteration time
    :level
    """
    encoder = [[None for _ in range(tot_level)] for _ in range(tot_iteration)]
    decoder = [[None for _ in range(tot_level)] for _ in range(tot_iteration)]
    with open(file, 'r') as f:
        for idx, l in enumerate(f.readlines()):
            if idx == 0:
                continue
            s = l.strip().split('_')
            tp, it, level = s[:3]
            pre = [int(v) for v in s[3:]]
            if tp == 'enc':
                if it == '0':
                    pre = [0]
                encoder[int(it)][int(level)] = pre
            else:
                decoder[int(it)][int(level)] = pre
    codes = []
    for i in range(tot_iteration):
        codes.append(encoder[i])
        codes.append(decoder[i])
    print(codes)
    return codes

def get_pareto_front(costs, return_mask = True):
    """
    find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def sample(args, codes, it, sampled):
    """
    draw a sample from the SuperNet
    """
    if it == 0:
        return [[0] for _ in range(len(codes[it]))]
    cur = codes[it]
    count = 0
    while count < 500:
        count += 1
        new_codes = []
        for l in range(len(cur)): # for each searching block
            if it == args.iter * 2 - 1: # last decoder!
                num = np.random.choice(range(0, len(cur[l])-1+1), 1, False)
                if num == 0:
                    samples = [0]
                else:
                    samples = list(np.random.choice(cur[l][1:], num, False))
                    samples = [0] + samples
            else:
                num = np.random.choice(range(1, len(cur[l])+1), 1, False)
                samples = list(np.random.choice(cur[l], num, False))
            new_codes.append(samples)

        if new_codes not in sampled:
            return new_codes            
    return None

def save_gene(folder, name, codes, iou, flops):
    """
    save genes to a file
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, name+'.txt'), 'w+') as f:
        f.write('%.3f_%f\n' % (iou, float(flops)))
        for it in range(len(codes)):
            for l in range(4): # 4 levels!!!!
                if it % 2 == 0:
                    f.write('enc')
                else:
                    f.write('dec')
                f.write('_'+str(it//2)+'_'+str(l))
                for pre in codes[it][l]:
                    f.write('_'+str(pre))
                f.write('\n')

def print_network(codes):
    print('print', codes, len(codes[0]))
    for it in range(len(codes)):
        for l in range(4): # 4 levels!!!!
            if it % 2 == 0:
                print('Encoder, iter '+str(it//2)+', level '+str(l), end=':')
            else:
                print('Decoder, iter '+str(it//2)+', level '+str(l), end=':')
            print(codes[it][l])        
    print()

def BFS(iters, codes):
    """
    check if a block is skipped or not
    BFS from the last extraction stage to the first one
    """
    new_codes = []

    pre_iter = [0, 1, 2, 3] # last block at last extraction stage
    compensate = [[] for _ in range(iters * 2)]

    for i in range(iters * 2 - 1, 0, -1): # for each extraction stage pair
        temp = {}
        temp_code = []
        for level in pre_iter: # for each non-skipped block 
            for out in codes[i][level]: # for each outgoing skip

                if i % 2== 0: # if BFS from encoder -> decoder
                    if out == 0: # if the skip is sequential 
                        if level == 0: 
                            continue
                        else:
                            pre_iter.append(level-1) 
                    else: # if not sequential
                        temp[out - 1] = 1
                        
                else: # if BFS from decoder -> encoder
                    if out == 0: # if the skip is sequential 
                        if level == 0: # if has bridge
                            temp[3] = 1 
                        else:
                            pre_iter.append(level-1)
                    else: # if not sequential
                        temp[out - 1] = 1

        for k in temp.keys():
            temp_code.append(k)
        temp_code = list(set(temp_code))
        pre_iter = temp_code[:] # to be searched in next round
        new_codes.append(pre_iter) # append non-skipped blocks at preceding stage to new_codes

    new_codes = [[0, 1, 2, 3]]+ new_codes #[:4] + [[0, 1, 2, 3]]
    new_codes = new_codes[::-1]

    skip_codes = [[True for _ in range(4)] for _ in range(iters*2)]
    for i in range(iters*2):
        new_codes[i] = list(set(new_codes[i]))
        for code in new_codes[i]: 
            skip_codes[i][code] = False
    print('These blocks as skipped: ', skip_codes)
    return skip_codes