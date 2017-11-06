'''
Created on Nov 2, 2017

@author: rpalyam
'''

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dist = np.load('distances.npy') # efficient was to save a label indicating pair, non-pair while calculating distances :(
    sorted_dist = np.sort(dist)
    
    print 'Min threshold: {} and Max threshold: {}'.format(min(sorted_dist), max(sorted_dist))
    
    pairs_file = '/home/rpalyam/Documents/Tutworks/test/src/pairs_ed.txt'

    frr_all = []
    far_all = []
    cardinal = 3000.0
    with open(pairs_file, 'r') as fl_pairs:
        lines = fl_pairs.readlines() # Read the pairs file
        # Loop for each threshold in [min,max]
        for i in range(len(sorted_dist)):
            far = 0.0
            frr = 0.0
            # Loop for each line and increment FAR or FRR count
            for j, line in enumerate(lines, 0):
                content = line.split()
                if len(content) == 3:
                    if dist[j] > sorted_dist[i]:
                        frr +=1.0
                elif len(content) == 4:
                    if dist[j] <= sorted_dist[i]:
                        far +=1.0
            # Calculate FAR and FRR ratio for particular threshold
            far_ratio = float(far/cardinal)
            frr_ratio = float(frr/cardinal)
            frr_all = np.append(frr_all, frr_ratio)
            far_all = np.append(far_all, far_ratio)
            print sorted_dist[i], far_ratio, frr_ratio
    diff_ratios = abs(far_all - frr_all)
    opt_idx = np.argmin(diff_ratios)
    opt_thres = sorted_dist[opt_idx]
    eer = (far_all[opt_idx]+frr_all[opt_idx]) / 2
    print eer, opt_thres
    
    plt.plot(far_all,frr_all)
    plt.show()
    
    eer_all = (far_all + frr_all) / 2
    plt.plot(sorted_dist,eer_all)
    plt.show()