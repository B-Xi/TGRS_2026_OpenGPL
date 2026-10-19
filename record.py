# -*- coding: utf-8 -*-
import numpy as np

def record_output(OA, AA, KAPPA, H_SCORE, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, path):
    f = open(path, 'a')
    
    f.write('Overall Accuracy (OA):\n')
    f.write(str(OA) + '\n')
    f.write(f'Mean OA: {np.mean(OA):.2f} ± {np.std(OA):.2f}\n\n')
    
    f.write('Average Accuracy (AA) on Known Classes:\n')
    f.write(str(AA) + '\n')
    f.write(f'Mean AA: {np.mean(AA):.2f} ± {np.std(AA):.2f}\n\n')

    f.write('Kappa Coefficient:\n')
    f.write(str(KAPPA) + '\n')
    f.write(f'Mean Kappa: {np.mean(KAPPA):.2f} ± {np.std(KAPPA):.2f}\n\n')

    f.write('H-Score:\n')
    f.write(str(H_SCORE) + '\n')
    f.write(f'Mean H-Score: {np.mean(H_SCORE):.2f} ± {np.std(H_SCORE):.2f}\n\n')

    f.write('Training Time:\n')
    f.write(str(TRAINING_TIME) + '\n')
    f.write(f'Mean Training Time: {np.mean(TRAINING_TIME):.2f}s\n\n')

    f.write('Testing Time:\n')
    f.write(str(TESTING_TIME) + '\n')
    f.write(f'Mean Testing Time: {np.mean(TESTING_TIME):.2f}s\n\n')

    f.write('Per-class Accuracy:\n')
    f.write(str(ELEMENT_ACC) + '\n')
    element_mean = np.mean(ELEMENT_ACC, axis=0)
    element_std = np.std(ELEMENT_ACC, axis=0)
    for i, (mean, std) in enumerate(zip(element_mean, element_std)):
        f.write(f'  - Class {i}: {mean:.2f} ± {std:.2f}\n')
        
    f.close()