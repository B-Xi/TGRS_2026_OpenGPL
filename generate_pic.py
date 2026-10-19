import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
indianpines_colors = np.array([[0, 0, 0], 
                              [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                              [0, 255, 255], [255, 0, 255],[255, 128, 0],[128, 0, 255],
                              [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128],
                              [0, 128, 128], [0, 0, 128], [255, 165, 0], 
                              [255, 255, 255]]) 
indianpines_colors = sklearn.preprocessing.minmax_scale(indianpines_colors, feature_range=(0, 1))

def classification_map(img, ground_truth, dpi, save_path):

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(
        ground_truth.shape[1] * 2.0 / dpi,
        ground_truth.shape[0] * 2.0 / dpi
    )

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return 0

#openset
def get_safe_color(label_idx, colors, n_known_classes):

    if label_idx == 0:
        return colors[0] 

    if label_idx > n_known_classes:  
        return colors[-1]
        
    return colors[label_idx % (len(colors) - 1)]

def generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, OA, halfsize, dataset, day_str, num, model_name):  
    number_of_rows, number_of_columns = gt.shape  
    
    gt_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))
    predicted_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))

    n_known_classes = len(index) - 1  

    for i in range(number_of_rows):
        for j in range(number_of_columns):
            try:
                label = int(gt[i,j])
                color = get_safe_color(label, indianpines_colors, n_known_classes)
                gt_thematic_map[i, j, :] = color
                predicted_thematic_map[i, j, :] = color
            except Exception as e:
                gt_thematic_map[i, j, :] = indianpines_colors[0]
                predicted_thematic_map[i, j, :] = indianpines_colors[0]

    n_classes = len(index)  
    global_ptr = 0  
    
    for i in range(n_classes):
        matrix = index[i]
        if matrix.size == 0:
            continue
            
        m = len(matrix)
        train_samples = nTrain_perClass[i] if i < len(nTrain_perClass) else 0
        valid_samples = nvalid_perClass[i] if i < len(nvalid_perClass) else 0
        start_idx = train_samples + valid_samples
        
        for j in range(m):
            if global_ptr >= len(test_pred): 
                break
                
            row, col = int(matrix[j, 0]), int(matrix[j, 1])
            pred_label = test_pred[global_ptr]  

            if j >= start_idx:
                if pred_label == n_known_classes:
                    draw_label = n_known_classes + 1 
                else:
                    draw_label = pred_label + 1 
                    
                if row < number_of_rows and col < number_of_columns:
                    predicted_thematic_map[row, col, :] = get_safe_color(draw_label, indianpines_colors, n_known_classes)
            global_ptr += 1  
    
    h_start = min(halfsize, number_of_rows)
    h_end = max(0, number_of_rows - halfsize)
    w_start = min(halfsize, number_of_columns)
    w_end = max(0, number_of_columns - halfsize)
    
    predicted_thematic_map = predicted_thematic_map[h_start:h_end, w_start:w_end, :]
    gt_thematic_map = gt_thematic_map[h_start:h_end, w_start:w_end, :]
    
    save_dir = os.path.join('.', 'classification_maps')
    os.makedirs(save_dir, exist_ok=True)
    
    pred_path = os.path.join(save_dir, f'{dataset}_{day_str}_{num}_OA_{round(OA, 2)}_predicted.png')
    gt_path = os.path.join(save_dir, f'{dataset}_{day_str}_{num}_OA_{round(OA, 2)}_gt.png')

    classification_map(predicted_thematic_map, gt, 600, pred_path)
    classification_map(gt_thematic_map, gt, 600, gt_path)
    
    return predicted_thematic_map, gt_thematic_map

font_path = './fonts/TIMES.TTF' 

prop = fm.FontProperties(fname=font_path)

def draw_tsne(features, labels, n_known_classes, classnames, dataset_name, plot_name):
    
    print(f"tsne-->")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title('t-SNE Visualization', fontproperties=prop, fontsize=18, fontweight='bold', pad=15)

    cmap = plt.get_cmap('tab20')

    for c in range(n_known_classes):
        idx = (labels == c)
        if np.sum(idx) > 0:
            color = cmap(c % 20)
            ax.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                        c=[color], label=f'Class {c+1}', 
                        s=30, alpha=0.7, edgecolors='none')

    unknown_idx = (labels >= n_known_classes)
    if np.sum(unknown_idx) > 0:
        ax.scatter(features_2d[unknown_idx, 0], features_2d[unknown_idx, 1], 
                    c='black', label='Unknown', 
                    s=40, alpha=0.8, marker='o', edgecolors='none')

    ax.set_xlabel('Component 1', fontproperties=prop, fontsize=15, fontweight='bold')
    ax.set_ylabel('Component 2', fontproperties=prop, fontsize=15, fontweight='bold')

    for label in ax.get_xticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(12)
    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('black')

    save_dir = os.path.join('tsne_results', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'tsne_{plot_name}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"==> t-SNE saved to: {save_path}")