import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA

def pca_whitening(image, number_of_pc):
    shape = image.shape
    image = np.reshape(image, [shape[0]*shape[1], shape[2]]) 
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components=number_of_pc)
    image = pca.fit_transform(image) 
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc), dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns)) 
    return pc_images

def load_data_openset(dataset):
    if dataset == '2013houston':
        image_file_HSI = r'./Houston2013/HSI.mat'
        image_file_LiDAR = r'./Houston2013/LiDAR.mat'
        label_file_tr = r'./Houston2013/TRLabel.mat'
        label_file_ts = r'./Houston2013/TSLabel.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data_tr = sio.loadmat(label_file_tr)
        label_data_ts = sio.loadmat(label_file_ts)
        image_HSI = image_data_HSI['HSI']
        image_LiDAR = image_data_LiDAR['LiDAR']
        train_label = label_data_tr['TRLabel']
        test_label = label_data_ts['TSLabel']
    elif dataset == 'Muufl':
        image_file_HSI = r'./Muufl/HSI.mat'
        image_file_LiDAR = r'./Muufl/LiDAR.mat'
        label_file_tr = r'./Muufl/TRLabel.mat'
        label_file_ts = r'./Muufl/TSLabel.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data_tr = sio.loadmat(label_file_tr)
        label_data_ts = sio.loadmat(label_file_ts)
        image_HSI = image_data_HSI['HSI']
        image_LiDAR = image_data_LiDAR['LiDAR']
        train_label = label_data_tr['TRLabel']
        test_label = label_data_ts['TSLabel']
    elif dataset == 'Augsburg':
        image_file_HSI = r'./Augsburg/HSI.mat'
        image_file_LiDAR = r'./Augsburg/LiDAR.mat'
        label_file_tr = r'./Augsburg/TRLabel.mat'
        label_file_ts = r'./Augsburg/TSLabel.mat'
        image_data_HSI = sio.loadmat(image_file_HSI)
        image_data_LiDAR = sio.loadmat(image_file_LiDAR)
        label_data_tr = sio.loadmat(label_file_tr)
        label_data_ts = sio.loadmat(label_file_ts)
        image_HSI = image_data_HSI['HSI']
        image_LiDAR = image_data_LiDAR['LiDAR']
        train_label = label_data_tr['TRLabel']
        test_label = label_data_ts['TSLabel']
    else:
        raise Exception('dataset does not find')

    image_HSI = image_HSI.astype(np.float32)
    image_LiDAR = image_LiDAR.astype(np.float32)
    train_label = train_label.astype(np.int64)
    test_label = test_label.astype(np.int64)
    return image_HSI, image_LiDAR, train_label, test_label

def readdata(type, dataset, windowsize, train_num_per_class, val_num, seed,log_print=False):

    or_image_HSI, or_image_LiDAR, or_train_label, or_test_label = load_data_openset(dataset)

    tr_classes_orig = np.unique(or_train_label)  
    tr_classes_orig = tr_classes_orig[tr_classes_orig > 0] 
    ts_classes_orig = np.unique(or_test_label)
    ts_classes_orig = ts_classes_orig[ts_classes_orig > 0] 
    
    known_classes_orig = tr_classes_orig
    unknown_classes_orig = np.setdiff1d(ts_classes_orig, tr_classes_orig) 
    n_known_classes = len(known_classes_orig)

    known_map = {orig_id: new_id for new_id, orig_id in enumerate(known_classes_orig)}  

    unknown_label_idx = n_known_classes

    halfsize = int((windowsize - 1) / 2)

    if or_image_LiDAR.ndim == 2:
        or_image_LiDAR = np.expand_dims(or_image_LiDAR, 2)  

    padded_image_hsi = np.pad(or_image_HSI, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')

    padded_image_lidar = np.pad(or_image_LiDAR, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')

    if type == 'PCA':
        image = pca_whitening(padded_image_hsi, number_of_pc=30)
    elif type == 'none':
        image = np.copy(padded_image_hsi)
    else:
        raise NotImplementedError

    image_LiDAR = np.copy(padded_image_lidar)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image_LiDAR = (image_LiDAR - np.min(image_LiDAR)) / (np.max(image_LiDAR) - np.min(image_LiDAR))
    
    train_coords, train_labels = [], []  
    val_coords, val_labels = [], []
    
    np.random.seed(seed)

    total_tr_samples = 0
    for orig_id, new_id in known_map.items():
        coords = np.argwhere(or_train_label == orig_id)
        total_class_samples = len(coords)
        total_tr_samples += total_class_samples
        np.random.shuffle(coords)
        
        train_count = min(train_num_per_class, total_class_samples)

        train_coords.extend(coords[:train_count])
        train_labels.extend([new_id] * train_count)
        
        val_coords.extend(coords[train_count:])
        val_labels.extend([new_id] * (len(coords) - train_count))
    
    val_indices = np.random.choice(len(val_coords), min(val_num, len(val_coords)), replace=False)
    final_val_coords = [val_coords[i] for i in val_indices]
    final_val_labels = [val_labels[i] for i in val_indices]
    
    def extract_patches(coords, patch_size, padded_hsi, padded_lidar):
        c_hsi, c_lidar = padded_hsi.shape[2], padded_lidar.shape[2] 
        patches_hsi = np.zeros((len(coords), patch_size, patch_size, c_hsi), dtype=np.float32)
        patches_lidar = np.zeros((len(coords), patch_size, patch_size, c_lidar), dtype=np.float32)
        half = patch_size // 2
        for i, (r, c) in enumerate(coords):
            pr, pc = r + half, c + half  
            patches_hsi[i] = padded_hsi[pr - half:pr + half + 1, pc - half:pc + half + 1]
            patches_lidar[i] = padded_lidar[pr - half:pr + half + 1, pc - half:pc + half + 1]
        return patches_hsi, patches_lidar

    train_image, train_image_LIDAR = extract_patches(train_coords, windowsize, image, image_LiDAR)
    validation_image, validation_image_LIDAR = extract_patches(final_val_coords, windowsize, image, image_LiDAR)

    train_coords_set = set(map(tuple, train_coords))  
    val_coords_set = set(map(tuple, final_val_coords))
    used_coords_set = train_coords_set.union(val_coords_set)  

    test_pixel_coords, test_pixel_labels = [], []

    all_ts_classes_with_bg = np.unique(or_test_label)
    all_ts_classes = all_ts_classes_with_bg[all_ts_classes_with_bg > 0]
    
    total_test_samples = 0
    for class_id in all_ts_classes:

        all_coords_in_class = np.argwhere(or_test_label == class_id)

        label = known_map.get(class_id, unknown_label_idx)

        fresh_coords = [c for c in all_coords_in_class if tuple(c) not in used_coords_set]
        
        num_class_test_samples = len(fresh_coords)
        total_test_samples += num_class_test_samples
        
        test_pixel_coords.extend(fresh_coords)
        test_pixel_labels.extend([label] * num_class_test_samples)


    for i, (coord, label) in enumerate(zip(test_pixel_coords[:10], test_pixel_labels[:10])):
        r, c = coord
        original_label = or_test_label[r, c]
        expected_label = known_map.get(original_label, unknown_label_idx)
        if label != expected_label:
            print(f"  sample {i}: index ({r},{c}) ord ={original_label} expect ={expected_label} actual={label}")

    label_counts = {}
    for label in test_pixel_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    index = [[] for _ in range(n_known_classes + 1)]
    for coord, label in zip(test_pixel_coords, test_pixel_labels):
        index[label].append(coord)
    index = [np.array(class_coords) for class_coords in index]

    nTrain_perClass = np.array([train_num_per_class] * n_known_classes)

    nvalid_perClass = np.bincount(final_val_labels, minlength=n_known_classes)

    s = np.count_nonzero(or_train_label)

    return (
        train_image, train_image_LIDAR, np.array(train_labels, dtype=np.int64),
        validation_image, validation_image_LIDAR, np.array(final_val_labels, dtype=np.int64),
        nTrain_perClass, nvalid_perClass,
        None, None,
        n_known_classes, index, image, image_LiDAR, or_test_label, s
    )