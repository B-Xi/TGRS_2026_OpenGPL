from scipy.io import savemat
import os
seed = 42 
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
import record
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import datetime
import numpy as np
import os
import random
from config import load_args
from data_read import readdata
from diffusion import create_diffusion
from generate_pic import generate
from hyper_dataset import HyperData
from augment import CenterResizeCrop
from util_CNN_clip import text_valid, text_target, fine_tune, test_batch, calculate_energy_scores
from pos_embed import interpolate_pos_embed
from torch.nn.init import trunc_normal_
from model import mae_vit_HSIandLIDAR_patch3, vit_HSI_LIDAR_patch3, DDPM_LOSS
from clip_model import build_model
from collections import OrderedDict
from generate_unknown import generate_pseudo_unknowns, joint_train_flow_classifier, Normalizing_Flow
try:
    from sklearn.mixture import GaussianMixture
except Exception:
    GaussianMixture = None
from simple_tokenizer import SimpleTokenizer as _Tokenizer

def read_classnames(text_file):
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames

def set_random_seed(seed_value: int = 42, use_seed: bool = True):
    import os, random, numpy as np, torch
    if not use_seed:
        print("Using non-deterministic mode")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return

    print(f"Setting seed = {seed_value}")

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("torch.use_deterministic_algorithms not available:", e)

args = load_args()
mask_ratio = args.mask_ratio
windowsize = args.windowsize
dataset = args.dataset
type = args.type
num_epoch = args.epochs
num_fine_tuned =args.fine_tuned_epochs
lr = args.lr
train_num_per = args.train_num_perclass
train_num_per_two=args.train_num_perclass_two
batch_size= args.batch_size

num_of_ex = 10

net_name = 'LDS2AE'
day = datetime.datetime.now() 
day_str = day.strftime('%m_%d_%H_%M')  
halfsize = int((windowsize-1)/2)
val_num = 1000

_, _, _, _, _,_,_,_, _,_, n_known_classes, _,_, _, gt, s = readdata(type, dataset, windowsize,train_num_per, val_num, 0,log_print=args.log_print)
num_of_samples = int(s * 0.2)
nclass = n_known_classes + 1 

if args.dataset == 'Muufl':
    in_chans_LIDAR = 2
    train_num_per=100
elif args.dataset == 'Houston2018':
    train_num_per=35
    in_chans_LIDAR = 1
else:
    in_chans_LIDAR = 1
    train_num_per=45

if args.dataset == 'Muufl':
    num_unknown_classes=5
elif args.dataset == '2013houston':
    num_unknown_classes=1
elif args.dataset == 'Augsburg':
    num_unknown_classes=2
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")
size = 1

seed=[0,1,4,9,41] #'41'is my lucky number
KAPPA = []
OA = []
AA = []
H_SCORE = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((len(seed), nclass))
af_result = np.zeros([nclass+3, len(seed)])

criterion = nn.CrossEntropyLoss()
print(f"Train on {args.dataset}")
num_of_ex = 10

pretrained_model_path = f'./pretrained_models/net_{args.dataset}.pt'
os.makedirs('./pretrained_models', exist_ok=True)


train_image, train_image_LIDAR, train_label, _, _, _, nTrain_perClass, _, \
    train_index, _, n_known_classes, index, image, image_LiDAR, gt,s = readdata(type, dataset, windowsize,train_num_per,num_of_samples,0,log_print=args.log_print)

nband = train_image.shape[3]

train_num = train_image.shape[0] 
train_image = np.transpose(train_image,(0,3,1,2))
train_image_LIDAR = np.transpose(train_image_LIDAR,(0,3,1,2))


if args.augment:

    transform_train = [CenterResizeCrop(scale_begin = args.scale, windowsize = windowsize)]
    untrain_dataset = HyperData((train_image, train_image_LIDAR, train_label), transform_train)
else:
    untrain_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_image_LIDAR), torch.tensor(train_label))
untrain_loader = DataLoader(dataset = untrain_dataset, batch_size = batch_size, shuffle = True)

print("=> creating model '{}'".format(net_name))

# pre-train
diffusion = create_diffusion(timestep_respacing="1000")
net = mae_vit_HSIandLIDAR_patch3(img_size=(windowsize,windowsize), in_chans=nband, in_chans_LIDAR=in_chans_LIDAR, hid_chans = args.hid_chans, hid_chans_LIDAR = args.hid_chans_LIDAR, embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,  mlp_ratio=args.mlp_ratio,
                        decoder_embed_dim=args.decoder_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, nb_classes=nclass, global_pool=False)

net.cuda()

if not os.path.exists(pretrained_model_path):
    print(f"Pre-trained model not found at {pretrained_model_path}. Starting Pre-training...")
    optimizer = optim.Adam(net.parameters(),lr = lr, weight_decay= 1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5,T_mult=2)

    tic1 = time.time()
    for epoch in range(num_epoch):
        net.train()
        total_loss = 0
        for idx, (x, x_LIDAR, y) in enumerate(untrain_loader):

            x = x.cuda()
            x_LIDAR = x_LIDAR.cuda()
            y = y.cuda()
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],)).cuda()
            model_kwargs = dict(y=y)
            model_kwargs['mask_ratio'] = mask_ratio

            # Method 1 of forward process
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            noise_LIDAR = torch.randn_like(x_LIDAR)
            x_LIDAR_t = diffusion.q_sample(x_LIDAR, t, noise_LIDAR)

            pred_imgs, pred_imgs_LIDAR, logits, mask, mask_LIDAR = net(x_t, x_LIDAR_t, t, **model_kwargs)

            cls_loss = criterion(logits / args.temperature, y)
            loss_mse_m, loss_mse_LIDAR_m, loss_mse_v, loss_mse_LIDAR_v = DDPM_LOSS(pred_imgs, pred_imgs_LIDAR, x, x_LIDAR, mask, mask_LIDAR, size)
            lambda_cls = 0 
            #removing classification loss
            loss =  0.1 * (loss_mse_m + loss_mse_LIDAR_m) + loss_mse_v + loss_mse_LIDAR_v + lambda_cls * cls_loss

            optimizer.zero_grad()                             
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss

        scheduler.step()
        total_loss = total_loss/(idx+1)
        state = {'model':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}       
        if epoch % 10 == 0:
            print('epoch:',epoch,'loss:',total_loss.data.cpu().numpy())

    toc1 = time.time()
    #torch.save(state, './net.pt') 
    torch.save(state, pretrained_model_path)
    print(f"Pre-trained model saved to {pretrained_model_path}")   
    del net, optimizer
    torch.cuda.empty_cache()
else:
    print(f"Found pre-trained model at {pretrained_model_path}. Skipping pre-training phase.")
    tic1 = time.time()
    toc1 = time.time()
for num in range(len(seed)):

    print('num:', num)    
    set_random_seed(seed[num], use_seed=True)
    print('Seed:', seed[num])

    print(f'\n=== Starting Experiment {num+1}/{len(seed)} (Seed: {num}) ===')

    diffusion = create_diffusion(timestep_respacing="1000")
    net = mae_vit_HSIandLIDAR_patch3(img_size=(windowsize,windowsize), in_chans=nband, in_chans_LIDAR=in_chans_LIDAR, hid_chans = args.hid_chans, hid_chans_LIDAR = args.hid_chans_LIDAR, embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,  mlp_ratio=args.mlp_ratio,
                            decoder_embed_dim=args.decoder_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, nb_classes=nclass, global_pool=False)

    net.cuda()
    checkpoint = torch.load(pretrained_model_path)

    net.load_state_dict(checkpoint['model'], strict=False) 
    print(f"=> Loaded pre-trained weights from {pretrained_model_path}")

        # pesudo unknown sample generation
    if args.use_pseudo_unknown:
        print("\npesudo unknown sample generation...")
        pseudo_unknown_samples, pseudo_unknown_samples_lidar, pseudo_unknown_labels = generate_pseudo_unknowns(
            strategy=args.pseudo_strategy,
            net=net,
            diffusion=diffusion,
            train_image=train_image,
            train_image_LIDAR=train_image_LIDAR,
            train_label=train_label,
            n_known_classes=n_known_classes,
            args=args,
            mask_ratio=mask_ratio,
            batch_size=batch_size
        )
    else:
        print("\npesudo unknown sample generation... skipped (use_pseudo_unknown=False)")
        pseudo_unknown_samples = np.array([]).reshape(0, *train_image.shape[1:])
        pseudo_unknown_samples_lidar = np.array([]).reshape(0, *train_image_LIDAR.shape[1:])
        pseudo_unknown_labels = np.array([], dtype=np.int64)

    print("\nStart Finetune")
    # finetune
    train_image, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR, validation_label, nTrain_perClass, nvalid_perClass, \
        train_index, val_index, n_known_classes, index, image, image_LiDAR, gt,s = readdata(type, dataset, windowsize,train_num_per_two,200,num,log_print=args.log_print)

    nband = train_image.shape[3]
    train_num = train_image.shape[0]
    train_image = np.transpose(train_image,(0,3,1,2))
    train_image_LIDAR = np.transpose(train_image_LIDAR,(0,3,1,2))
    validation_image = np.transpose(validation_image,(0,3,1,2))
    validation_image_LIDAR = np.transpose(validation_image_LIDAR,(0,3,1,2))

    train_image = train_image.astype(np.float32)
    train_image_LIDAR = train_image_LIDAR.astype(np.float32)

    try:
        validation_image = validation_image.astype(np.float32)
        validation_image_LIDAR = validation_image_LIDAR.astype(np.float32)
    except NameError:
        pass

    if args.augment:
        transform_train = [CenterResizeCrop(scale_begin = args.scale, windowsize = windowsize)]

        untrain_dataset = TensorDataset(
        torch.tensor(train_image, dtype=torch.float32),
        torch.tensor(train_image_LIDAR, dtype=torch.float32),
        torch.tensor(train_label, dtype=torch.long)
    )
    else:
        untrain_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_image_LIDAR), torch.tensor(train_label))
    untrain_loader = DataLoader(dataset = untrain_dataset, batch_size = batch_size, shuffle = True)

    if args.dataset == '2013houston':
        classnames_file = "./classnames_houston.txt"
    elif args.dataset == 'Trento':
        classnames_file = "./classnames_trento.txt"
    elif args.dataset == 'Muufl':
        classnames_file = "./classnames_muufl.txt"
    elif args.dataset == 'Augsburg':
        classnames_file = "./classnames_augsburg.txt"
    elif args.dataset == 'Houston2018':
        classnames_file = "./classnames_houston2018.txt"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    classnames_dict = read_classnames(classnames_file)
    classnames = list(classnames_dict.values())

    model = build_model(img_size=(windowsize, windowsize), in_chans=nband, in_chans_LIDAR=in_chans_LIDAR,
                        hid_chans=args.hid_chans, hid_chans_LIDAR=args.hid_chans_LIDAR,
                        embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,
                        mlp_ratio=args.mlp_ratio, num_classes=nclass, global_pool=False).cuda()
    
    tic2 = time.time()
    num_known_classes = len(classnames) - num_unknown_classes
    classnames = classnames[:num_known_classes]
    classnames.append("Unknown")

    # feature-level flow negatives
    flow_feat_neg = None
    trained_flow = None
    
    if args.ues_flow:
        # Step1: Joint-train flow + classifier 
        try:
            print("Running Step1: Joint training Flow + Classifier...")
            
            trained_flow = joint_train_flow_classifier(train_image, train_image_LIDAR, train_label,n_known_classes, model, classnames, args)
            if trained_flow is not None:
                trained_flow.eval()

            if trained_flow is not None and getattr(args, 'save_trained_flow', False):
                try:
                    os.makedirs('saved_flow', exist_ok=True)
                    torch.save(trained_flow.state_dict(), os.path.join('saved_flow', f'flow_{day_str}.pth'))

                except Exception as e:
                    print(f"WARN: failed to save trained_flow: {e}")
            print("=> Step1 joint training finished.")
        except Exception as e:
            print(f"WARN: joint_train_flow_classifier failed: {e}. Proceeding without a trained flow.")
            trained_flow = None

        # Step2: Generate feature-level flow negatives
        try:
            print("Running Step2: Generating feature-level flow negatives...")
            ret = Normalizing_Flow(
                train_image, train_image_LIDAR, train_label, num_known_classes, args, model.eval(), classnames, flow=trained_flow
            )
            if isinstance(ret, (tuple, list)):
                if len(ret) == 5:
                    _, _, _, flow_feat_neg, returned_flow = ret
                    if trained_flow is None and returned_flow is not None:
                        trained_flow = returned_flow
                elif len(ret) == 4:
                    _, _, _, flow_feat_neg = ret
                else:
                    raise ValueError(f"Unexpected return length from generate_flow_pseudo_unknowns_feature: {len(ret)}")
            else:
                raise ValueError("generate_flow_pseudo_unknowns_feature must return a tuple/list")

            if isinstance(flow_feat_neg, np.ndarray):
                flow_feat_neg = torch.from_numpy(flow_feat_neg)
            if not isinstance(flow_feat_neg, torch.Tensor):
                raise ValueError("flow_feat_neg must be a torch.Tensor or numpy array convertible to it")

        except Exception as e:
            print(f"WARN: feature-level flow generation failed: {e}")
            flow_feat_neg = None

    # Short OE fine-tune
    if flow_feat_neg is not None and isinstance(flow_feat_neg, torch.Tensor) and flow_feat_neg.numel() > 0:
        model.train()
        device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        token_full = text_valid(classnames).to(device)
        ce_loss_local = nn.CrossEntropyLoss()
        optimizer_oe = optim.Adam(model.parameters(), lr=max(5e-5, args.fine_tuned_lr*0.5), weight_decay=1e-5)

        bs_img = batch_size
        bs_flow = getattr(args, 'flow_feature_bs', 128)
        epochs_oe = getattr(args, 'flow_feature_oe_epochs', 2)
        unknown_index = num_known_classes
        for ep in range(epochs_oe):
            with torch.no_grad():
                text_feats = model.encode_text(token_full)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats.detach()

            perm_flow = torch.randperm(flow_feat_neg.size(0))
            ptr_flow = 0

            img_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_image_LIDAR), torch.tensor(train_label))
            img_loader = DataLoader(img_dataset, batch_size=bs_img, shuffle=True)
            for img_b, lid_b, lbl_b in img_loader:
                img_b = img_b.to(device).float(); lid_b = lid_b.to(device).float(); lbl_b = lbl_b.to(device).long()

                t_b = torch.zeros(img_b.size(0), dtype=torch.long, device=device)
                logits_img, _ = model(img_b, lid_b, t_b, token_full)
                if isinstance(logits_img, tuple): logits_img = logits_img[-1]
                loss_img = ce_loss_local(logits_img, lbl_b)

                end = min(ptr_flow + bs_flow, flow_feat_neg.size(0))
                feats_b = flow_feat_neg[perm_flow[ptr_flow:end]].to(device)
                ptr_flow = end if end < flow_feat_neg.size(0) else 0

                img_feats = feats_b / feats_b.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits_flow = logit_scale * img_feats @ text_feats.t()
                labels_unk = torch.full((logits_flow.size(0),), unknown_index, dtype=torch.long, device=device)
                loss_unk = ce_loss_local(logits_flow, labels_unk)

                probs_k = torch.softmax(logits_flow[:, :unknown_index], dim=1)
                uniform = torch.ones_like(probs_k) / unknown_index
                m = 0.5 * (probs_k + uniform)
                kl_p_m = (probs_k * (probs_k / m).clamp_min(1e-9).log()).sum(dim=1)
                kl_u_m = (uniform * (uniform / m).clamp_min(1e-9).log()).sum(dim=1)
                jsd = 0.5 * (kl_p_m + kl_u_m).mean()

                loss = loss_img + loss_unk 
                optimizer_oe.zero_grad()
                loss.backward()
                optimizer_oe.step()
        print("Flow's two-steps training completed.")
    else:
        print(" No feature-level flow negatives available; skipping short OE fine-tune.")

    print(f" Integrating pseudo-unknown samples into training data...")
    unknown_label_index = n_known_classes 
    if len(pseudo_unknown_samples) > 0:
        pseudo_unknown_labels = np.full((pseudo_unknown_samples.shape[0],), unknown_label_index, dtype=np.int64)
        if train_image.shape[1:] != pseudo_unknown_samples.shape[1:]:
            print(f"   WARNING: Shape mismatch detected!")
            print(f"   train_image shape: {train_image.shape}")
            print(f"   pseudo_unknown_samples shape: {pseudo_unknown_samples.shape}")
        pseudo_unknown_samples = pseudo_unknown_samples.astype(np.float32)
        pseudo_unknown_samples_lidar = pseudo_unknown_samples_lidar.astype(np.float32)

        enhanced_train_image = np.concatenate([train_image, pseudo_unknown_samples], axis=0)
        enhanced_train_image_LIDAR = np.concatenate([train_image_LIDAR, pseudo_unknown_samples_lidar], axis=0)
        enhanced_train_label = np.concatenate([train_label, pseudo_unknown_labels], axis=0)

    else:
        enhanced_train_image = train_image
        enhanced_train_image_LIDAR = train_image_LIDAR
        enhanced_train_label = train_label
        print(f"   Using only original training samples: {len(enhanced_train_image)}")
    assert np.sum(enhanced_train_label == unknown_label_index) == pseudo_unknown_samples.shape[0]

    optimizer = optim.Adam(model.parameters(), lr = args.fine_tuned_lr, weight_decay= 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [100, 131], gamma = 0.1, last_epoch=-1)

    model = fine_tune(model, enhanced_train_image, enhanced_train_image_LIDAR, enhanced_train_label,
                validation_image, validation_image_LIDAR, validation_label,
                num_fine_tuned, optimizer, scheduler, batch_size, diffusion, classnames, val = False,
                use_learned_prompt=args.use_learned_prompt,use_Neg_prompt=args.use_Neg_prompt)

    toc2 = time.time()
    state_finetune = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}

    if os.path.exists('Best_val_model/net_params.pkl'):
        print("=> loading pre-trained model...")
        model.load_state_dict(torch.load('Best_val_model/net_params.pkl'))
    
    # openset
    val_image_for_energy = np.transpose(validation_image, (0, 2, 3, 1))
    val_lidar_for_energy = np.transpose(validation_image_LIDAR, (0, 2, 3, 1))
    val_energy_scores = calculate_energy_scores(model.eval(), val_image_for_energy, val_lidar_for_energy, halfsize, batch_size, diffusion, classnames)

    val_mean = np.mean(val_energy_scores)
    val_std = np.std(val_energy_scores)
    validation_size = len(val_energy_scores)

    is_trento_like = (validation_size <= 300) and (val_mean < -11.0)
    # energy_threshold = np.percentile(val_energy_scores, 95)
    if args.dataset == 'Trento' :
        energy_threshold = np.percentile(val_energy_scores, 98)
    elif args.dataset == '2013houston':
        energy_threshold = np.percentile(val_energy_scores, 97.5)
    elif args.dataset == 'Muufl':
        energy_threshold = np.percentile(val_energy_scores, 97)
    elif args.dataset == 'Augsburg':
        energy_threshold = np.percentile(val_energy_scores, 94.7)
    else:
        energy_threshold = np.percentile(val_energy_scores, 95)
    use_greater_than = True

    # test
    true_cla, overall_accuracy, average_accuracy, kappa, h_score, true_label, test_pred, test_index, cm, pred_array = \
        test_batch(model.eval(), image, image_LiDAR, index, batch_size, nTrain_perClass, nvalid_perClass, halfsize, diffusion, classnames, energy_threshold, use_greater_than,
                    use_learned_prompt=args.use_learned_prompt,use_Neg_prompt=args.use_Neg_prompt)

    toc3 = time.time()

    af_result[:nclass,num] = true_cla
    af_result[nclass,num] = overall_accuracy
    af_result[nclass+1,num] = average_accuracy
    af_result[nclass+2,num] = kappa

    OA.append(overall_accuracy)
    AA.append(average_accuracy)
    KAPPA.append(kappa)
    H_SCORE.append(h_score)
    TRAINING_TIME.append(toc1 - tic1 + toc2 - tic2)
    TESTING_TIME.append(toc3 - toc2)
    ELEMENT_ACC[num, :] = true_cla
    classification_map, gt_map = generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, overall_accuracy, halfsize, dataset, day_str, num, net_name)

    dataset=args.dataset
    result_dir = os.path.join('result', dataset)
    os.makedirs(result_dir, exist_ok=True)
    file_name = os.path.join(result_dir, f'40_11_0.7_p3{overall_accuracy}.mat')
    savemat(file_name, {'map': classification_map})

    # save
    model_dir = os.path.join('model', dataset)
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f'40_11_0.7_p3{overall_accuracy}net.pt')
    torch.save(state_finetune, model_file)

result = np.mean(af_result, axis = 1)
print(result)
print(" " + net_name + " Training Finished")

# finishi & print
if num_of_ex >= 0:
    print("\n\n" + "="*25 + " All Experiments Summary " + "="*25)
    print(f"Finished {num_of_ex} experiments, here are the average performance metrics and standard deviations:\n")

    ELEMENT_ACC = np.array(ELEMENT_ACC)
    OA = np.array(OA)
    AA = np.array(AA)
    KAPPA = np.array(KAPPA)
    H_SCORE = np.array(H_SCORE)
    TRAINING_TIME = np.array(TRAINING_TIME)
    TESTING_TIME = np.array(TESTING_TIME)

    mean_element_acc = np.mean(ELEMENT_ACC, axis=0)
    std_element_acc = np.std(ELEMENT_ACC, axis=0)

    print("Per-Class Accuracy")
    for i in range(len(mean_element_acc)):
        if i < n_known_classes:
            class_name = classnames[i] if i < len(classnames) else f"Known Class {i}"
            is_unknown_str = ""
        else:
            class_name = "unknown"
            is_unknown_str = " (Unknown Class)"
        print(f"  - Class {class_name}{is_unknown_str}: {mean_element_acc[i]:.2f}% ± {std_element_acc[i]:.2f}%")
    print("-" * 70)

    print("\n--- Overall Metrics ---")
    print(f"  - OA:            {np.mean(OA):.2f}% ± {np.std(OA):.2f}%")
    print(f"  - AA:      {np.mean(AA):.2f}% ± {np.std(AA):.2f}%")
    print(f"  - Kappa:         {np.mean(KAPPA):.2f}% ± {np.std(KAPPA):.2f}%")
    print(f"  - H-Score:       {np.mean(H_SCORE):.2f}% ± {np.std(H_SCORE):.2f}%")
    print("-" * 70)

    print("\n--- Average Time ---")
    print(f"  - Average Training Time: {np.mean(TRAINING_TIME):.2f}s")
    print(f"  - Average Testing Time: {np.mean(TESTING_TIME):.2f}s")
    print("=" * 75 + "\n")

record.record_output(OA, AA, KAPPA, H_SCORE, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                    'records/' + net_name + '_' + dataset + '_' + str(train_num_per) + 'shot.txt')
