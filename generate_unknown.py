import numpy as np
import torch
import random
from typing import Optional
import torch.nn.functional as F
from tqdm import tqdm

class _ActNorm1D(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, dim))
        self.log_scale = torch.nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if not reverse:
            y = (x + self.bias) * torch.exp(self.log_scale)
            log_det = self.log_scale.sum() * torch.ones(x.size(0), device=x.device)
            return y, log_det
        else:
            y = x * torch.exp(-self.log_scale) - self.bias
            log_det = -self.log_scale.sum() * torch.ones(x.size(0), device=x.device)
            return y, log_det

class _Inv1x1(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        w_init, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.weight = torch.nn.Parameter(w_init)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if not reverse:
            y = x @ self.weight.T
            log_det = torch.slogdet(self.weight)[1] * torch.ones(x.size(0), device=x.device)
            return y, log_det
        else:
            inv_w = torch.inverse(self.weight)
            y = x @ inv_w.T
            log_det = -torch.slogdet(self.weight)[1] * torch.ones(x.size(0), device=x.device)
            return y, log_det

class _AffineCoupling(torch.nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x: torch.Tensor, reverse: bool = False):
        x_m = x * self.mask
        s_t = self.net(x_m)
        s, t = s_t.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y = x_m + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = ((1 - self.mask) * s).sum(dim=1)
            return y, log_det
        else:
            y = x_m + (1 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -((1 - self.mask) * s).sum(dim=1)
            return y, log_det

class DenseFlow1D(torch.nn.Module):
    def __init__(self, dim: int, layers: int = 6, hidden: int = 256):
        super().__init__()
        masks = []
        for i in range(layers):
            m = torch.zeros(dim)
            if i % 2 == 0:
                m[: dim // 2] = 1.0
            else:
                m[dim // 2 :] = 1.0
            masks.append(m)
        self.blocks = torch.nn.ModuleList([
            _ActNorm1D(dim), _Inv1x1(dim), _AffineCoupling(dim, hidden, masks[0])
        ])

        for i in range(1, layers):
            self.blocks.extend([_ActNorm1D(dim), _Inv1x1(dim), _AffineCoupling(dim, hidden, masks[i])])
        self.dim = dim

    def f(self, x: torch.Tensor):
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for b in self.blocks:
            z, ld = b(z, reverse=False)
            log_det = log_det + ld
        return z, log_det

    def f_inv(self, z: torch.Tensor):
        log_det = torch.zeros(z.size(0), device=z.device)
        x = z
        for b in reversed(self.blocks):
            x, ld = b(x, reverse=True)
            log_det = log_det + ld
        return x, log_det

    def log_prob(self, x: torch.Tensor):
        z, ld = self.f(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return log_pz + ld

    def sample(self, n: int, device):
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.f_inv(z)
        return x

def _build_feature_flow(dim: int, args):
    layers = getattr(args, 'flow_layers', 6)
    hidden = getattr(args, 'flow_hidden', 256)
    return DenseFlow1D(dim, layers=layers, hidden=hidden)

def _tokenize_known(classnames):
    from util_CNN_clip import tokenize
    ctx = "a multimodal fusion patch of a"
    names = [n.replace('_', ' ') for n in classnames[:-1]]
    prompts = [ctx + " " + n + "." for n in names]
    return torch.cat([tokenize(p) for p in prompts])

@torch.no_grad()
def _collect_features(model, images, lidars):
    device = next(model.parameters()).device
    N = images.shape[0]
    img_b = torch.from_numpy(images).float().to(device)
    lid_b = torch.from_numpy(lidars).float().to(device)
    t = torch.zeros(N, dtype=torch.long, device=device)
    if hasattr(model, 'encode_image'):
        #features
        feats = model.encode_image(img_b, lid_b, t)
    else:
        feats, _ = model.get_features(img_b, lid_b, t, _tokenize_known(['tmp','Unknown']).to(device))
        feats = feats
    return feats.detach() 


def _logits_from_features(model, feats: torch.Tensor, tokenized: torch.Tensor):
    device = feats.device
    if hasattr(model, 'encode_text'):
        text_feats = model.encode_text(tokenized.to(device))
    else:
        return torch.zeros(feats.size(0), tokenized.size(0), device=device)
    feats_n = feats / feats.norm(dim=-1, keepdim=True)
    text_n = text_feats / text_feats.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else torch.ones([], device=device)
    return logit_scale * feats_n @ text_n.t()

def Normalizing_Flow(train_image, train_image_LIDAR, train_label, n_known_classes, args, classifier_model, classnames, flow: Optional[torch.nn.Module]=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    known_mask = train_label < n_known_classes
    if known_mask.sum() == 0:
        return (np.zeros((0, *train_image.shape[1:])), np.zeros((0, *train_image_LIDAR.shape[1:])),
                np.array([], dtype=np.int64), torch.zeros((0, 1), device=device), flow)

    feats = _collect_features(classifier_model, train_image[known_mask], train_image_LIDAR[known_mask]) 
    feat_dim = feats.shape[1]

    if flow is None:
        flow = _build_feature_flow(feat_dim, args).to(device)
        opt = torch.optim.Adam(flow.parameters(), lr=getattr(args, 'flow_lr', 1e-4))
        flow.train()
        n = feats.size(0)
        batch_size_flow = getattr(args, 'flow_batch_size', 256)
        epochs = getattr(args, 'flow_epochs', 5)
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size_flow):
                idx = perm[i:i+batch_size_flow]
                b = feats[idx]  
                nll = -flow.log_prob(b).mean()
                opt.zero_grad()
                nll.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                opt.step()
        flow.eval()

    token_known = _tokenize_known(classnames).to(device)

    total_needed = n_known_classes * getattr(args, 'pseudo_per_class', 5)
    kept = []
    tries = 0
    with torch.no_grad(): 
        ref_logits = _logits_from_features(classifier_model, feats.to(device), token_known)
        ref_probs = torch.softmax(ref_logits[:, :n_known_classes], dim=1)
        ref_entropy = -(ref_probs * (ref_probs.clamp_min(1e-9)).log()).sum(dim=1)
        ent_thr = torch.quantile(ref_entropy, getattr(args, 'flow_ref_quantile', 0.8))

        max_tries = getattr(args, 'flow_sample_max_tries', 2000)
        batch_z = getattr(args, 'flow_sample_batch', 128)
        while len(kept) < total_needed and tries < max_tries:
            tries += 1
            z = torch.randn(min(batch_z, total_needed - len(kept)), feat_dim, device=device)
            cand = flow.f_inv(z)[0]  
            logits = _logits_from_features(classifier_model, cand, token_known) 
            probs = torch.softmax(logits[:, :n_known_classes], dim=1)
            entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=1)
            max_known = probs.max(dim=1)[0]
            mask = (entropy > 0.6 * ent_thr) & (max_known < getattr(args, 'flow_max_known_thresh', 0.85))
            if mask.any():
                take = cand[mask]
                n_take = min(take.size(0), total_needed - len(kept))
                kept.append(take[:n_take]) 

    if len(kept) == 0:
        z = torch.randn(total_needed, feat_dim, device=device)
        kept_feats = flow.f_inv(z)[0]  
    else:
        kept_feats = torch.cat(kept, dim=0)[:total_needed] 

    empty_imgs = np.zeros((0, *train_image.shape[1:]), dtype=np.float32)
    empty_lids = np.zeros((0, *train_image_LIDAR.shape[1:]), dtype=np.float32)
    pseudo_labels = np.full(kept_feats.size(0), n_known_classes, dtype=np.int64)

    return empty_imgs, empty_lids, pseudo_labels, kept_feats, flow

def joint_train_flow_classifier(train_image, train_image_LIDAR, train_label, n_known_classes, model, classnames, args):

    device = next(model.parameters()).device
    known_mask = train_label < n_known_classes
    feats = _collect_features(model, train_image[known_mask], train_image_LIDAR[known_mask]) 
    feat_dim = feats.shape[1]
    flow = _build_feature_flow(feat_dim, args).to(device)

    opt_flow = torch.optim.Adam(flow.parameters(), lr=getattr(args, 'flow_lr', 1e-4), weight_decay=1e-5)
    opt_cls = torch.optim.Adam(model.parameters(), lr=getattr(args, 'flow_classifier_lr', 1e-4), weight_decay=1e-6)

    sup_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(train_image).float(),
        torch.from_numpy(train_image_LIDAR).float(),
        torch.from_numpy(train_label).long(),
    )
    sup_loader = torch.utils.data.DataLoader(sup_ds, batch_size=max(32, getattr(args, 'flow_batch_size', 256)//4), shuffle=True)

    token_known = _tokenize_known(classnames).to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    jsd_lambda = getattr(args, 'flow_jsd_lambda', 0.5)

    n_vec = feats.size(0)
    vec_batch = getattr(args, 'flow_batch_size', 256)
    flow_epochs = getattr(args, 'flow_joint_epochs', 5)

    flow.train();model.train()
    for _ in range(flow_epochs):
        vec_perm = torch.randperm(n_vec, device=device)
        sup_iter = iter(sup_loader)

        for i in range(0, n_vec, vec_batch):

            model.requires_grad_(False); flow.requires_grad_(True)
            idx = vec_perm[i:i+vec_batch]
            vb = feats[idx]   
            nll = -flow.log_prob(vb).mean()
            opt_flow.zero_grad(); nll.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0); opt_flow.step()
            model.requires_grad_(True); flow.requires_grad_(False)
            try:
                img_b, lid_b, lbl_b = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_loader)
                img_b, lid_b, lbl_b = next(sup_iter)

            img_b = img_b.to(device); lid_b = lid_b.to(device); lbl_b = lbl_b.to(device)
            t = torch.zeros(img_b.size(0), dtype=torch.long, device=device)
            logits_real, _ = model(img_b, lid_b, t, token_known)
            if isinstance(logits_real, tuple): logits_real = logits_real[-1]
            sup_loss = ce_loss(logits_real, lbl_b)

            with torch.no_grad():
                z = torch.randn(max(64, vec_batch//2), feat_dim, device=device)
                flow_feats = flow.f_inv(z)[0]

            logits_flow = _logits_from_features(model, flow_feats, token_known)
            probs_flow = torch.softmax(logits_flow[:, :n_known_classes], dim=1)
            uniform = torch.ones_like(probs_flow) / n_known_classes

            # strict JSD
            m = 0.5 * (probs_flow + uniform)
            kl_p_m = (probs_flow * (probs_flow / m).clamp_min(1e-9).log()).sum(dim=1)
            kl_u_m = (uniform * (uniform / m).clamp_min(1e-9).log()).sum(dim=1)
            jsd = 0.5 * (kl_p_m + kl_u_m).mean()

            total = sup_loss + jsd_lambda * jsd
            opt_cls.zero_grad(); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_cls.step()

    flow.eval()
    return flow

def calculate_class_centroids(images, labels, n_classes, device='cuda'):
    centroids = []
    print("   Calculating class centroids for quality filtering...")
    with torch.no_grad():
        for i in range(n_classes):
            class_mask = (torch.tensor(labels) == i)
            class_images = torch.tensor(images[class_mask], dtype=torch.float32).to(device)
            if len(class_images) > 0:
                centroid = class_images.mean(dim=0)
                centroids.append(centroid)
            else:
                centroids.append(torch.zeros_like(torch.tensor(images[0], dtype=torch.float32)).to(device))
    return torch.stack(centroids)

def calculate_pixel_centroids(images, labels, n_classes, device):
    centroids = []
    if not torch.is_tensor(images):
        images_t = torch.tensor(images).float().to(device)
    else:
        images_t = images.float().to(device)
        
    for c in range(n_classes):
        indices = np.where(labels == c)[0]
        if len(indices) == 0:
            centroids.append(torch.zeros_like(images_t[0]))
            continue
        class_imgs = images_t[indices]
        centroid = class_imgs.mean(dim=0)
        centroids.append(centroid)
    
    return torch.stack(centroids)  

def compute_decision_boundary_proximity(samples, centroids, top_k=2):

    B = samples.shape[0]
    K = centroids.shape[0]

    samples_flat = samples.reshape(B, -1)  
    centroids_flat = centroids.reshape(K, -1)  

    dists = torch.cdist(samples_flat, centroids_flat, p=2)

    topk_dists, _ = torch.topk(dists, k=min(top_k, K), largest=False, dim=1)
    
    boundary_score = topk_dists.var(dim=1) 
    
    return boundary_score

def normalize_score(x, inverse=False):

    x = np.array(x)
    normalized = (x - np.mean(x)) / (np.std(x) + 1e-9)
    return -normalized if inverse else normalized

def generate_diffusion_pseudo_unknowns(net, diffusion, train_image, train_image_LIDAR, train_label,
                                       n_known_classes, pseudo_per_class=5, timestep=600,
                                       mask_ratio=0.7, batch_size=32,
                                       oversampling_factor=4.0, device='cuda'):
    
    final_total_pseudo = n_known_classes * pseudo_per_class
    candidate_per_class = int(np.ceil(pseudo_per_class * oversampling_factor))
    
    # print(f"   Target: {pseudo_per_class} samples/class × {n_known_classes} classes = {final_total_pseudo}")
    # print(f"   Will generate {candidate_per_class} candidates/class (oversampling {oversampling_factor}x)")
    
    net.to(device)
    net.eval()

    class_centroids = calculate_pixel_centroids(train_image, train_label, n_known_classes, device)

    all_candidates_hsi = []
    all_candidates_lidar = []
    all_metrics = {
        'entropy': [],
        'distance': [],
        'boundary': [],
        'source_class': [] 
    }

    stages = [
        {
            'name': 'Stage 1',
            't_range': (100, 300),
            'label_strategy': 'original', 
            'weight': 0.3,  
            'description': ''
        },
        {
            'name': 'Stage 2',
            't_range': (400, 700),
            'label_strategy': 'shifted',
            'weight': 0.5,
            'description': ''
        },
        {
            'name': 'Stage 3',
            't_range': (750, 950),
            'label_strategy': 'random',
            'weight': 0.2, 
            'description': ''
        }
    ]
    
    with torch.no_grad():
        for stage_idx, stage_config in enumerate(stages):

            stage_total = int(candidate_per_class * n_known_classes * stage_config['weight'])
            samples_per_class = max(1, stage_total // n_known_classes)

            sample_indices = []
            for class_id in range(n_known_classes):
                class_indices = np.where(train_label == class_id)[0]
                if len(class_indices) == 0:
                    continue
                
                if len(class_indices) >= samples_per_class:
                    selected = np.random.choice(class_indices, samples_per_class, replace=False)
                else:
                    selected = np.random.choice(class_indices, samples_per_class, replace=True)
                sample_indices.extend(selected)
            
            sample_indices = np.array(sample_indices)
            np.random.shuffle(sample_indices)

            t_min, t_max = stage_config['t_range']
            pbar = tqdm(range(0, len(sample_indices), batch_size), 
                       desc=f"        Generating", leave=False)
            
            for i in pbar:
                batch_indices = sample_indices[i:i + batch_size]
                B = len(batch_indices)
                if B == 0:
                    continue

                x = torch.tensor(train_image[batch_indices]).float().to(device)
                x_lidar = torch.tensor(train_image_LIDAR[batch_indices]).float().to(device)
                y_real = torch.tensor(train_label[batch_indices]).long().to(device)

                if stage_config['label_strategy'] == 'original':
                    y_cond = y_real
                elif stage_config['label_strategy'] == 'shifted':
                    shift = torch.randint(1, n_known_classes, (B,)).to(device)
                    y_cond = (y_real + shift) % n_known_classes
                else:  # random
                    y_cond = torch.randint(0, n_known_classes, (B,)).to(device)

                t = torch.randint(t_min, t_max, (B,), dtype=torch.long).to(device)
                
                # noise injection for diffusion
                noise = torch.randn_like(x)
                noise_lidar = torch.randn_like(x_lidar)
                x_t = diffusion.q_sample(x, t, noise)
                x_lidar_t = diffusion.q_sample(x_lidar, t, noise_lidar)
                
                # MAE
                model_kwargs = dict(y=y_cond, mask_ratio=mask_ratio)
                pred_hsi, pred_lidar, logits, _, _ = net(x_t, x_lidar_t, t, **model_kwargs)
                
                # compute metrics for filtering

                # 1.entropy (uncertainty)
                if logits is not None:
                    probs = F.softmax(logits[:, :n_known_classes], dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
                else:
                    entropy = torch.zeros(B).to(device)
                
                # 2. distance to nearest known class centroid
                diff = pred_hsi.unsqueeze(1) - class_centroids.unsqueeze(0) 
                dist_sq = torch.sum(diff ** 2, dim=(2, 3, 4))  # (B, K)
                min_dist, _ = torch.min(torch.sqrt(dist_sq), dim=1)  # (B,)
                
                # 3. decision boundary proximity
                boundary_prox = compute_decision_boundary_proximity(pred_hsi, class_centroids, top_k=2)
                
                # record candidates and metrics
                all_candidates_hsi.append(pred_hsi.cpu())
                all_candidates_lidar.append(pred_lidar.cpu())
                all_metrics['entropy'].append(entropy.cpu())
                all_metrics['distance'].append(min_dist.cpu())
                all_metrics['boundary'].append(boundary_prox.cpu())
                all_metrics['source_class'].append(y_real.cpu())
    
    # combine all candidates and metrics
    candidates_hsi = torch.cat(all_candidates_hsi, dim=0)
    candidates_lidar = torch.cat(all_candidates_lidar, dim=0)
    
    entropy_all = torch.cat(all_metrics['entropy']).numpy()
    distance_all = torch.cat(all_metrics['distance']).numpy()
    boundary_all = torch.cat(all_metrics['boundary']).numpy()
    source_class_all = torch.cat(all_metrics['source_class']).numpy()

    # filter

    # 1. Apply quality thresholds

    max_possible_ent = np.log(n_known_classes)
    valid_mask = (entropy_all > 0.3 * max_possible_ent) & (entropy_all < 0.95 * max_possible_ent)

    dist_threshold = np.percentile(distance_all, 25)  
    valid_mask &= (distance_all > dist_threshold)
    
    if valid_mask.sum() == 0:
        print("        [WARNING] No candidates passed quality filter! Using all.")
        valid_mask = np.ones(len(entropy_all), dtype=bool)
    
    # 2. normalize metrics for fair combination
    entropy_norm = normalize_score(entropy_all[valid_mask], inverse=False)  
    distance_norm = normalize_score(distance_all[valid_mask], inverse=False)  
    boundary_norm = normalize_score(boundary_all[valid_mask], inverse=True)  
    
    # 3. score combination with weights
    # design rationale:
    # - Entropy (1.0): core metric
    # - Distance (0.8): important for ensuring novelty, but not as critical as entropy
    # - Boundary (0.6): useful for diversity, but less critical than the other two
    w_ent, w_dist, w_bound = 1.0, 0.8, 0.6
    
    composite_score = (w_ent * entropy_norm + 
                      w_dist * distance_norm + 
                      w_bound * boundary_norm)
    
    # 4. Balance across source classes to avoid over-representation
    source_class_valid = source_class_all[valid_mask]

    class_counts = np.bincount(source_class_valid, minlength=n_known_classes)

    max_count = class_counts.max()
    for i in range(len(composite_score)):
        src_cls = source_class_valid[i]
        if class_counts[src_cls] > max_count * 0.5: 
            penalty = 0.9  
            composite_score[i] *= penalty
    
    # 5. Top-K
    valid_indices = np.where(valid_mask)[0]
    
    num_valid = len(valid_indices)
    
    if num_valid < final_total_pseudo:
        print(f" Insufficient valid candidates ({num_valid}/{final_total_pseudo}). Padding data...")
        
        # A: sample with replacement from valid candidates to fill up
        if num_valid > 0:

            num_missing = final_total_pseudo - num_valid

            extra_indices = np.random.choice(valid_indices, num_missing, replace=True)

            final_indices = np.concatenate([valid_indices, extra_indices])
            
        # B: If no valid samples , not only alert but also fall back to using all samples
        else:
            print(" No valid samples. Fallback to top-score unfiltered candidates.")

            fallback_indices = np.arange(len(candidates_hsi))
            np.random.shuffle(fallback_indices)
            final_indices = fallback_indices[:final_total_pseudo]
            
    else:
        # normal case
        sorted_idx = np.argsort(-composite_score)  
        selected_idx = sorted_idx[:final_total_pseudo]
        final_indices = valid_indices[selected_idx]
    
    # final selection
    pseudo_unknown_samples = candidates_hsi[final_indices].numpy()
    pseudo_unknown_samples_lidar = candidates_lidar[final_indices].numpy()
    pseudo_unknown_labels = np.full(len(pseudo_unknown_samples), n_known_classes, dtype=np.int64)

    return pseudo_unknown_samples, pseudo_unknown_samples_lidar, pseudo_unknown_labels

def generate_pseudo_unknowns(strategy, net, diffusion, train_image, train_image_LIDAR, train_label,
                            n_known_classes, args, mask_ratio=0.7, batch_size=32,model=None, trained_flow=None,
                            classifier_model=None, classnames=None, flow=None):
    strategy = strategy.lower()
    if args.dataset == '2013houston':
        oversample=20
    elif args.dataset == 'Muufl':
        oversample=2     
    elif args.dataset == 'Augsburg':
        oversample=11
    if strategy == 'diffusion':
        return generate_diffusion_pseudo_unknowns(
            net, diffusion, train_image, train_image_LIDAR, train_label,
            n_known_classes, getattr(args, 'pseudo_per_class', 5), getattr(args, 'diffusion_timestep', 600),
            mask_ratio, batch_size,oversampling_factor=oversample 
        )
    else:
        raise ValueError(f"Unknown pseudo-unknown generation strategy: {strategy}")    
    