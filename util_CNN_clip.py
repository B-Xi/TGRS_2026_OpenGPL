import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import tempfile
import shutil
import packaging
from simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
ce_loss = nn.CrossEntropyLoss()

def entropy_from_probs(p, eps=1e-12):
    p_clamped = p.clamp(min=eps, max=1.0)
    return -torch.sum(p_clamped * torch.log(p_clamped), dim=-1)

def tokenize(texts, context_length: int = 77, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def make_class_unknown_flags(classnames, device=None):

    n = len(classnames)
    flags = torch.zeros(n, dtype=torch.float32)
    flags[-1] = 1.0
    if device is not None:
        flags = flags.to(device)
    return flags

def labels_to_unknown_flag(labels, unknown_class_idx):

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    return (labels == unknown_class_idx).float()

def freeze_all_but_prompts(model,use_neg_prompt=False):

    orig_req = {}
    for name, p in model.named_parameters():
        orig_req[name] = p.requires_grad
        if use_neg_prompt:
            if 'prompt_context' in name or 'unknown_prompt_context' in name or 'negative_prompt_context' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            if 'prompt_context' in name or 'unknown_prompt_context' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
    return orig_req

def restore_requires_grad(model, orig_req):
    for name, p in model.named_parameters():
        if name in orig_req:
            p.requires_grad = orig_req[name]
        else:
            p.requires_grad = True

# openset
def text_valid(classnames):
    ctx_init_3 = "a multimodal fusion patch of a"
    prompts = []
    for name in classnames[:-1]: # Exclude "unknown"
        prompts.append(ctx_init_3 + " " + name.replace("_", " ") + ".")
    prompts.append("a multimodal fusion patch of an ambiguous unidentified region.")

    return torch.cat([tokenize(p) for p in prompts])

def text_target(classnames, batch_target):

    ctx_init_3 = "a multimodal fusion patch of a"
    known_classnames = classnames[:-1]  # Exclude "unknown"
    unknown_class_idx = len(known_classnames)  # unknown
    
    prompts = []
    for target in batch_target:
        if target < unknown_class_idx:
            # known class
            prompts.append(ctx_init_3 + " " + known_classnames[target].replace("_", " ") + ".")
        else:
            # unknown class
            prompts.append("a multimodal fusion patch of an ambiguous unidentified region.")
    
    return torch.cat([tokenize(p) for p in prompts])

def tr_acc(model, image, image_LIDAR, label, diffusion, classnames, 
           use_learned_prompt, use_Neg_prompt, 
           use_visual_prompt=False, prototypes=None): 

    if isinstance(image, torch.Tensor): image = image.cpu().numpy()
    if isinstance(image_LIDAR, torch.Tensor): image_LIDAR = image_LIDAR.cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.cpu().numpy()

    dataset = TensorDataset(torch.from_numpy(image.copy()).float(),
                            torch.from_numpy(image_LIDAR.copy()).float(),
                            torch.from_numpy(label.copy()))
    
    loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

    corr_num = 0
    total_loss = 0
    model.eval()
    
    device = next(model.parameters()).device

    with torch.no_grad():

        pass 

        for i, (img_batch, lidar_batch, lbl_batch) in enumerate(loader):
            img_batch, lidar_batch, lbl_batch = img_batch.to(device), lidar_batch.to(device), lbl_batch.to(device)
            t = torch.zeros(img_batch.shape[0], dtype=torch.long).to(device)

            text_prompts = text_valid(classnames).to(device)

            forward_kwargs = {}
            if use_learned_prompt:
                unknown_flag_text = make_class_unknown_flags(classnames, device=device)
                forward_kwargs['unknown_flag'] = unknown_flag_text

            if use_visual_prompt and prototypes is not None:
                forward_kwargs['class_prototypes'] = prototypes

            logits, _ = model(img_batch, lidar_batch, t, text_prompts, **forward_kwargs)

            if isinstance(logits, tuple): logits = logits[-1]
            
            pred = torch.max(logits, dim=1)[1]
            loss = ce_loss(logits, lbl_batch)
            total_loss += loss.item()
            corr_num += torch.eq(pred, lbl_batch).float().sum().item()
            
    return corr_num / len(dataset), total_loss / (i + 1)

def fine_tune(model, train_image, train_image_LIDAR, train_label, validation_image, validation_image_LIDAR,
            validation_label, epoch, optimizer, scheduler, bs, diffusion, classnames, val = False,
            use_learned_prompt:bool=False,
            use_Neg_prompt:bool=False,
            do_prompt_tuning: bool = True,          
            prompt_tuning_steps: int = 200,         
            prompt_tuning_lr: float = 1e-3,):       
    train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_image_LIDAR), torch.tensor(train_label))
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    
    Train_loss = []
    Train_acc = []
    Val_loss = []
    Val_acc = []
    BestAcc = 0
    unknown_idx = len(classnames) - 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_tokens = text_valid(classnames).to(device)                
    class_unknown_flags = make_class_unknown_flags(classnames, device=device)

    if do_prompt_tuning and use_learned_prompt:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"==> Prompt-only warm-up: {prompt_tuning_steps}")
        # 1) freeze all parameters except prompt-related ones
        orig_req = freeze_all_but_prompts(model,use_Neg_prompt)

        # 2) create optimizer for prompt parameters only
        pos_params = []
        unk_params = []
        neg_params = []

        if hasattr(model, 'prompt_context'):
            pos_params.append(model.prompt_context)

        if hasattr(model, 'unknown_prompt_context'):
            unk_params.append(model.unknown_prompt_context)

        if use_Neg_prompt and hasattr(model, 'negative_prompt_context'):
            neg_params.append(model.negative_prompt_context)    
        
        param_groups = []

        if len(pos_params) > 0 or len(unk_params) > 0:
            group_params = []
            group_params.extend(pos_params)
            group_params.extend(unk_params)
            param_groups.append({'params': group_params, 'lr': prompt_tuning_lr})

        if len(neg_params) > 0:
            param_groups.append({'params': neg_params, 'lr': prompt_tuning_lr * 0.1})

        if not param_groups:
            raise ValueError("No prompt parameters found to optimize. Check if your model has 'prompt_context' or other prompt attributes.")

        prompt_optimizer = torch.optim.AdamW(param_groups, lr=prompt_tuning_lr, weight_decay=0.0)

        # 3) compute class-level text features (prototypes) for the current prompt
        class_tokens = text_valid(classnames).to(device)
        class_unknown_flags = make_class_unknown_flags(classnames, device=device)
        
        # 4) prompt tuning loop
        model.train()
        it = 0
        loader_iter = iter(train_loader)
        while it < prompt_tuning_steps:
            try:
                image_batch, image_LIDAR_batch, label_batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                image_batch, image_LIDAR_batch, label_batch = next(loader_iter)

            image_batch = image_batch.cuda()
            image_LIDAR_batch = image_LIDAR_batch.cuda()
            label_batch = label_batch.cuda()

            class_text_feats = model.encode_text(class_tokens, use_learned_prompt=True, unknown_flag=class_unknown_flags)
            class_text_feats = class_text_feats / (class_text_feats.norm(dim=-1, keepdim=True) + 1e-12)

            if use_Neg_prompt:
                neg_class_text_feats = model.encode_negative_text(class_tokens)
                neg_class_text_feats = neg_class_text_feats / (neg_class_text_feats.norm(dim=-1, keepdim=True) + 1e-12)
                text_semantic_loss = (class_text_feats * neg_class_text_feats).sum(dim=1).mean()

            t = torch.randint(0, diffusion.num_timesteps, (image_batch.shape[0],)).cuda()

            image_feats = model.encode_image(image_batch, image_LIDAR_batch, t)
            image_feats = image_feats / (image_feats.norm(dim=-1, keepdim=True) + 1e-12)

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (image_feats @ class_text_feats.t())  

            loss = ce_loss(logits, label_batch)

            if use_Neg_prompt:

                neg_class_text_feats = model.encode_negative_text(class_tokens)
                neg_class_text_feats = neg_class_text_feats / (neg_class_text_feats.norm(dim=-1, keepdim=True) + 1e-12)

                it_frac = min(1.0, it / max(1, prompt_tuning_steps))
                lambda_rank = 1.0 * it_frac     
                lambda_textneg = 0.2 * it_frac   

                # 1. Margin Ranking Loss
                pos_feats_for_batch = class_text_feats[label_batch]
                neg_feats_for_batch = neg_class_text_feats[label_batch]
                s_pos = (image_feats * pos_feats_for_batch).sum(dim=1)
                s_neg = (image_feats * neg_feats_for_batch).sum(dim=1)
                margin = 0.5
                rank_loss = torch.relu(margin - s_pos + s_neg).mean()

                # 2. Text Semantic Loss
                text_semantic_loss = (class_text_feats * neg_class_text_feats).sum(dim=1).mean()

                loss = loss + lambda_rank * rank_loss + lambda_textneg * text_semantic_loss

            reg_loss = 0.0
            if use_Neg_prompt and hasattr(model, 'negative_prompt_context'):
                # mean vector of shared prompt vs negative prompt
                pos_mean = model.prompt_context.mean(dim=0)        
                neg_mean = model.negative_prompt_context.view(-1, model.prompt_dim).mean(dim=0) 
                ortho = torch.sum(pos_mean * neg_mean)
                reg_loss += 1e-2 * ortho.abs()  

                neg_norm = model.negative_prompt_context.norm()
                reg_loss += 1e-3 * (neg_norm ** 2)

            if reg_loss != 0.0:
                loss = loss + reg_loss


            prompt_optimizer.zero_grad()
            loss.backward()  
            prompt_optimizer.step()

            if (it + 1) % 100 == 0:
                print(f"  prompt warmup step {it+1}/{prompt_tuning_steps}, loss={loss.item():.4f}")
            it += 1

        # 5) restore original requires_grad states for all parameters
        restore_requires_grad(model, orig_req)
        print("==> Prompt-only warm-up done. Restored requires_grad states.")
    # ---------- end prompt-only warm-up ----------

    for i in range(epoch):
        model.train()
        train_loss = 0
        for idx, (image_batch, image_LIDAR_batch, label_batch) in enumerate(train_loader):
            trans_image_batch = image_batch.cuda()
            unknown_idx = len(classnames) - 1
            trans_image_LIDAR_batch = image_LIDAR_batch.cuda()
            label_batch = label_batch.cuda()
            text = text_target(classnames, label_batch).cuda()
            t = torch.randint(0, diffusion.num_timesteps, (trans_image_batch.shape[0],)).cuda()

            if use_learned_prompt:
                try:
                    unknown_flag_batch = labels_to_unknown_flag(label_batch, unknown_idx).cuda()
                except Exception:
                    unknown_flag_batch = (label_batch == unknown_idx).cuda()
                logits_per_image_x, logits_per_text = model(trans_image_batch, trans_image_LIDAR_batch, t,
                                            text,  unknown_flag=unknown_flag_batch)
            else:
                logits_per_image_x, logits_per_text = model(trans_image_batch, trans_image_LIDAR_batch, t, text)
            
            labels = torch.arange(len(logits_per_image_x)).to(logits_per_image_x.device)
            ce_loss_im = ce_loss(logits_per_image_x, labels)
            ce_loss_txt = ce_loss(logits_per_text, labels)
            total_loss = ce_loss_im + ce_loss_txt 

            if use_Neg_prompt:
                # 1) compute class-level text features (prototypes) for the current prompt
                class_tokens = text_valid(classnames).to(device) 
                class_unknown_flags = make_class_unknown_flags(classnames, device=device)
                class_text_feats = model.encode_text(class_tokens, use_learned_prompt=True, unknown_flag=class_unknown_flags)
                class_text_feats = class_text_feats / (class_text_feats.norm(dim=-1, keepdim=True) + 1e-12)

                # 2) negative text feats
                neg_class_text_feats = model.encode_negative_text(class_tokens)
                neg_class_text_feats = neg_class_text_feats / (neg_class_text_feats.norm(dim=-1, keepdim=True) + 1e-12)

                # 3) batch text features for positive and negative classes
                pos_feats_for_batch = class_text_feats[label_batch]     
                neg_feats_for_batch = neg_class_text_feats[label_batch]  

                # 4) image features
                image_feats = model.encode_image(trans_image_batch, trans_image_LIDAR_batch, t)
                image_feats = image_feats / (image_feats.norm(dim=-1, keepdim=True) + 1e-12)

                # 5) s_pos / s_neg
                s_pos = (image_feats * pos_feats_for_batch).sum(dim=1)
                s_neg = (image_feats * neg_feats_for_batch).sum(dim=1)

                # 6) margin ranking loss: push s_pos > s_neg + margin
                margin = 0.5
                rank_loss = torch.relu(margin - s_pos + s_neg).mean()

                # 7) class-level text semantic-opposite loss (push class_text_feats and neg_class_text_feats apart)
                text_semantic_loss = (class_text_feats * neg_class_text_feats).sum(dim=1).mean()

                lambda_rank = 1.0     
                lambda_textneg = 0.1     
                total_loss = total_loss + lambda_rank * rank_loss + lambda_textneg * text_semantic_loss

                diversity_loss = 0.0
                if use_Neg_prompt and hasattr(model, 'negative_prompt_context'):
                    # model.negative_prompt_context shape: [K, L, D]
                    neg_ctx = model.negative_prompt_context
                    K = neg_ctx.shape[0]
                    
                    if K > 1:
                        neg_vecs = neg_ctx.mean(dim=1)
                        
                        neg_vecs_norm = neg_vecs / (neg_vecs.norm(dim=-1, keepdim=True) + 1e-12)

                        sim_matrix = torch.mm(neg_vecs_norm, neg_vecs_norm.t())

                        off_diag_sum = sim_matrix.sum() - K
                        diversity_loss = off_diag_sum / (K * (K - 1))

                    total_loss = total_loss + 0.1 * diversity_loss

                    dummy_unknown_token = text_valid([classnames[-1]]).to(device)
                    unknown_flag_tensor = torch.ones(1, dtype=torch.long, device=device)
                    unknown_text_feat = model.encode_text(dummy_unknown_token, use_learned_prompt=True, unknown_flag=unknown_flag_tensor)
                    unknown_text_feat_norm = unknown_text_feat / (unknown_text_feat.norm(dim=-1, keepdim=True) + 1e-12)

                    s_unknown_for_batch = image_feats @ unknown_text_feat_norm.t() 

                    margin_unk = 0.3

                    unknown_rejection_loss = torch.relu(margin_unk - s_pos.unsqueeze(1) + s_unknown_for_batch).mean()

                    lambda_unk_reject = 0.5
                    total_loss = total_loss + lambda_unk_reject * unknown_rejection_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss = train_loss + total_loss.detach()

        scheduler.step()
        train_loss = train_loss / (idx + 1)
        train_acc, tr_loss = tr_acc(model.eval(), train_image, train_image_LIDAR, train_label, diffusion, classnames,use_learned_prompt,use_Neg_prompt)
        val_acc, val_loss = tr_acc(model.eval(), validation_image, validation_image_LIDAR, validation_label, diffusion,
                                classnames,use_learned_prompt,use_Neg_prompt)
        if val_acc > BestAcc:

            save_dir = 'Best_val_model'
            os.makedirs(save_dir, exist_ok=True) 
            final_path = os.path.join(save_dir, 'net_params.pkl')

            tmp_fd, tmp_path = tempfile.mkstemp(prefix='tmp_net_', suffix='.pkl', dir=save_dir)
            os.close(tmp_fd)
            try:
                torch.save(model.state_dict(), tmp_path)
                try:
                    os.replace(tmp_path, final_path)
                except Exception as e:
                    ts_name = os.path.join(save_dir, f'net_params_{int(time.time())}.pkl')
                    shutil.move(tmp_path, ts_name)
                    print(f'WARN: os.replace failed ({e}), fallback saved to {ts_name}')
            except Exception as e:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                raise e

            BestAcc = val_acc
        if i%10==0:
            print("epoch {}, training loss: {:.4f}, train acc:{:.4f}, valid acc:{:.4f}".format(i, train_loss.item(),train_acc * 100,val_acc * 100))

        if val:
            Train_loss.append(tr_loss)
            Val_loss.append(val_loss)
            Train_acc.append(train_acc)
            Val_acc.append(val_acc)
    if val:
        return model, [Train_loss, Train_acc, Val_loss, Val_acc]
    else:
        return model

# openset
def calculate_energy_scores(model, image, image_LIDAR, halfsize, BATCH_SIZE, diffusion, classnames):

    n_known_classes = len(classnames) - 1
    windowsize = 2 * halfsize + 1
    
    num_samples = image.shape[0]
    ind = np.arange(num_samples)
    
    padded_ind = ind
    if num_samples % BATCH_SIZE != 0:
        add_num = BATCH_SIZE - num_samples % BATCH_SIZE
        padded_ind = np.concatenate((ind, ind[np.random.choice(range(num_samples), add_num)]), axis=0)

    logit_array = np.zeros([padded_ind.shape[0], len(classnames)], dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for i in range(padded_ind.shape[0] // BATCH_SIZE):
            batch_indices = padded_ind[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            
            image_batch = image[batch_indices]
            lidar_batch = image_LIDAR[batch_indices]

            image_b = torch.from_numpy(np.transpose(image_batch, (0, 3, 1, 2))).cuda()
            lidar_b = torch.from_numpy(np.transpose(lidar_batch, (0, 3, 1, 2))).cuda()
            t = torch.zeros(image_b.shape[0], dtype=torch.long).cuda()

            text = text_valid(classnames).cuda()

            unknown_flag_text = make_class_unknown_flags(classnames, device=text.device)
            logits, _ = model(image_b, lidar_b, t, text, unknown_flag=unknown_flag_text)

            if isinstance(logits, tuple): logits = logits[-1]
            logit_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = logits.cpu().numpy()

    logit_array = logit_array[:num_samples]
    
    T = 1.0
    energy_scores = -T * np.log(np.sum(np.exp(logit_array[:, :n_known_classes] / T), axis=1))
    return energy_scores

# test
def test_batch(model, image, image_LIDAR, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize, diffusion, classnames, energy_threshold, use_greater_than=True,
                use_learned_prompt=False,use_Neg_prompt=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nclass = len(classnames)

    n_known_classes = len(index) - 1

    all_coords = []
    true_labels = []

    for i in range(n_known_classes + 1):
        coords = index[i]
        if coords.shape[0] == 0: continue
        all_coords.append(coords)
        true_labels.extend([i] * coords.shape[0])

    if not all_coords:
        print("warning: no coordinates found for any class in test_batch; returning empty results.")
        return [0]*nclass, 0, 0, 0, 0, np.array([]), np.array([]), np.array([]), np.zeros((nclass,nclass)), np.array([])
    
    ind = np.concatenate(all_coords, axis=0)
    true_label = np.array(true_labels, dtype=np.int32)
    test_index = np.copy(ind)
    length = ind.shape[0]

    padded_ind = ind
    if length % BATCH_SIZE != 0:
        add_num = BATCH_SIZE - (length % BATCH_SIZE)
        add_indices = np.random.choice(range(length), add_num, replace=True)
        padded_ind = np.concatenate((ind, ind[add_indices]), axis=0)

    pred_array = np.zeros([padded_ind.shape[0], nclass], dtype=np.float32)
    logit_array = np.zeros([padded_ind.shape[0], nclass], dtype=np.float32) 
    n_batches = padded_ind.shape[0] // BATCH_SIZE
    windowsize = 2 * halfsize + 1
    class_tokens = text_valid(classnames).to(device)
    if use_learned_prompt:

        class_unknown_flags = make_class_unknown_flags(classnames, device=device)

        with torch.no_grad():
            class_text_feats = model.encode_text(class_tokens, use_learned_prompt=True, unknown_flag=class_unknown_flags)
            class_text_feats = class_text_feats / (class_text_feats.norm(dim=-1, keepdim=True) + 1e-12) 
            if use_Neg_prompt:

                neg_class_text_feats = model.encode_negative_text(class_tokens)
                neg_class_text_feats = neg_class_text_feats / (neg_class_text_feats.norm(dim=-1, keepdim=True) + 1e-12)
    else:

        class_text_feats = None
        neg_class_text_feats = None
    
    s_pos_max_array = np.zeros([padded_ind.shape[0]], dtype=np.float32)
    
    s_neg_max_array = np.zeros([padded_ind.shape[0]], dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            batch_indices = padded_ind[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            image_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image.shape[2]], dtype=np.float32)
            lidar_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image_LIDAR.shape[2]], dtype=np.float32)
            
            for j, m in enumerate(batch_indices):
                image_batch[j] = image[m[0]:m[0]+windowsize, m[1]:m[1]+windowsize, :]
                lidar_batch[j] = image_LIDAR[m[0]:m[0]+windowsize, m[1]:m[1]+windowsize, :]
            
            image_b = torch.from_numpy(np.transpose(image_batch, (0, 3, 1, 2))).cuda()
            lidar_b = torch.from_numpy(np.transpose(lidar_batch, (0, 3, 1, 2))).cuda()
            
            t = torch.zeros(image_b.shape[0], dtype=torch.long).cuda()
            text = text_valid(classnames).cuda()

            from config import load_args

            if  use_learned_prompt:
                unknown_flag_text = make_class_unknown_flags(classnames, device=text.device)
                logits, _ = model(image_b, lidar_b, t, text,  unknown_flag=unknown_flag_text)
            else:
                logits, _ = model(image_b, lidar_b, t, text)
            if isinstance(logits, tuple): logits = logits[-1]

            if use_learned_prompt and use_Neg_prompt:
                # compute image features for this batch (B, D)
                image_feats = model.encode_image(image_b, lidar_b, t)   # make sure this returns (B, D)
                image_feats = image_feats / (image_feats.norm(dim=-1, keepdim=True) + 1e-12)

                # s_pos: (B, C)  = image_feats @ class_text_feats.T
                s_pos = image_feats @ class_text_feats.t()
                # s_neg: (B, C)
                s_neg = image_feats @ neg_class_text_feats.t()

                # we store per-sample the max over known classes (exclude the last 'unknown' class)
                n_known = len(classnames) - 1
                # clamp in case C==1 etc
                if n_known > 0:
                    s_pos_max_batch = s_pos[:, :n_known].max(dim=1).values.cpu().numpy()
                    s_neg_max_batch = s_neg[:, :n_known].max(dim=1).values.cpu().numpy()
                else:
                    s_pos_max_batch = s_pos.max(dim=1).values.cpu().numpy()
                    s_neg_max_batch = s_neg.max(dim=1).values.cpu().numpy()

                s_pos_max_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = s_pos_max_batch
                s_neg_max_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = s_neg_max_batch
            else:

                s_pos_max_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = 0.0
                s_neg_max_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = 0.0

            pred_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(logits, dim=1).cpu().numpy()
            logit_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = logits.cpu().numpy()

            pred_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(logits, dim=1).cpu().numpy()
            logit_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = logits.cpu().numpy()

    pred_array = pred_array[:length]
    logit_array = logit_array[:length]

    # two-stage OSR strategy: energy-based + unknown prompt-based
    # 1) Energy-based detection
    T = 1.0
    energy_scores = -T * np.log(np.sum(np.exp(logit_array[:, :n_known_classes] / T), axis=1))

    if use_greater_than:
        is_unknown_by_energy = energy_scores > energy_threshold
    else:
        is_unknown_by_energy = energy_scores < energy_threshold


    # 2) Unknown prompt-based detection
    prob_unknown = pred_array[:, n_known_classes]
    max_prob_known = np.max(pred_array[:, :n_known_classes], axis=1)

    pk = pred_array[:, :n_known_classes]
    pk_sum = np.clip(np.sum(pk, axis=1, keepdims=True), 1e-12, 1.0)
    pk_norm = pk / pk_sum
    entropy_known = -np.sum(np.clip(pk_norm, 1e-12, 1.0) * np.log(np.clip(pk_norm, 1e-12, 1.0)), axis=1)
    max_entropy = np.log(max(2, n_known_classes))
    normalized_entropy = np.clip(entropy_known / max_entropy, 0.0, 1.0)


    alpha = 0.9 - 0.4 * normalized_entropy            
    beta = 0.20 + 0.50 * normalized_entropy          

    margin = 0.05 * (np.abs(energy_threshold) + 1e-6)
    near_boundary = np.abs(energy_scores - energy_threshold) <= margin
    alpha = np.where(near_boundary, np.clip(alpha + 0.05, 0.0, 1.0), alpha)

    is_unknown_by_prompt = (prob_unknown > (alpha * np.clip(max_prob_known, 1e-12, 1.0))) | (prob_unknown > beta)

    
    if use_Neg_prompt:
        s_pos_max_arr = s_pos_max_array[:length]
        s_neg_max_arr = s_neg_max_array[:length]

        s_pos = s_pos_max_arr[:length]
        s_neg = s_neg_max_arr[:length]
        # 1) s_diff = s_neg - s_pos  
        s_diff = s_neg - s_pos 

        # 2) normalize s_diff to z-score
        mu = s_diff.mean() if np.std(s_diff) > 0 else 0.0
        sigma = s_diff.std() if s_diff.std() > 1e-12 else 1.0
        s_diff_z = (s_diff - mu) / sigma

        # 3) estimate combined score: energy + alpha * s_diff_z
        alpha_neg = 0.5 
        combined_score = energy_scores + alpha_neg * s_diff_z

        # 4) final unknown decision based on combined score
        is_unknown = combined_score > energy_threshold

    else:
        is_unknown = is_unknown_by_energy | is_unknown_by_prompt

    predict_label = np.argmax(pred_array[:, :n_known_classes], axis=1)
    predict_label[is_unknown] = n_known_classes

    cm = metrics.confusion_matrix(true_label, predict_label, labels=range(n_known_classes + 1))
    per_class_acc = np.diag(cm) / (np.sum(cm, axis=1) + 1e-9)
    overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    average_accuracy_known = np.mean(per_class_acc[:n_known_classes])

    average_accuracy_all = np.mean(per_class_acc) 

    average_accuracy = average_accuracy_all 
    
    po = overall_accuracy
    pe = np.sum(np.sum(cm, axis=1) * np.sum(cm, axis=0)) / (np.sum(cm)**2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    acc_known = np.mean(per_class_acc[:n_known_classes])
    recall_unknown = per_class_acc[n_known_classes] if n_known_classes < len(per_class_acc) else 0.0
    h_score = 2 * (acc_known * recall_unknown) / (acc_known + recall_unknown) if (acc_known + recall_unknown) > 0 else 0

    unknown_class_index = n_known_classes
    true_unknown_indices = np.where(true_label == unknown_class_index)[0]
    true_known_indices = np.where(true_label != unknown_class_index)[0]

    print(f'OA: {overall_accuracy*100:.2f}% | AA : {average_accuracy*100:.2f}% | Kappa: {kappa*100:.2f} | H-Score: {h_score*100:.2f}%')
    print(f'  - Known Acc: {acc_known*100:.2f}%, Unknown Recall: {recall_unknown*100:.2f}%')
    
    # results summary
    print("\n" + "="*20 + " results summary " + "="*20)
    print("  - Per-Class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        class_name = classnames[i] if i < len(classnames) else f"unknown {i}"
        is_unknown_str = " (unknown)" if i == n_known_classes else ""
        print(f"    - Class {class_name}{is_unknown_str}: {acc*100:.2f}%")
    print("-" * 50)

    # === Final Summary Output ===
    print(f'Final Results: OA: {overall_accuracy*100:.2f}% | AA: {average_accuracy*100:.2f}% | Kappa: {kappa*100:.2f} | H-Score: {h_score*100:.2f}%')
    #print(f'  Detailed Metrics: AA: {average_accuracy*100:.2f}%, Known Acc: {acc_known*100:.2f}%, Unknown Recall: {recall_unknown*100:.2f}%')
    print(f'Detailed Metrics:  Known Acc: {acc_known*100:.2f}%, Unknown Recall: {recall_unknown*100:.2f}%')
    
    return per_class_acc * 100, overall_accuracy * 100, average_accuracy * 100, kappa * 100, h_score * 100, true_label, predict_label, test_index, cm, pred_array
