import argparse
import torch.nn

def load_args():
    parser = argparse.ArgumentParser()

    # Pre training
    parser.add_argument('--train_num_perclass', type=int, default=100)
    parser.add_argument('--train_num_perclass_two', type=int, default=20)
    # Pre training
    
    parser.add_argument('--windowsize', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--fine_tuned_epochs', type=int, default=150)
    #parser.add_argument('--fine_tuned_epochs', type=int, default=10)
    # Prompt Tuning Warmup
    parser.add_argument('--lambda_feat_l2', type=float, default=1e-4, 
                    help='Weight for L2 normalization on raw image features (stabilizer)')
    parser.add_argument('--do_prompt_tuning', type=bool, default=True,
                        help='Whether to perform prompt-only warm-up before finetuning (default: True)')
    parser.add_argument('--prompt_tuning_steps', type=int, default=200,
                        help='Number of steps for prompt-only warm-up (default: 200)')
    parser.add_argument('--prompt_tuning_lr', type=float, default=1e-4,
                        help='Learning rate for prompt-only warm-up (default: 1e-3)')

    # Finetune Loss Weights (Positive/Negative Prompt)
    parser.add_argument('--lambda_rank', type=float, default=0.2,
                        help='Weight for Margin Ranking Loss (pushing neg features away) (default: 1.0)')
    parser.add_argument('--lambda_textneg', type=float, default=0.1,
                        help='Weight for Text Semantic Loss (pushing pos/neg prompts apart) (default: 0.1)')
    parser.add_argument('--lambda_unk_reject', type=float, default=0.2,
                        help='Weight for Unknown Rejection Loss (pushing known feats away from unknown prompt) (default: 0.5)')
    
    # Diffusion/MAE Pretrain Loss Weights
    parser.add_argument('--recon_loss_weight', type=float, default=0.1,
                        help='Total weight for Reconstruction Loss in pretraining (default: 0.1)')
    parser.add_argument('--cls_loss_weight', type=float, default=0.05,
                        help='Weight for Classification Loss in pretraining (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fine_tuned_lr', type=float, default=1e-4)

    parser.add_argument('--type', type=str, default='none')

    parser.add_argument('--dataset', type=str, default='2013houston') #2013houston  Muufl  Augsburg
    #parser.add_argument('--dataset', type=str, default='Muufl') #2013houston  Muufl  Augsburg
    #parser.add_argument('--dataset', type=str, default='Augsburg') #2013houston  Muufl  Augsburg

    # Network
    parser.add_argument('--mask_ratio', default=0.7, type=float,
                        help='mask_ratio (default: 0.8)')
    parser.add_argument('--mlp_ratio', default=2.0, type=float,
                        help='mlp ratio of encoder/decoder (default: 2.0)')
    parser.add_argument('--hid_chans', default=128, type=int,
                        help='hidden channels for dimension reduction (default: 128)')     
    parser.add_argument('--hid_chans_LIDAR', default=128, type=int,
                        help='hidden channels for dimension reduction (default: 128)')
    # Augmentation
    parser.add_argument('--augment', default=True, type=bool,
                        help='either use data augmentation or not (default: False)')
    parser.add_argument('--scale', default=9, type=int,
                        help='the minimum scale for center crop (default: 19)')
        # Random seed control
    parser.add_argument('--use_seed', type=bool, default=True,
                        help='whether to use fixed random seed for reproducibility (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility (default: 42)')

    # MAE encoder specifics
    parser.add_argument('--encoder_dim', default=128, type=int,
                        help='feature dimension for encoder (default: 64)')
    parser.add_argument('--encoder_depth', default = 4, type=int,
                        help='encoder_depth; number of blocks ')
    parser.add_argument('--encoder_num_heads', default=8, type=int,
                        help='number of heads of encoder (default: 8)')    
    
    # MAE decoder specifics
    parser.add_argument('--decoder_dim', default= 128, type=int,
                        help='feature dimension for decoder (default: 64)')
    parser.add_argument('--decoder_depth', default = 3, type=int,
                        help='decoder_depth; number of blocks ')    
    parser.add_argument('--decoder_num_heads', default=8, type=int,
                        help='number of heads of decoder (default: 8)') 

    # options for supervised MAE    
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='temperature for classification logits')
    parser.add_argument('--cls_loss_ratio', default=0.005, type=float,
                        help='ratio for classification loss')

    parser.add_argument('--m_pseudo', default=0, type=int,
                        help='number of pseudo labels for each class')
    parser.add_argument('--tau',           type=float, default=0.3,
                        help='open-set threshold for unknown vs known')
    
    # Pseudo-unknown generation control
    parser.add_argument('--use_pseudo_unknown', action='store_true', default=True,
                        help='Whether to use pseudo-unknown samples for training (default: False)')
    parser.add_argument('--pseudo_strategy', type=str, default='diffusion', choices=['diffusion','mixup','hybrid'],
                        help='Strategy for pseudo-unknown generation: diffusion, beta, or flow (default: beta)')

    # Flow-guided pseudo-unknown parameters (Plan B: uncertainty-guided masked mixing)
    parser.add_argument('--ues_flow', action='store_true', default=True,
                        help='Wheather to use flow to strengthen the model about classifier ')    

    parser.add_argument('--flow_use_pixel_heat', action='store_true', default=True,
                        help='Whether to use pixel heat for pseudo-unknown generation (default: False)')   
    parser.add_argument('--flow_guided_mask_min', type=float, default=0.4,
                        help='Min side ratio of rectangular paste mask (relative to patch size)')
    parser.add_argument('--flow_guided_mask_max', type=float, default=0.8,
                        help='Max side ratio of rectangular paste mask (relative to patch size)')
    parser.add_argument('--flow_guided_beta_alpha', type=float, default=2.0,
                        help='Alpha for Beta distribution of mixing strength in mask')
    parser.add_argument('--flow_guided_beta_beta', type=float, default=5.0,
                        help='Beta for Beta distribution of mixing strength in mask')
    parser.add_argument('--flow_guided_topk_confuse', type=int, default=1,
                        help='Use top-k confusing classes to pick partner patches')
    parser.add_argument('--flow_guided_num_pieces_min', type=int, default=1,
                        help='Minimum number of small masks per paste')
    parser.add_argument('--flow_guided_num_pieces_max', type=int, default=3,
                        help='Maximum number of small masks per paste')
    parser.add_argument('--flow_guided_feather_px', type=int, default=2,
                        help='Feather width (pixels) for soft-edge masks')
    parser.add_argument('--flow_guided_extreme_prob', type=float, default=0.4,
                        help='Probability to inject extreme lambda (very small or very large)')
    parser.add_argument('--flow_guided_large_mask_prob', type=float, default=0.3,
                        help='Probability to add one larger mask piece to increase diversity')
    parser.add_argument('--flow_guided_large_mask_min', type=float, default=0.6,
                        help='Min side ratio for large mask (relative to patch size)')
    parser.add_argument('--flow_guided_large_mask_max', type=float, default=0.9,
                        help='Max side ratio for large mask (relative to patch size)')
    parser.add_argument('--flow_guided_target_cap_ratio', type=float, default=0.15,
                        help='Cap per target class proportion among generated pseudo unknowns')
    
    # Diffusion-based pseudo-unknown parameters
    parser.add_argument('--diffusion_timestep', type=int, default=200,
                        help='Timestep for diffusion-based pseudo-unknown generation (default: 600)')
    
    # Beta distribution pseudo-unknown parameters
    parser.add_argument('--beta_alpha', type=float, default=0.5,
                        help='Alpha parameter for Beta distribution in pseudo-unknown generation (default: 0.5)')
    parser.add_argument('--beta_beta', type=float, default=0.5,
                        help='Beta parameter for Beta distribution in pseudo-unknown generation (default: 0.5)')
    
    # Common pseudo-unknown parameters
    parser.add_argument('--pseudo_per_class', type=int, default=20,
                        help='Number of pseudo-unknown samples per known class (default: 5)')
    
    parser.add_argument('--use_learned_prompt', action='store_true', default=True,
                        help='Number of pseudo-unknown samples per known class (default: 5)')
    parser.add_argument('--log_print', action='store_true', default=False,
                        help='Number of pseudo-unknown samples per known class (default: 5)')    

    parser.add_argument('--use_Neg_prompt', action='store_true', default=True,
                        help='Number of pseudo-unknown samples per known class (default: 5)')
    
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Whether to use test-time adaptation (default: False)')

    args = parser.parse_args()

    return args

