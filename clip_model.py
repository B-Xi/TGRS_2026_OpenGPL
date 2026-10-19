from collections import OrderedDict
from typing import Tuple, Union
from simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from pos_embed import interpolate_pos_embed
from  timm.layers import trunc_normal_
from model import vit_HSI_LIDAR_patch3


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(width, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width 
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32 
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for i in range(layers)])
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class cnn_encoder_A_and_B(nn.Module):
    def __init__(self, l1, l2,vision_patch_size,out_channels):
        super(cnn_encoder_A_and_B, self).__init__()

        self.conv1 = conv_bn_relu(l1, 32, 3, 1, 1)
        self.conv2 = conv_bn_relu(l2, 32, 3, 1, 1)
        self.conv1_1 = conv_bn_relu_max(32, out_channels, 3, 1, 1, vision_patch_size)
        self.conv2_1 = conv_bn_relu_max(32, out_channels, 3, 1, 1, vision_patch_size)
        

        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda


    def forward(self, x11, x21):
        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x1_1 = self.conv1_1(x11) #64,64,8,8
        x2_1 = self.conv2_1(x21)
        x_add = x1_1*self.xishu1 + x2_1*self.xishu2

        return x_add,x1_1,x2_1

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor,csa=True):
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)  
        x_vit = x


        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x,x_vit


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for i in range(layers)])
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self,img_size,
                in_chans, in_chans_LIDAR,hid_chans, hid_chans_LIDAR, embed_dim, depth, num_heads,  mlp_ratio,num_classes, global_pool,

                context_length: int,
                vocab_size: int,
                transformer_width: int,
                transformer_heads: int,
                transformer_layers: int,
                prompt_length: int = 8,
                num_unknown_prompts: int = 4,
                use_learned_prompt: bool = True,
                use_Neg_prompt: bool = True,
                ):
        super().__init__()
        self.context_length = context_length
        from config import load_args
        args=load_args()
        if args.dataset == '2013houston':
            prompt_length=23
            num_prompts = 13
        elif args.dataset == 'Muufl':
            prompt_length=14
            num_prompts = 4      
        elif args.dataset == 'Augsburg':
            prompt_length=16
            num_prompts = 4
        self.visual = vit_HSI_LIDAR_patch3(img_size=img_size,
                in_chans=in_chans, in_chans_LIDAR=in_chans_LIDAR,hid_chans=hid_chans,
                hid_chans_LIDAR=hid_chans_LIDAR, embed_dim=embed_dim, depth=depth, num_heads=num_heads,  
                mlp_ratio=mlp_ratio,num_classes=num_classes, global_pool=global_pool)
        # Load pre-trained weights
        try:
            checkpoint = torch.load(f'./pretrained_models/net_{args.dataset}.pt', weights_only=True)
            checkpoint_model = checkpoint['model']  
            
            state_dict = self.visual.state_dict()
            for k in ['head.weight', 'head.bias','head_LIDAR.weight', 'head_LIDAR.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
                    del checkpoint_model[k]
                                
            interpolate_pos_embed(self.visual, checkpoint_model)
            msg = self.visual.load_state_dict(checkpoint_model, strict=False)
            print("Successfully loaded pre-trained visual model. Missing keys:", msg.missing_keys)
            
            trunc_normal_(self.visual.head.weight, std=2e-5)
        except FileNotFoundError:
            print("Pre-trained model './net.pt' not found. Initializing 'visual' from scratch.")

        # Text Transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        self.use_learned_prompt = bool(use_learned_prompt)
        self.use_Neg_prompt = bool(use_Neg_prompt)
        if  use_learned_prompt:

            if isinstance(prompt_length, bool) and isinstance(use_learned_prompt, int):
                tmp = prompt_length
                prompt_length = use_learned_prompt
                use_learned_prompt = tmp

            try:
                prompt_length = int(prompt_length)
            except Exception:
                raise ValueError(f"prompt_length must be int-like (got {type(prompt_length)})")

            self.prompt_length = prompt_length
            self.prompt_dim = transformer_width

            # learned prompts for known classes

            self.prompt_context = nn.Parameter(torch.randn(self.prompt_length, self.prompt_dim))

            self.num_unknown_prompts = num_unknown_prompts

            self.num_unknown_prompts = num_prompts
            self.unknown_prompt_context = nn.Parameter(torch.randn(self.num_unknown_prompts, self.prompt_length, self.prompt_dim))

            nn.init.normal_(self.prompt_context, std=0.02)
            nn.init.normal_(self.unknown_prompt_context, std=0.02)

            self._tokenizer = _Tokenizer()
            self.SOT_ID = self._tokenizer.encoder["<|startoftext|>"]
            self.EOT_ID = self._tokenizer.encoder["<|endoftext|>"]

            self._init_prompts_from_text("a hyperspectral image of", noise_std=1e-2)

        self.TEST_SCHEME='A'

        if self.use_Neg_prompt:
            self.prompt_length = prompt_length
            self.prompt_dim = transformer_width 
            self.prompt_context = nn.Parameter(torch.randn(self.prompt_length, self.prompt_dim))
            self._tokenizer = _Tokenizer()
            self.SOT_ID = self._tokenizer.encoder["<|startoftext|>"]
            self.EOT_ID = self._tokenizer.encoder["<|endoftext|>"]
            self.num_unknown_prompts = num_unknown_prompts
            self.unknown_prompt_context = nn.Parameter(torch.randn(self.num_unknown_prompts, self.prompt_length, self.prompt_dim))

            self.num_negative_prompts = num_prompts

            self.negative_prompt_context = nn.Parameter(torch.randn(self.num_negative_prompts, self.prompt_length, self.prompt_dim))
            nn.init.normal_(self.negative_prompt_context, std=0.02)    
            self._init_prompts_from_text("a hyperspectral image of", noise_std=1e-2)
            self._init_neg_prompts_from_text("not a hyperspectral image of", noise_std=1e-2)

        positive_templates = [
            "a hyperspectral image of",
            "a remote sensing image of",
            "an aerial view of",
            "hyperspectral and lidar data showing", 
            "topography and spectral features of"
        ]

        negative_templates = [
            "not a hyperspectral image of",
            "background clutter",
            "unidentified ground object",
            "irrelevant region",
            "noise and shadow"
        ]

        if self.use_learned_prompt:
            self._init_prompts_from_text(positive_templates, noise_std=0.02)
            
        if self.use_Neg_prompt:
            self._init_neg_prompts_from_text(negative_templates, noise_std=0.02)

    def _init_prompts_from_text(self, base_texts: Union[str, list], noise_std=1e-2):
        if isinstance(base_texts, str):
            base_texts = [base_texts]

        all_embeddings = []
        with torch.no_grad():
            for text in base_texts:
                toks = self._tokenizer.encode(" " + text)
                toks = torch.tensor(toks, dtype=torch.long, device=self.token_embedding.weight.device).unsqueeze(0)
                token_embs = self.token_embedding(toks).squeeze(0) 
                mean_vec = token_embs.mean(dim=0, keepdim=True)
                all_embeddings.append(mean_vec)

            ensemble_mean = torch.stack(all_embeddings).mean(dim=0)

            # Init shared prompt (Context)
            self.prompt_context.data.copy_(
                ensemble_mean.repeat(self.prompt_length, 1) + 
                noise_std * torch.randn_like(self.prompt_context)
            )
            
            # Init unknown prompts (Unknown)
            up = ensemble_mean.repeat(self.num_unknown_prompts, self.prompt_length, 1)
            up = up + noise_std * torch.randn_like(self.unknown_prompt_context)
            self.unknown_prompt_context.data.copy_(up)

    def _init_neg_prompts_from_text(self, neg_texts: Union[str, list], noise_std=1e-2):
        if isinstance(neg_texts, str):
            neg_texts = [neg_texts]

        all_embeddings = []
        with torch.no_grad():
            for text in neg_texts:
                toks = self._tokenizer.encode(" " + text)
                toks = torch.tensor(toks, dtype=torch.long, device=self.token_embedding.weight.device).unsqueeze(0)
                token_embs = self.token_embedding(toks).squeeze(0)
                mean_vec = token_embs.mean(dim=0, keepdim=True)
                all_embeddings.append(mean_vec)
            
            ensemble_mean = torch.stack(all_embeddings).mean(dim=0)

            # Init negative prompts
            nup = ensemble_mean.repeat(self.num_negative_prompts, self.prompt_length, 1) + \
                  noise_std * torch.randn_like(self.negative_prompt_context)
            self.negative_prompt_context.data.copy_(nup)

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return next(self.visual.parameters()).dtype

    def encode_image(self, image, image_li, t):
        return self.visual(image, image_li, t)


    def encode_text(self, text, use_learned_prompt: bool = None, unknown_flag=None,):

        use_Neg_prompt=self.use_Neg_prompt
        if use_learned_prompt is None:
            use_learned_prompt = self.use_learned_prompt
        else:
            use_learned_prompt = bool(use_learned_prompt)

        if not use_learned_prompt:
            x = self.token_embedding(text).type(self.dtype)
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x).type(self.dtype)
            # Extract features at the EOT token
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x

        # use learned prompt
        batch_size = text.shape[0]
        # 1) name_lens: count non-pad tokens minus SOT/EOT
        non_pad = (text != 0).long().sum(dim=1)      
        name_lens = torch.clamp(non_pad - 2, min=0).tolist()

        # 2) token embeddings for class names
        name_embed = self.token_embedding(text).type(self.dtype)  # (N, L, D)

        # 3) unknown flag -> mask tensor
        if unknown_flag is not None:
            flag = unknown_flag.view(batch_size, 1, 1).to(self.dtype).to(text.device)
        else:
            flag = torch.zeros(batch_size, 1, 1, device=text.device, dtype=self.dtype)

        # 4) prepare prompts, move to device/dtype
        known_prompts = self.prompt_context.unsqueeze(0).expand(batch_size, -1, -1).to(text.device).type(self.dtype)
        
        # SOT/EOT embeddings（use tokenizer ids from self）
        sot_embed = self.token_embedding(torch.tensor([self.SOT_ID], device=text.device, dtype=torch.long)).type(self.dtype)
        eot_embed = self.token_embedding(torch.tensor([self.EOT_ID], device=text.device, dtype=torch.long)).type(self.dtype)
        sot_embed = sot_embed.view(1, self.prompt_dim).expand(batch_size, -1).view(batch_size, 1, self.prompt_dim)
        eot_embed = eot_embed.view(1, self.prompt_dim).expand(batch_size, -1).view(batch_size, 1, self.prompt_dim)
        
        # Test ：'A' = Input Mean (qucik), 'B' = Feature Ensemble (precise)
        TEST_SCHEME = 'A'  
        TEST_SCHEME = self.TEST_SCHEME
        if self.training:

            idx = torch.randint(0, self.num_unknown_prompts, (1,)).item()
            target_unknown_prompts_list = [self.unknown_prompt_context[idx]] 
        else:
            if TEST_SCHEME == 'A':
                target_unknown_prompts_list = [self.unknown_prompt_context.mean(dim=0)]
            else:
                ensemble_limit = min(self.num_unknown_prompts, 32) 
                ensemble_limit = self.num_unknown_prompts
                target_unknown_prompts_list = [self.unknown_prompt_context[i] for i in range(ensemble_limit)]

        
        feature_accumulator = []

        for unk_prompt_template in target_unknown_prompts_list:
            
            current_unknown_prompts = unk_prompt_template.unsqueeze(0).expand(batch_size, -1, -1).to(text.device).type(self.dtype)
            
            prompts = (1 - flag) * known_prompts + flag * current_unknown_prompts  # (B, prompt_length, D)

            # 6) build sequence tensor and fill
            x = torch.zeros(batch_size, self.context_length, self.prompt_dim, device=text.device, dtype=self.dtype)
            eot_indices = []
            for i in range(batch_size):
                name_len = int(name_lens[i])
                # SOT
                x[i, 0, :] = sot_embed[i, 0, :]
                # PROMPT
                x[i, 1: 1 + self.prompt_length, :] = prompts[i]
                # CLASS NAME tokens (skip SOT token in name_embed)
                class_start = 1 + self.prompt_length
                class_end = class_start + name_len
                if name_len > 0:
                    x[i, class_start:class_end, :] = name_embed[i, 1:1 + name_len, :]
                # EOT 
                eot_pos = class_end if class_end < self.context_length else (self.context_length - 1)
                x[i, eot_pos, :] = eot_embed[i, 0, :]
                eot_indices.append(eot_pos)

            # 7) add positional embedding and transformer
            x = x + self.positional_embedding.type(self.dtype).to(text.device)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x).type(self.dtype)

            eot_indices = torch.tensor(eot_indices, device=x.device)
            final = x[torch.arange(batch_size, device=x.device), eot_indices] @ self.text_projection
            
            feature_accumulator.append(final)

        if len(feature_accumulator) > 1:
            final_features = torch.stack(feature_accumulator).mean(dim=0)
        else:
            final_features = feature_accumulator[0]

        return final_features
    
    def encode_negative_text(self, text):

        batch_size = text.shape[0]
        non_pad = (text != 0).long().sum(dim=1)
        name_lens = torch.clamp(non_pad - 2, min=0).tolist()
        name_embed = self.token_embedding(text).type(self.dtype)

        # SOT / EOT embedding
        sot_embed = self.token_embedding(torch.tensor([self.SOT_ID], device=text.device, dtype=torch.long)).type(self.dtype)
        eot_embed = self.token_embedding(torch.tensor([self.EOT_ID], device=text.device, dtype=torch.long)).type(self.dtype)
        sot_embed = sot_embed.view(1, self.prompt_dim).expand(batch_size, -1).view(batch_size, 1, self.prompt_dim)
        eot_embed = eot_embed.view(1, self.prompt_dim).expand(batch_size, -1).view(batch_size, 1, self.prompt_dim)


        TEST_SCHEME = 'A' 
        TEST_SCHEME = self.TEST_SCHEME
        if self.training:
            idx = torch.randint(0, self.num_negative_prompts, (1,)).item()
            target_neg_prompts_list = [self.negative_prompt_context[idx]]
        else:
            if TEST_SCHEME == 'A':
                target_neg_prompts_list = [self.negative_prompt_context.mean(dim=0)]
            else:
                ensemble_limit = 10 
                ensemble_limit =self.num_negative_prompts
                target_neg_prompts_list = [self.negative_prompt_context[i] for i in range(min(self.num_negative_prompts, ensemble_limit))]

        feature_accumulator = []

        for neg_prompt_template in target_neg_prompts_list:

            neg_prompts = neg_prompt_template.unsqueeze(0).expand(batch_size, -1, -1).to(text.device).type(self.dtype)

            x = torch.zeros(batch_size, self.context_length, self.prompt_dim, device=text.device, dtype=self.dtype)
            eot_indices = []
            for i in range(batch_size):
                name_len = int(name_lens[i])
                x[i, 0, :] = sot_embed[i, 0, :]
                x[i, 1: 1 + self.prompt_length, :] = neg_prompts[i]
                class_start = 1 + self.prompt_length
                class_end = class_start + name_len
                if name_len > 0:
                    x[i, class_start:class_end, :] = name_embed[i, 1:1 + name_len, :]
                eot_pos = class_end if class_end < self.context_length else (self.context_length - 1)
                x[i, eot_pos, :] = eot_embed[i, 0, :]
                eot_indices.append(eot_pos)

            # transformer + LN + projection
            x = x + self.positional_embedding.type(self.dtype).to(text.device)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x).type(self.dtype)

            eot_indices = torch.tensor(eot_indices, device=x.device)
            final = x[torch.arange(batch_size, device=x.device), eot_indices] @ self.text_projection
            
            feature_accumulator.append(final)

        if len(feature_accumulator) > 1:
            final_features = torch.stack(feature_accumulator).mean(dim=0)
        else:
            final_features = feature_accumulator[0]

        return final_features

    def forward(self, image, image_li, t, text, unknown_flag=None):

        image_features = self.encode_image(image, image_li, t)
        use_Neg_prompt=self.use_Neg_prompt
        use_learned_prompt=self.use_learned_prompt

        if use_learned_prompt:
            text_features = self.encode_text(text, use_learned_prompt, unknown_flag)
        else:
            text_features = self.encode_text(text)
        # Normalize features for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def get_features(self, image, image_lidar, t, text, unknown_flag=None):

        use_Neg_prompt=self.use_Neg_prompt
        use_learned_prompt=self.use_learned_prompt
        image_features = self.encode_image(image, image_lidar, t)
        
        if use_learned_prompt:
            text_features = self.encode_text(text, use_learned_prompt, unknown_flag)
        else:
            text_features = self.encode_text(text)
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        logit_scale = self.logit_scale.exp()
    
        return image_features, text_features, logit_scale

def build_model(img_size, in_chans, in_chans_LIDAR, hid_chans, hid_chans_LIDAR,
                embed_dim, depth, num_heads, mlp_ratio, num_classes, global_pool=False,
                use_learned_prompt=True,use_Neg_prompt:bool = True ,prompt_length=8, num_unknown_prompts=4):
    context_length = 77
    vocab_size = 49408
    transformer_width = 128
    transformer_heads = 8
    transformer_layers = 3 
    
    model = CLIP(
        img_size=img_size,
        in_chans=in_chans,
        in_chans_LIDAR=in_chans_LIDAR,
        hid_chans=hid_chans,
        hid_chans_LIDAR=hid_chans_LIDAR,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        global_pool=global_pool,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        prompt_length=prompt_length,
        num_unknown_prompts=num_unknown_prompts,
        use_learned_prompt=use_learned_prompt,
        use_Neg_prompt=use_Neg_prompt
    )
    return model