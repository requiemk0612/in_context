import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import openai_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData  

from mmseg.registry import MODELS

from torchvision import transforms
import torch.nn.functional as F
from open_clip import create_model, tokenizer
from myutils import UnNormalize
from typing import Tuple
from PIL import Image
import numpy as np
import os

class ImageNorm(nn.Module):
    # 初始化逐 token 的 L2 归一化层。
    # 该层不包含可训练参数或持久状态，只用于把图像 token 映射到单位特征球面，
    # 以便后续与已经归一化的文本特征直接计算余弦相似度。
    def __init__(self):
        super().__init__()
        
    # 沿最后一个特征维度对输入张量做 L2 归一化。
    # x 通常形如 [B, N, C]，返回形状保持不变；这里使用原地除法，因此会直接
    # 改写传入张量。调用方应确保 token 范数非零，避免产生 NaN 或 Inf。
    def forward(self, x):
        x /= x.norm(dim=-1, keepdim=True)
        return x

class ImageTextDotProduct(nn.Module):
    # 保存文本类别原型，构造无可训练参数的图文相似度计算层。
    # text_features 预期形如 [Q, C]，Q 是文本查询/类别同义词数量；通过
    # register_buffer 注册后，它会随模块迁移设备并写入 state_dict，但不会参与梯度更新。
    def __init__(self, text_features: torch.Tensor):
        super().__init__()
        self.register_buffer('text_features', text_features)
        
    # 计算每个图像 token 与所有文本查询原型的点积相似度。
    # image_features 通常为 [B, N, C]，text_features.T 为 [C, Q]，
    # 返回 logits 的形状为 [B, N, Q]；若两侧都已归一化，该点积等价于余弦相似度。
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        logits = image_features @ self.text_features.T
        return logits

@MODELS.register_module()
class GLA_CLIPSegmentation(BaseSegmentor):
    # 构造 GLA-CLIP 开放词汇语义分割器，并初始化其推理所需的全部组件。
    # 实际模型类型、设备、滑窗参数和注意力超参数主要从 model_cfg 读取；构造函数中
    # 同名的独立参数保留用于 MMSeg 配置兼容。初始化过程会创建半精度 CLIP，按
    # vfm_model 加载并冻结 DINO/DINOv2 视觉特征模型，可选加载 FARE 权重；随后读取
    # name_path 中的类别名称/同义词，使用多套 prompt 模板编码并平均成归一化文本原型。
    # 最后注册图像归一化、图文点积层及后处理所需的类别映射。该方法会访问模型权重、
    # 将模型移到 model_cfg.device，并可能通过 torch.hub 触发外部模型加载。
    def __init__(self, clip_type, model_type, vfm_model, name_path, checkpoint=None, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, beta=1.2, gamma=3.0, fare=False, slide_stride=112, slide_crop=224, scale=336, bg_idx=0, model_cfg=None):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)
        
        self.clip = create_model(model_cfg.model_type, pretrained=model_cfg.clip_type, precision='fp16')
        self.clip.eval().to(model_cfg.device)

        # CLIP_type
        self.CLIP_type = model_cfg.CLIP_type

        # Slide setting
        self.slide_stride = model_cfg.slide_stride
        self.slide_crop = model_cfg.slide_crop

        # etc
        self.scale = model_cfg.scale
        self.work_dir = model_cfg.work_dir
        
        # token norm
        self.token_norm = model_cfg.token_norm
        
        # kv token extention setting
        self.KV_token_extension = model_cfg.KV_token_extension
        self.temperature = model_cfg.temperature
        self.cutting_hp = model_cfg.cutting_hp
        
        # cfg
        self.model_cfg = model_cfg
        
        # Load 
        if fare == True:
            checkpoint = torch.load('model/fare_eps_4.pt', map_location=torch.device('cpu'))
            self.clip.visual.load_state_dict(checkpoint)
        
        self.tokenizer = tokenizer.tokenize

        self.vfm_model = model_cfg.vfm_model
        checkpoint = model_cfg.checkpoint
    
        if model_cfg.vfm_model == 'dino':
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
           
        elif model_cfg.vfm_model == 'dinov2':
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        
        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(model_cfg.device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        query_words, self.query_idx = get_cls_idx(model_cfg.name_path)
        
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(model_cfg.device)
        
        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(model_cfg.device)
                feature = self.clip.encode_text(query)
                
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.dtype = self.query_features.dtype
        self.logit_scale = model_cfg.logit_scale
        self.prob_thd = model_cfg.prob_thd
        self.bg_idx = bg_idx

        self.beta = model_cfg.beta
        self.gamma = model_cfg.gamma
        
        # added module
        self.imgnorm = ImageNorm()
        self.iq_dotproduct = ImageTextDotProduct(self.query_features)

    # 为一批完整图像或滑窗 crop 提取逐 token 的 CLIP 图像特征。
    # img 形如 [B, 3, H, W]，首先从 CLIP 归一化空间反归一化，再变换到 DINO 所需  (DINO需要什么统计分布？)
    # 的统计分布。ProxyCLIP 分支会额外提取 DINO/DINOv2 patch 特征，并作为
    # external_feats 传入 CLIP 最后一层；indices 可携带每个 patch 在原图中的位置，
    # 供跨窗口 smoothing 使用。返回 encode_image 产生的 [B, N, C] token 特征。
    # @torch.no_grad() 保证整个特征提取过程不构建反向传播图。
    @torch.no_grad()
    def forward_feature(self, img, indices=None):
        clip_token_size = img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[1]  # [14, 14]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.half()

        if self.model_cfg.CLIP_type == "ProxyCLIP":
            if self.vfm_model == 'dino':
                
                feat_out = {}
                # 捕获 DINO 最后一个注意力块的原始 QKV 投影输出。
                # module、input 是 PyTorch forward hook 规定的接口参数；output 为该层
                # 输出张量。函数仅把 output 保存到外层 feat_out 字典，无返回值，
                # 以便随后的同一次 DINO 前向传播结束后拆分 q、k、v 特征。
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                    hook_fn_forward_qkv)

                # Forward pass in the model
                feat = self.vfm.get_intermediate_layers(imgs_norm)[0]   #[21, 785, 768]
                nb_im = feat.shape[0]  # Batch size
                nb_tokens = feat.shape[1]  # Number of tokens
                nh = self.vfm.blocks[0].attn.num_heads  # Number of heads
                
                qkv = (
                    feat_out["qkv"]
                    .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
                k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
                q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
                v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
                
                patch_size = self.vfm.patch_embed.patch_size
                I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size   # [H, W]
                ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
                
            elif self.vfm_model == 'dinov2':
                patch_size = self.vfm.patch_embed.patch_size
                I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
                ex_feats = self.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]

        else:
            I, J = clip_token_size
            ex_feats = None

        image_features = self.clip.encode_image(img.half(), external_feats=ex_feats, beta=self.beta, gamma=self.gamma, indices=indices, model_cfg=self.model_cfg)

        return image_features
    
    # 对输入图像执行带重叠的滑动窗口推理，并在需要时启用跨窗口 K/V 扩展。
    # img 支持 [B, C, H, W] 张量或 MMSeg 传入的单元素列表；stride 和 crop_size
    # 可以是整数或 (高, 宽) 二元组。方法会生成所有窗口、窗口边界及每个 patch
    # 对应的原图线性索引，必要时将窗口对称填充到 56 的整数倍，然后一次性提取特征。
    # 每个窗口的 token 与文本原型计算 logits，双线性上采样回 crop 分辨率后贴回原图；
    # 重叠像素通过 count_mat 做算术平均，最后再缩放到 img_metas 指定的原始尺寸。
    # 返回 [B, num_queries, ori_h, ori_w] 的查询级 logits，并会同步更新 model_cfg 中
    # 的网格数、token 尺寸、原图尺寸、bbox_list 和当前图像名等运行时元数据。
    def forward_kv_expansion(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride     
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        crop_images = []
        bbox_list = []
        pads_list = []

        idx_map = range(h_img * w_img)
        idx_map = torch.tensor(idx_map, dtype=torch.int64, device=img.device)
        idx_map = idx_map.reshape(batch_size, 1, h_img, w_img)
        crop_indices = []

        feat_patch_h_sz = self.clip.visual.patch_size[0] // 2
        feat_patch_w_sz = self.clip.visual.patch_size[1] // 2

        h_feat = h_img // feat_patch_h_sz
        w_feat = w_img // feat_patch_w_sz

        self.model_cfg.h_feat = h_feat
        self.model_cfg.w_feat = w_feat

        for h_idx in range(h_grids):
            for w_idx in range(w_grids): 
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_idx = idx_map[:, :, y1:y2, x1:x2]  # [1, 1, h_crop, w_crop]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)
                pads_list.append(pad)
                
                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                    crop_idx = nn.functional.pad(crop_idx, pad) if crop_idx is not None else None
                   
                crop_images.append(crop_img)
                crop_indices.append(crop_idx)
                bbox_list.append((x1, x2, y1, y2))
        
        crop_images = torch.cat(crop_images, dim=0)   # [21, 3, 224, 224]
        crop_indices = torch.cat(crop_indices, dim=0)   # [21, 1, 224, 224]

        patch_size, _ = self.clip.visual.patch_size
        token_h = h_crop // patch_size
        token_w = w_crop // patch_size
        self.model_cfg.token_h = token_h
        self.model_cfg.token_w = token_w
        self.model_cfg.bbox_list = bbox_list

        if self.model_cfg.CLIP_type == "ProxyCLIP":
            
            if self.model_cfg.vfm_model == 'dino':
                vfm_patch_size = self.vfm.patch_embed.patch_size       
            else:
                vfm_patch_size = self.vfm.patch_embed.patch_size[0]

            crop_indices = crop_indices[:, :, ::vfm_patch_size, ::vfm_patch_size]
            
            vfm_token_h = h_crop // vfm_patch_size
            vfm_token_w = w_crop // vfm_patch_size
            crop_indices = crop_indices.view(h_grids, w_grids, 1, vfm_token_h, vfm_token_w)
        
        else:
            crop_indices = crop_indices[:, :, ::patch_size, ::patch_size]
            crop_indices = crop_indices.view(h_grids, w_grids, 1, token_h, token_w)
        
        self.model_cfg.h_grids = h_grids
        self.model_cfg.w_grids = w_grids
        self.model_cfg.img_h = h_img
        self.model_cfg.img_w = w_img
        self.model_cfg.img_name = img_metas[0]['img_path'].split('/')[-1][:-4]

        # Last block forwarding
        image_features = self.forward_feature(crop_images, crop_indices)        

        # [21, 224, 224, dim]
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        
        image_features = self.imgnorm(image_features)   # [21, 28 * 28, dim]
        logits = self.iq_dotproduct(image_features)

        H = W = int(logits.shape[1] ** 0.5)
        
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], H, W)
        logits = F.interpolate(logits, size=(h_crop, w_crop), mode='bilinear', align_corners=False)   # [21, 19, 224, 224]
        
        for idx in range(len(bbox_list)):
            x1, x2, y1, y2 = bbox_list[idx]
            logit_idx = logits[idx].unsqueeze(0)
            pad = pads_list[idx]

            crop_seg_logit = logit_idx
            if any(pad):
                l, t = pad[0], pad[2]
                crop_seg_logit = logit_idx[:, :, t:t+(y2-y1), l:l+(x2-x1)]

            preds += nn.functional.pad(crop_seg_logit,
                                        (int(x1), int(preds.shape[3] - x2), int(y1),
                                        int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
        preds = preds / count_mat

        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')  
        
        return logits

    # 实现 MMSeg 的预测入口，负责组织元数据、选择推理路径并生成标准结果对象。
    # inputs 是预处理后的 [B, C, H, W] 张量；data_samples 存在时从中读取每张图的
    # metainfo，否则使用 ori_shape 构造最小元数据。slide_crop>0 时走滑窗及 K/V
    # 扩展路径，否则直接提取整图特征。随后调用 postprocess_result，把查询 logits
    # 转为类别预测。返回更新后的 data_samples；无 data_samples 时返回分割索引张量。
    def predict(self, inputs, data_samples, ori_shape=None):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
            H, W = inputs.shape[2:]

            # label = Image.open(batch_img_metas[0]['seg_map_path'])
            # label = np.array(label, dtype=np.float32)  
            # label = torch.as_tensor(label).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device='cuda')
            # label = F.interpolate(label, size=(H, W), mode='nearest')
            # gt_path = batch_img_metas[0]['seg_map_path']

        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=ori_shape,
                                      img_shape=inputs.shape[-2:],
                                      pad_shape=inputs.shape[-2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        with torch.no_grad():
            if self.slide_crop > 0:
                seg_logits = self.forward_kv_expansion(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self.forward_feature(inputs, None)

        pos_res = self.postprocess_result(seg_logits, data_samples)

        return pos_res



    # 将文本查询级 logits 聚合为最终语义类别概率和像素预测。
    # seg_logits 形如 [B, Q, H, W]，先乘 logit_scale 并沿 Q 维 softmax；当多个
    # query/同义词属于同一类别时，根据 self.query_idx 使用 amax 聚合为
    # [num_classes, H, W]。之后取 argmax 得到 [1, H, W] 的类别图，并把最大概率
    # 低于 prob_thd 的像素重置为 bg_idx。若 data_samples 不为空，会为每个样本写入
    # seg_logits 和 pred_sem_seg 两个 PixelData 字段并返回列表；否则直接返回首个
    # 样本的类别图。gt_path 当前为保留参数，方法内没有使用。
    def postprocess_result(self, seg_logits, data_samples, gt_path=None):
        """
        seg_logits: [B, Q, H, W]
        """
        batch_size, num_queries, H, W = seg_logits.shape
        num_cls = max(self.query_idx) + 1

        # query_idx를 tensor로 준비
        if not torch.is_tensor(self.query_idx):
            query_idx = torch.as_tensor(self.query_idx, device=seg_logits.device, dtype=torch.long)
        else:
            query_idx = self.query_idx.to(device=seg_logits.device, dtype=torch.long)

        results = []

        for i in range(batch_size):
            probs = seg_logits[i] * self.logit_scale          # [Q, H, W]
            probs = probs.softmax(0)                          # [Q, H, W]

            # query -> class aggregation
            if num_cls != num_queries:
                cls_probs = torch.full(
                    (num_cls, H, W),
                    fill_value=torch.finfo(probs.dtype).min,
                    device=probs.device,
                    dtype=probs.dtype,
                )

                # [Q, H, W] 형태의 class index
                scatter_index = query_idx[:, None, None].expand(-1, H, W)

                cls_probs.scatter_reduce_(
                    dim=0,
                    index=scatter_index,
                    src=probs,
                    reduce="amax",
                    include_self=True,
                )
            else:
                cls_probs = probs

            seg_pred = cls_probs.argmax(0, keepdim=True)      # [1, H, W]

            uncertainty_mask = cls_probs.max(0, keepdim=True)[0] < self.prob_thd
            if uncertainty_mask.any():
                seg_pred[uncertainty_mask] = self.bg_idx

            if data_samples is None:
                return seg_pred

            data_samples[i].set_data({
                'seg_logits': PixelData(data=cls_probs),
                'pred_sem_seg': PixelData(data=seg_pred),
            })

        return data_samples


    # 计算把 H×W 图像补齐到 patch_size 整数倍所需的对称 padding。
    # 宽度余量优先平均分配到左、右，高度余量平均分配到上、下；当差值为奇数时，
    # 右侧或下侧会比另一侧多 1。返回顺序为 (left, right, top, bottom)，可直接传给
    # torch.nn.functional.pad；已经整除的维度对应 padding 为 0。
    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    # MMSeg BaseSegmentor 要求的底层张量前向接口占位。
    # data_samples 名称沿用当前源码，但该签名没有 self，且函数体尚未实现；
    # 当前调用会隐式返回 None，不能作为有效的模型推理路径使用。
    def _forward(data_samples):
        """
        """

    # MMSeg 风格的 inference 接口占位，预期根据 img 和 batch_img_metas 执行推理。
    # 当前函数体尚未实现，调用后只会隐式返回 None；实际预测逻辑位于 predict 和
    # forward_kv_expansion 中。
    def inference(self, img, batch_img_metas):
        """
        """

    # MMSeg 编码—解码推理接口占位，预期把输入特征转换为与原图对齐的分割 logits。
    # 当前没有任何执行逻辑并隐式返回 None，尚未连接本类已有的特征提取与后处理流程。
    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    # MMSeg 特征提取接口占位，理论上应返回 backbone/neck 的多尺度特征。
    # 当前函数体为空并隐式返回 None；本项目真正使用的 token 提取方法是
    # forward_feature，二者暂未建立适配关系。
    def extract_feat(self, inputs):
        """
        """

    # MMSeg 训练损失接口占位，预期根据 inputs 和带标注的 data_samples 返回损失字典。
    # 当前模型仅实现推理流程，此方法没有训练计算并隐式返回 None。
    def loss(self, inputs, data_samples):
        """
        """

# 从类别名称文件构建文本查询列表，以及查询到语义类别的映射关系。
# path 指向文本文件，每行代表一个类别，同一行可用“; ”分隔多个同义名称；
# 函数按文件行号为这些同义词分配相同类别索引，并去掉名称末尾的换行符。
# 返回 (class_names, class_indices)：前者是展开后的所有查询字符串，后者是等长的
# 整数类别编号列表。文件为空或格式不符合约定时，本函数不做额外校验。
def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
