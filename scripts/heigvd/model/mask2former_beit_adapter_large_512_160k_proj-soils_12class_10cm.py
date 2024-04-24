# to train: adjust num_classes and every line with 'pretrained' in it

num_things_classes = 1
num_stuff_classes = 12
num_classes = 13
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderMask2Former',
    # pretrained='pretrained/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiTAdapter',
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        img_size=512,
        init_values=1e-06,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[1024, 1024, 1024, 1024],
        feat_channels=1024,
        out_channels=1024,
        in_index=[0, 1, 2, 3],
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=1024,
                        num_heads=32,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=1024,
                        feedforward_channels=4096,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                        with_cp=True),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=512, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=512, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=1024,
                    num_heads=32,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=1024,
                    feedforward_channels=4096,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                    with_cp=True),
                feedforward_channels=4096,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0 for _ in range(num_classes)] + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True,
        mode='slide',
        crop_size=(512, 512),
        stride=(512, 512)),
    init_cfg=None)

dataset_type = 'ProjSoilsDataset'
data_root = '/proj-soils/data/training/dataset_12cl_seed6-adjusted_10cm'
img_norm_cfg = dict(
    mean=[108.72, 117.99, 93.29], std=[33.5, 30.6, 30.28], to_rgb=False)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

test_pipeline = val_pipeline
data = dict(
    samples_per_gpu=1, # batch size
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/ipt',
        ann_dir='train/tgt',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/ipt',
        ann_dir='val/tgt',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/ipt',
        ann_dir='val/tgt',
        pipeline=test_pipeline))

log_config = dict(
    interval=100, hooks=[ # 
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
        ])
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = '/proj-soils/data/training/M2F_ViTlarge_best_mIoU_iter_160000.pth'

resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup=None,
    power=1.0,
    min_lr=1e-08,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=160_000) #
checkpoint_config = dict(by_epoch=False, interval=2640, max_keep_ckpts=1) #
evaluation = dict(
    interval=2640, metric='mIoU', pre_eval=True, save_best='mIoU') #

work_dir = '/proj-soils/data/training/logs_checkpoints/mask2former_beit_adapter_large_512_160k_proj-soils_12class_10cm'
gpu_ids = 0
auto_resume = False
device = 'cpu'


# exoscale gpu1:
# 2000 iterations ~ 105min
# 1 iteration ~ 0.0525min ~ 3.15s 
# 48000 iterations ~ 2520min ~ 42h ~ 1.75d
# 60000 iterations ~ 3150min ~ 52.5h ~ 2.2d

# infomaniak vm-gpu-03:
# 2000 iterations ~ 149min
# 1 iteration ~ 0.0745min ~ 4.47s
# 48000 iterations ~ 3576min ~ 59.6h ~ 2.5d
# 60000 iterations ~ 4470min ~ 74.5h ~ 3.1d

# REMINDER: add the following snippet at /usr/local/lib/python3.8/dist-packages/mmseg/models/segmentors/base.py:line 159

# log_vars_val = OrderedDict()
# for k,v in log_vars.items():
#     new_key = 'val_' + k
#     log_vars_val[new_key] = v

# and change line 166 to: log_vars=log_vars_val,