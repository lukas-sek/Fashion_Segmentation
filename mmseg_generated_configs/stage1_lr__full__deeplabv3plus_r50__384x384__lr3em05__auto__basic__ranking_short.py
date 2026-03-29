_base_ = [r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\mmsegmentation\configs\deeplabv3plus\deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py']

custom_imports = dict(
    imports=['fashion_mmseg_transforms'],
    allow_failed_imports=False,
)

crop_size = (384, 384)
num_classes = 47
work_dir = r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\mmseg_workdirs\stage1_lr__full__deeplabv3plus_r50__384x384__lr3em05__auto__basic__ranking_short'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(384, 384), ratio_range=(0.8, 1.2), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=10, pad_val=0, seg_pad_val=255),
    dict(type='RandomCrop', crop_size=(384, 384), cat_max_ratio=0.9),
    dict(type='PhotoMetricDistortion'),
    dict(type='FashionSegCutOut', prob=0.3, num_holes=2, min_ratio=0.05, max_ratio=0.12, fill_value=0),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(384, 384), keep_ratio=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root='',
        ann_file=r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\mmseg_artifacts\debug_dataset\train.txt',
        data_prefix=dict(img_path=r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\data\train', seg_map_path=r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\data\segmentations_train'),
        img_suffix='.jpg',
        seg_map_suffix='_seg.png',
        metainfo=dict(classes=['background', 'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'], palette=[[0, 0, 0], [37, 67, 97], [74, 134, 194], [111, 201, 35], [148, 12, 132], [185, 79, 229], [222, 146, 70], [3, 213, 167], [40, 24, 8], [77, 91, 105], [114, 158, 202], [151, 225, 43], [188, 36, 140], [225, 103, 237], [6, 170, 78], [43, 237, 175], [80, 48, 16], [117, 115, 113], [154, 182, 210], [191, 249, 51], [228, 60, 148], [9, 127, 245], [46, 194, 86], [83, 5, 183], [120, 72, 24], [157, 139, 121], [194, 206, 218], [231, 17, 59], [12, 84, 156], [49, 151, 253], [86, 218, 94], [123, 29, 191], [160, 96, 32], [197, 163, 129], [234, 230, 226], [15, 41, 67], [52, 108, 164], [89, 175, 5], [126, 242, 102], [163, 53, 199], [200, 120, 40], [237, 187, 137], [18, 254, 234], [55, 65, 75], [92, 132, 172], [129, 199, 13], [166, 10, 110]]),
        reduce_zero_label=False,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root='',
        ann_file=r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\mmseg_artifacts\debug_dataset\val.txt',
        data_prefix=dict(img_path=r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\data\test', seg_map_path=r'C:\Users\Daniel\Documents\Masters\S2\OR\Fashion_Segmentation\data\segmentations_val'),
        img_suffix='.jpg',
        seg_map_suffix='_seg.png',
        metainfo=dict(classes=['background', 'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'], palette=[[0, 0, 0], [37, 67, 97], [74, 134, 194], [111, 201, 35], [148, 12, 132], [185, 79, 229], [222, 146, 70], [3, 213, 167], [40, 24, 8], [77, 91, 105], [114, 158, 202], [151, 225, 43], [188, 36, 140], [225, 103, 237], [6, 170, 78], [43, 237, 175], [80, 48, 16], [117, 115, 113], [154, 182, 210], [191, 249, 51], [228, 60, 148], [9, 127, 245], [46, 194, 86], [83, 5, 183], [120, 72, 24], [157, 139, 121], [194, 206, 218], [231, 17, 59], [12, 84, 156], [49, 151, 253], [86, 218, 94], [123, 29, 191], [160, 96, 32], [197, 163, 129], [234, 230, 226], [15, 41, 67], [52, 108, 164], [89, 175, 5], [126, 242, 102], [163, 53, 199], [200, 120, 40], [237, 187, 137], [18, 254, 234], [55, 65, 75], [92, 132, 172], [129, 199, 13], [166, 10, 110]]),
        reduce_zero_label=False,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator

model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(num_classes=num_classes, ignore_index=255),
)

if 'auxiliary_head' in model:
    if isinstance(model['auxiliary_head'], list):
        for head in model['auxiliary_head']:
            head['num_classes'] = num_classes
            head['ignore_index'] = 255
    else:
        model['auxiliary_head']['num_classes'] = num_classes
        model['auxiliary_head']['ignore_index'] = 255

optim_wrapper = dict(optimizer=dict(lr=3e-05))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100, val_interval=50)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=50, save_best='mDice'),
    logger=dict(interval=50),
)
randomness = dict(seed=42)
