import csv, os
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

from vocab import load_vocab
from coco_dataset import CocoDataset, collate_fn
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomHorizontalFlip

scale = [0.75,1.0]
ratio = [3/4,4/3]

augmentation_trans = {
    'rCrop': RandomResizedCrop(256, scale, ratio),
    'rRotation': RandomRotation(5.0),
    'rhFlip': RandomHorizontalFlip()
}

# Builds your datasets here based on the configuration.
# You are not required to modify this code but you are allowed to.
def get_datasets(config_data):
    images_root_dir = config_data['dataset']['images_root_dir']
    root_train = os.path.join(images_root_dir, 'train')
    root_val = os.path.join(images_root_dir, 'val')
    root_test = os.path.join(images_root_dir, 'test')

    train_ids_file_path = config_data['dataset']['training_ids_file_path']
    val_ids_file_path = config_data['dataset']['validation_ids_file_path']
    test_ids_file_path = config_data['dataset']['test_ids_file_path']

    train_annotation_file = config_data['dataset']['training_annotation_file_path']
    test_annotation_file = config_data['dataset']['test_annotation_file_path']
    coco = COCO(train_annotation_file)
    coco_test = COCO(test_annotation_file)

    vocab_threshold = config_data['dataset']['vocabulary_threshold']
    vocabulary = load_vocab(train_annotation_file, vocab_threshold)
    
    aug_flags = config_data['dataset']['aug_flags']
    input_transform = None if len(aug_flags) == 0 else transforms.Compose([
        augmentation_trans[t] for t in aug_flags
    ])

    train_data_loader = get_coco_dataloader(train_ids_file_path, root_train, train_annotation_file, coco, vocabulary,
                                            config_data, input_transform)
    val_data_loader = get_coco_dataloader(val_ids_file_path, root_val, train_annotation_file, coco, vocabulary,
                                          config_data)
    test_data_loader = get_coco_dataloader(test_ids_file_path, root_test, test_annotation_file, coco_test, vocabulary,
                                           config_data)

    return coco_test, vocabulary, train_data_loader, val_data_loader, test_data_loader


def get_coco_dataloader(img_ids_file_path, imgs_root_dir, annotation_file_path, coco_obj, vocabulary, config_data, trans=None):
    with open(img_ids_file_path, 'r') as f:
        reader = csv.reader(f)
        img_ids = list(reader)

    img_ids = [int(i) for i in img_ids[0]]

    ann_ids = [coco_obj.imgToAnns[img_ids[i]][j]['id'] for i in range(0, len(img_ids)) for j in
               range(0, len(coco_obj.imgToAnns[img_ids[i]]))]

    dataset = CocoDataset(root=imgs_root_dir,
                          json=annotation_file_path,
                          ids=ann_ids,
                          vocab=vocabulary,
                          img_size=config_data['dataset']['img_size'],
                          transform=trans)
    return DataLoader(dataset=dataset,
                      batch_size=config_data['dataset']['batch_size'],
                      shuffle=True,
                      num_workers=config_data['dataset']['num_workers'],
                      collate_fn=collate_fn,
                      pin_memory=True)
