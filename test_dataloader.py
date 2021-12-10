from dataset.coco_dataset import COCODataLoader
from types import SimpleNamespace

if __name__ == '__main__':
    args = SimpleNamespace()
    args.dataroot = r'D:\Data\ultrapose'
    args.eval_dataroot = r'D:\Data\ultrapose'
    args.rank = 0
    loader = COCODataLoader(args, False)
    for i, item in enumerate(loader):
        print(item['input_conditon'].shape)
        print(item['body_s_class'].shape)
        print(item['dp_xy_iuv'].shape)