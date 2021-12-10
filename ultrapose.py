import json
import cv2
import numpy as np
import os
import torch
import typing
import pycocotools.mask as mask_util

class Ultrapose(torch.utils.data.Dataset):
    def __init__(self,
        json_path:      str,

    ):
        data = json.load(open(json_path))
        root = os.path.join(os.path.dirname(json_path), os.pardir)
        folder = f"{'train' if 'train' in os.path.basename(json_path) else 'val'}2014"
        self.filenames, self.keypoints, self.denseposes, self.bboxes = [], [], [], []
        for i, a in zip(data['images'], data['annotations']):
            if i['id'] != a['image_id']:
                print(i)
            self.filenames.append(os.path.join(root, folder, i['file_name']))
            self.keypoints.append(np.array(a['keypoints']).reshape(-1, 3))
            self.denseposes.append(a['dp_masks'])
            self.bboxes.append(np.array(a['bbox'], dtype=np.int32))


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: str) -> typing.Dict[str, torch.Tensor]:
        filename = self.filenames[index]
        keypoints = self.keypoints[index]
        densepose = self.denseposes[index]
        bbox = self.bboxes[index]
        img = cv2.imread(filename) / 255.0
        h, w, c = img.shape
        mask = np.zeros((256, 256), dtype=np.int64)
        for i in range(1, 15):
            current = densepose[i - 1]
            if current:
                mask[mask_util.decode(current) > 0] = i
        bx, by, bw, bh = bbox
        bbox = np.array([bx, by, bx + bw - 1, by + bh - 1], dtype=np.float32)
        cx, cy = bx + bw * 0.5, by + bh * 0.5
        affine_image = cv2.getRotationMatrix2D((cx, cy), 0.0, 1.0)
        affine_image[:, 2] += np.array([w * 0.5 - cx, h * 0.5 - cy])
        affine_mask = cv2.getRotationMatrix2D((bw * 0.5, bh * 0.5), 0.0, 1.0)
        affine_mask[:, 2] += np.array([w * 0.5 - bw * 0.5, h * 0.5 - bh * 0.5])

        img = cv2.warpAffine(img, affine_image, (w, h))
        mask = cv2.warpAffine(
            cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_NEAREST),
            affine_mask, (w, h), flags=cv2.INTER_NEAREST
        )
        keypoints -= np.array([[bx, by, 0]], dtype=np.int32)
        points = keypoints[:, :2]
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(points, affine_mask, points.shape)
        keypoints[:, :2] = np.squeeze(points)
        return {
            'color': torch.from_numpy(img.transpose(2, 0, 1)),
            'densepose': torch.from_numpy(mask)[np.newaxis, ...],
            'keypoints': torch.from_numpy(keypoints),
        }


if __name__ == '__main__':
    import tqdm
    from dataset.base_dataset import BaseDataset
    __COLOR_MAP__ = [
        [0, 0, 0], [255, 200, 200], [255, 0, 200], [255, 0, 100],
        [200, 0, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0],
        [0, 100, 255], [255, 155, 100], [255, 200, 255], [0, 200, 255], 
        [100, 0, 0], [100, 100, 0], [0, 100, 100], [50, 100, 255]
    ]
    # __ANNOTATIONS_FILE__ = r'D:\Data\ultrapose\annotations\densepose_train2014.json'
    __ANNOTATIONS_FILE__ = r'D:\Data\ultrapose\annotations\densepose_valminusminival2014.json'
    output_folder = os.path.join(__ANNOTATIONS_FILE__, os.pardir, os.pardir, 'overlay')
    os.makedirs(output_folder, exist_ok=True)
    dataset = Ultrapose(json_path=__ANNOTATIONS_FILE__)
    for i, data in tqdm.tqdm(enumerate(dataset), desc="Playback"):
        img = data['color'].numpy().transpose(1, 2, 0)
        # img = np.array(Image.fromarray((img * 255.0).astype(np.uint8)).resize((256, 256)))
        mask = data['densepose'].numpy().squeeze()
        seg = np.zeros(img.shape, dtype=np.uint8)
        for index in range(15):
            seg[mask == index] = __COLOR_MAP__[index]
        img = (255.0 * img).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f"{i}_img.png"), img)
        cv2.imwrite(os.path.join(output_folder, f'{i}_seg.png'), seg)
        silhouette = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_folder, f'{i}_mask.png'), silhouette)
        merged = np.where(
            mask[..., np.newaxis].repeat(3, axis=-1) > 0,
            0.65 * seg + 0.35 * img, 
            img
        )
        cv2.imwrite(os.path.join(output_folder, f'{i}_merged.png'), merged)
        h, w = img.shape[:2]
        pose_img = BaseDataset.draw_connect_keypoints(
            data['keypoints'].numpy()[np.newaxis, ...], w, h
        )
        pose_img = np.where(
            np.logical_and(
                np.logical_and(pose_img[..., 0] > 0, pose_img[..., 1] > 0),
                pose_img[..., 2] > 0
            )[..., np.newaxis].repeat(3, axis=-1),
            0.85 * pose_img + 0.15 * img, 
            img
        )
        cv2.imwrite(os.path.join(output_folder, f'{i}_keypoints.png'), pose_img)

