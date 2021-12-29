import argparse
import cv2
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    # parser.add_argument('imgname', help='image file name')

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    imwrite(img, outfp)


def main():

    list_train_path = "/home/yanzuo/datasets/gather_split/v3/list_train.txt"
    list_query_path = "/home/yanzuo/datasets/gather_split/v3/list_query.txt"
    list_gallery_path = "/home/yanzuo/datasets/gather_split/v3/list_gallery.txt"
    
    with open(list_train_path, "r") as f:
        list_train = f.readlines()
        list_train = [line.strip() for line in list_train]
    with open(list_query_path, "r") as f:
        list_query = f.readlines()
        list_query = [line.strip() for line in list_query]
    with open(list_gallery_path, "r") as f:
        list_gallery = f.readlines()
        list_gallery = [line.strip() for line in list_gallery]

    args = parse_args()
    cfg = Config.fromfile(args.config)
    # imgname = args.imgname
    class_names = cfg.class_names
    engine, data_pipeline, device = prepare(cfg)

    for i, lst in enumerate([list_train, list_query, list_gallery]):
        for item in tqdm(lst):
            image_path = ' '.join(item.split(' ')[2:])
            image_dir, image_name = os.path.dirname(image_path), os.path.basename(image_path)
            save_dir = image_dir.replace("xiaotong/ReID", "yanzuo").replace(["train", "test", "test"][i], "bbox_" + ["train", "test", "test"][i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = image_name.split('.')[0] + ".pkl"
            save_path = os.path.join(save_dir, save_name)

            if os.path.isfile(save_path):
                continue
            
            data = data_pipeline(dict(img_info=dict(filename=image_path), img_prefix=None))
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, [device])[0]
            result = engine.infer(data["img"], data["img_metas"])[0]

            with open(save_path, "wb") as f:
                pickle.dump(result, f)

    # img_path1 = "/home/xiaotong/ReID/datasets/gather_split/v3/train/BJ/BBY_20210608_1900_2100/女1/a7410804-cc6d-45f4-acbf-533cc9166c36_20210608193321956_body.jpg"
    # img_path2 = "/home/xiaotong/ReID/datasets/gather_split/v3/train/BJ/BBY_20210608_1900_2100/女1/89da60d0-0e40-418d-9aad-ed81201782d4_20210608201715312_body.jpg"
    
    # data1 = data_pipeline(dict(img_info=dict(filename=img_path1), img_prefix=None))
    # data2 = data_pipeline(dict(img_info=dict(filename=img_path2), img_prefix=None))

    # data = collate([data1, data2], samples_per_gpu=1)
    # if device != 'cpu':
    #     # scatter to specified GPU
    #     data = scatter(data, [device])
    # else:
    #     # just get the actual data from DataContainer
    #     data['img_metas'] = data['img_metas'][0].data
    #     data['img'] = data['img'][0].data
    
    # item1 = scatter(data, [device])[0]
    # item2 = scatter(data, [device])[0]
    # print(item1, item2)
    # result = engine.infer(item1['img'], item1['img_metas'])
    # print(result)
    # result = engine.infer(item2['img'], item2['img_metas'])
    # print(result)
    # plot_result(result, imgname, class_names)


if __name__ == '__main__':
    main()
