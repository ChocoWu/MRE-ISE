import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import requests
import matplotlib.pyplot as plt
import json
import pickle
import ast
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import cv2
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    x = torch.tensor([img_w, img_h, img_w, img_h], dtype=out_bbox.dtype).to(torch.get_device(out_bbox))
    b = b * x
    return b


def find_repeat(new_bbox, bbox):
    flag = 0
    index = 0

    def get_list(x):
        return [i for i in range(max(0, x-5), x+5)]
    for idx, b in enumerate(bbox):
        if (new_bbox[0] in get_list(b[0])) and (new_bbox[1] in get_list(b[1])) and (new_bbox[2] in get_list(b[2])) and (new_bbox[3] in get_list(b[3])):
            flag = 1
            index = idx
            break
    return flag, index


def construct_scene_graph(data, original_img_dir, target_file, mode='train'):
    position_embedding = PositionEmbeddingSine(128, normalize=True)
    backbone = Backbone('resnet50', False, False, False)
    backbone = Joiner(backbone, position_embedding)
    backbone.num_channels = 2048

    transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                              dim_feedforward=2048,
                              num_encoder_layers=6,
                              num_decoder_layers=6,
                              normalize_before=False,
                              return_intermediate_dec=True)

    model = RelTR(backbone, transformer, num_classes=151, num_rel_classes=51,
                  num_entities=100, num_triplets=200)

    # The checkpoint is pretrained on Visual Genome
    ckpt = torch.hub.load_state_dict_from_url(
        url='https://cloud.tnt.uni-hannover.de/index.php/s/PB8xTKspKZF7fyK/download/checkpoint0149.pth',
        # map_location=lambda storage, loc: storage.cuda(0),
        check_hash=True)
    # map_location='cpu'
    model.load_state_dict(ckpt['model'])
    model.to(torch.device('cuda'))

    # Some transformation functions
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # img_list = os.listdir(dirname)
    # img_list = []
    # for d in data:
    #     img_list.append(d['img_id'])
    # img_list = ['twitter_stream_2018_10_10_25_0_2_7.jpg']
    res_list = []
    with torch.no_grad():
        model.eval()
        for d in tqdm(data, total=len(data)):
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            res = dict()
            img_id = d['img_id']
            # res['img'] = i
            im = Image.open(os.path.join(original_img_dir, mode, img_id))

            img = transform(im).unsqueeze(0)
            img = img.to(torch.device('cuda'))

            # propagate through the model

            outputs = model(img)
            # outputs = dict()
            # for k, v in res.items():
            #     outputs[k] = v.to(torch.device('cpu'))
            keep_thresh = 0.35

            # keep only predictions with >0.3 confidence
            probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
            probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
            probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
            keep = torch.logical_and(probas.max(-1).values > keep_thresh,
                                     torch.logical_and(probas_sub.max(-1).values > keep_thresh,
                                                       probas_obj.max(-1).values > keep_thresh))

            # convert boxes from [0; 1] to image scales
            sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
            obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

            # topk = 10  # display up to 10 images
            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            topk = keep_queries.size(0)  # display up to 10 images
            indices = torch.argsort(
                -probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] *
                probas_obj[keep_queries].max(-1)[
                    0])[
                      :topk]
            keep_queries = keep_queries[indices]
            bbox = [[0, 0, im.width, im.height]]
            bbox_attri = ['img']
            # s_bbox = []
            # s_bbox_attri = []
            # o_bbox = []
            # o_bbox_attri = []
            rel = [{'s_index': 0, 'o_index': 0, 'name': 'self'}]
            for idx, s, o in zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                (sxmin, symin, sxmax, symax) = [max(0, round(x)) for x in s.tolist()]
                sxmax, symax = min(im.width, sxmax), min(im.height, symax)
                (oxmin, oymin, oxmax, oymax) = [max(0, round(x)) for x in o.tolist()]
                oxmax, oymax = min(im.width, oxmax), min(im.height, oymax)
                # new_img = im.new('RGB', size=s)
                _flag, _idx = find_repeat((sxmin, symin, sxmax, symax), bbox)
                if not _flag:
                    bbox.append((sxmin, symin, sxmax, symax))
                    bbox_attri.append(CLASSES[probas_sub[idx].argmax()])
                    s_index = len(bbox)-1
                else:
                    s_index = _idx
                _flag, _idx = find_repeat((oxmin, oymin, oxmax, oymax), bbox)
                if not _flag:
                    bbox.append((oxmin, oymin, oxmax, oymax))
                    bbox_attri.append(CLASSES[probas_obj[idx].argmax()])
                    o_index = len(bbox) - 1
                else:
                    o_index = _idx
                # if (sxmin, symin, sxmax, symax) not in bbox:
                #     bbox.append((sxmin, symin, sxmax, symax))
                #     bbox_attri.append(CLASSES[probas_sub[idx].argmax()])
                # if (oxmin, oymin, oxmax, oymax) not in bbox:
                #     bbox.append((oxmin, oymin, oxmax, oymax))
                #     bbox_attri.append(CLASSES[probas_obj[idx].argmax()])
                # s_index = bbox.index((sxmin, symin, sxmax, symax))
                # o_index = bbox.index((oxmin, oymin, oxmax, oymax))
                rel_attri = REL_CLASSES[probas[idx].argmax()]
                rel.append({'s_index': s_index, 'o_index': o_index, 'name': rel_attri})
            # res['s_bbox'] = s_bbox
            # res['s_bbox_attri'] = s_bbox_attri
            # res['o_bbox'] = o_bbox
            # res['o_bbox_attri'] = o_bbox_attri
            res['bbox'] = bbox
            res['bbox_attri'] = bbox_attri
            res['rel'] = rel
            d['VSG'] = res
            res_list.append(d)
            # torch.empty_cache()

        assert len(data) == len(res_list)

        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(res_list, f)


def get_obj_features(dirname):
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if torch.cuda.is_available():
        vision_model.to(torch.device('cuda'))
    with open(os.path.join(os.path.dirname(dirname), os.path.basename(dirname).split('.')[0] + '.json')) as f:
        data = json.load(f)
    mode = os.path.basename(dirname).split('.')[0].split('_')[1]
    with torch.no_grad():
        vision_model.eval()
        for d in tqdm(data, total=len(data)):
            imgid = d['img']
            img_path = os.path.join('../../data/img_org', mode, imgid)
            bbox = d['bbox']
            features = []
            for b in bbox:
                crop_img = cv2.imread(img_path)
                # print(crop_img.shape)
                try:
                    crop_region = crop_img[b[1]:b[3], b[0]:b[2]]
                except TypeError as e:
                    print(e)
                    print(bbox)
                    print(b)
                    print(imgid)
                    exit(0)
                im = Image.fromarray(crop_region, mode="RGB")
                # print(im.size)
                images = processor(images=im, return_tensors="pt")
                images = images.to(torch.device('cuda'))
                image_features = vision_model.get_image_features(**images).squeeze()
                features.append(image_features.tolist())
            d['features'] = features
    with open(os.path.join(os.path.dirname(dirname), os.path.basename(dirname).split('.')[0] + '.pk'), 'wb') as fout:
        pickle.dump(data, fout)


if __name__ == '__main__':
    FILE_DIR = '../data/tsg/'
    IMG_DIR = '../data/img_org/'
    DIST_DIR = '../data/vsg_tsg/'
    for i in ['ours_train.json', 'ours_val.json', 'ours_test.json']:
        print(f'parsing {i} ... ')
        with open(os.path.join(FILE_DIR, i)) as f:
            data = json.load(f)
        base_name = os.path.basename(i).split('.')[0]
        # dirname = os.path.join(FILE_DIR, i)
        mode = i.split('.')[0].split('_')[1]
        print(mode)
        construct_scene_graph(data, original_img_dir=IMG_DIR, target_file=os.path.join(DIST_DIR, f'{base_name}.json'),
                              mode=mode)
        # get_obj_features(dirname)


    # data = [1, 2, 3]
    # output_file = os.path.join(os.path.dirname(dirname), os.path.basename(dirname).split('.')[0] + '.json')
    # print(output_file)
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(data, f)
    # with open(os.path.join(os.path.dirname(dirname), os.path.basename(dirname).split('.')[0] + '.pk'), 'wb') as fout:
    #     pickle.dump(data, fout)
    # x = [{'s_index': 0, 'o_index': 0, 'name': 'self'}, {'s_index': 1, 'o_index': 2, 'name': 'wearing'}, {'s_index': 1, 'o_index': 3, 'name': 'wearing'}, {'s_index': 4, 'o_index': 5, 'name': 'wearing'}, {'s_index': 4, 'o_index': 6, 'name': 'has'}, {'s_index': 4, 'o_index': 7, 'name': 'wearing'}, {'s_index': 8, 'o_index': 9, 'name': 'on'}]
    # b = [[i['s_index'], i['o_index']] for i in x]
    # print(b)
    # with open(os.path.join(os.path.dirname(dirname), os.path.basename(dirname) + '.pk'), 'wb') as fout:
    #     pickle.dump([1, 2], fout)


