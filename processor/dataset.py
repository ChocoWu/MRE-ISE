import pickle
import random
import os

import numpy as np
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from transformers import CLIPTokenizer
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops
import logging
import cv2
from processor.create_bow import extract_tbow_features, extract_vbow_features


logger = logging.getLogger(__name__)


def printf(param, name):
    print(name, param)


class Vocabulary():
    def __init__(self):
        self.UNK = 'UNK'
        self.PAD = 'PAD'
        self.vocab = {self.UNK: 0, self.PAD: 1}
        self.rev_vocab = {0: self.UNK, 1: self.PAD}

    def build_vocab(self, data):
        for m in data:
            for n in m:
                if n not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[n] = idx
                    self.rev_vocab[idx] = n

    def id2token(self, idx):
        return self.rev_vocab.get(idx) if self.rev_vocab.get(idx) else self.rev_vocab.get(0)

    def token2id(self, token):
        return self.vocab.get(token) if self.vocab.get(token) else self.vocab.get(self.UNK)


def construct_adjacent_matrix(relation, seq_len):
    matrix = torch.tensor([[0 for _ in range(seq_len)] for _ in range(seq_len)], dtype=torch.float)
    for i, r in enumerate(relation):
        matrix[i][i] = 1
        if r != 0:
            matrix[i][r - 1] = 1
    return matrix


class MMREProcessor(object):
    def __init__(self, data_path, re_path, img_path, vit_name,
                 visual_bow_size=2000, textual_bow_size=2000,
                 clip_processor=None):
        self.data_path = data_path
        self.re_path = re_path
        self.img_path = img_path
        self.visual_bow_size = visual_bow_size
        self.textual_bow_size = textual_bow_size
        self.vit_name = vit_name
        self.tokenizer = CLIPTokenizer.from_pretrained(vit_name, do_lower_case=True)
        self.clip_processor = clip_processor

    def load_from_json(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        words, relations, heads, tails, imgids, dataid, VSG, TSG = [], [], [], [], [], [], [], []
        with open(os.path.join(load_file)) as f:
            lines = json.load(f)
            for i, line in enumerate(lines):
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])  # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                VSG.append(line['VSG'])
                TSG.append(line['TSG'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids)) == len(VSG) == len(TSG)

        vbow_features, vbow_id2token, vocab, visual_word = extract_vbow_features(self.data_path[mode], self.data_path['vbow'],
                                                             visual_bow_size=self.visual_bow_size,
                                                             original_img_dir=self.img_path['train'],
                                                             clip_version=self.vit_name)
        # file_path, textual_word_path, textual_bow_size
        tbow_features, tbow_id2token = extract_tbow_features(self.data_path[mode], self.data_path['tbow'],
                                                             textual_bow_size=self.textual_bow_size)
        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'VSG': VSG, 'TSG': TSG, 'dataid': dataid,
                'vbow_features': vbow_features, 'vbow_id2token': vbow_id2token,
                'tbow_features': tbow_features, 'tbow_id2token': tbow_id2token}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id


class NewMMREDatasetForIB(Dataset):
    def __init__(self, processor, transform, img_path=None, max_seq=40,
                 mode="train", max_tobj_num=40, max_vobj_num=40) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.mode = mode
        self.data_dict = self.processor.load_from_json(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.re2id = self.processor.get_rel2id(self.processor.data_path[mode])
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.max_tobj_num = max_tobj_num
        self.max_vobj_num = max_vobj_num

        self.text_bow_size = len(self.data_dict['tbow_id2token'])
        self.visual_bow_size = len(self.data_dict['vbow_id2token'])
        # self.tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(),
        #                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                     self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                     self.data_dict['imgids'][idx]
        _relation = self.re2id[idx]
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        head_tail_pos = torch.tensor(head_d['pos'] + tail_d['pos'])
        head_object_tokens = self.tokenizer.tokenize(head_d['name'])
        tail_object_tokens = self.tokenizer.tokenize(tail_d['name'])

        tokens = [self.tokenizer.tokenize(word) for word in word_list]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = self.tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = [self.tokenizer.cls_token_id] + _bert_inputs + [self.tokenizer.sep_token_id]
        input_ids = np.zeros(self.max_seq, np.int)
        input_ids[:len(_bert_inputs)] = _bert_inputs
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.zeros(self.max_seq, dtype=torch.long)
        attention_mask[:len(_bert_inputs)] = 1
        token_type_ids = torch.zeros(self.max_seq, dtype=torch.long)

        length = len(word_list)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        start = 0
        for i, pieces in enumerate(tokens):
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start + len(pieces)))
            _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
            start += len(pieces)
        # max_pie = np.max([len(x) for x in tokens])
        pieces2word = np.zeros((self.max_seq, self.max_seq), dtype=np.bool)
        pieces2word[:_pieces2word.shape[0], :_pieces2word.shape[1]] = _pieces2word
        pieces2word = torch.tensor(pieces2word)

        re_label = self.re_dict[relation]  # label to id

        # adjacent matrix for TSG
        t_objects = self.data_dict['TSG'][idx]['obj']   # list
        t_attributes = self.data_dict['TSG'][idx]['attr']  # list
        assert len(t_objects) == len(t_attributes)
        t_relations = ['None'] + self.data_dict['TSG'][idx]['rel']  # list[obj, rel, obj]
        t_objects_tokens = torch.tensor([self.tokenizer.tokenize(obj) for obj in t_objects], dtype=torch.long)
        t_attributes_tokens = torch.tensor([self.tokenizer.tokenize(' '.join(attr.reverse())) for attr in t_attributes], dtype=torch.long)
        t_relations_tokens = torch.tensor([self.tokenizer.tokenize(' '.join(rel)) for rel in t_relations], dtype=torch.long)
        # t_relations_tokens = [rel for rel in t_relations_tokens if len(rel) > 0]
        TSG_edge_index = torch.tensor([[t_objects.index(rel[0]), t_objects.index(rel[-1])] for rel in t_relations], dtype=torch.long)
        TSG_edge_attr = torch.tensor([idx+1 for idx, _ in enumerate(t_relations)], dtype=torch.long)
        # TSG_edge_index = add_self_loops(TSG_edge_index)[0]
        TSG_adj_matrix = to_dense_adj(TSG_edge_index, edge_attr=TSG_edge_attr).squeeze()

        TSG_edge_mask = torch.zeros(self.max_tobj_num, self.max_tobj_num)
        TSG_edge_mask[:len(t_objects), :len(t_objects)] = 1

        # adjacent matrix for VSG
        v_objects = self.data_dict['VSG'][idx]['bbox']  # already adding a default bbox for the whole image [0, 0, width, height]
        v_relations = ['similar'] + self.data_dict['VSG'][idx]['rel']  # 'similar' is a default relation between TSG and VSG
        v_attributes = self.data_dict['VSG'][idx]['bbox_attri']
        v_attributes_tokens = torch.tensor([self.tokenizer.tokenize(attr) for attr in v_attributes], dtype=torch.long)
        v_relations_tokens = torch.tensor([self.tokenizer.tokenize(rel['name']) for rel in v_relations], dtype=torch.long)
        VSG_edge_index = torch.tensor([[rel['s_index'], rel['o_index']] for rel in v_relations], dtype=torch.long)
        VSG_edge_attr = torch.tensor([idx+1 for idx, _ in enumerate(v_relations)], dtype=torch.long)
        # VSG_edge_index = add_self_loops(VSG_edge_index)[0]
        VSG_adj_matrix = to_dense_adj(VSG_edge_index, edge_attr=VSG_edge_attr).squeeze()

        VSG_edge_mask = torch.zeros(self.max_vobj_num, self.max_vobj_num)
        VSG_edge_mask[:len(v_objects), :len(v_objects)] = 1

        # text_bow features
        tbow_features = self.data_dict['tbow_features'][idx]

        # visual_bow features
        vbow_features = self.data_dict['vbow_features'][idx]

        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                # image = self.transform(image)
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
                
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            v_objects_tokens = []
            for b in v_objects:
                crop_img = cv2.imread(img_path)
                crop_region = crop_img[b[1]:b[3], b[0]:b[2]]
                im = Image.fromarray(crop_region, mode="RGB")
                # print(im.size)
                bbox = self.clip_processor(images=im, return_tensors="pt")['pixel_values'].squeeze()
                v_objects_tokens.append(bbox)
        

        return {"input_ids": input_ids, "pieces2word": pieces2word, "attention_mask": attention_mask,"token_type_ids": token_type_ids,
                "head_tail_pos": head_tail_pos, "head_object_tokens": head_object_tokens, "tail_object_tokens": tail_object_tokens,
                "t_objects_tokens": t_objects_tokens, "t_attributes_tokens": t_attributes_tokens, "t_relations_tokens": t_relations_tokens,
                "v_objects_tokens": v_objects_tokens, "v_attributes_tokens": v_attributes_tokens, "v_relations_tokens": v_relations_tokens,
                "re_label": torch.tensor(re_label), "image": image, "TSG_adj_matrix": TSG_adj_matrix, "VSG_adj_matrix": VSG_adj_matrix,
                "TSG_edge_mask": TSG_edge_mask, "VSG_edge_mask": VSG_edge_mask, "vbow_features": vbow_features, "tbow_features": tbow_features,
                "item_id": item_id, "_relation": _relation}


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = padded_stack([s[key] for s in batch])

    return padded_batch



