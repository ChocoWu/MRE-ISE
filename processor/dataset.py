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

        # file_path, visual_word_path, visual_bow_size, original_img_dir,  clip_version = "openai/clip-vit-base-patch32"
        vbow_features, vbow_id2token = extract_vbow_features(self.data_path[mode], self.data_path['vbow'],
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
                 mode="train", max_obj_num=40) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.mode = mode
        self.data_dict = self.processor.load_from_json(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.max_obj_num = max_obj_num

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
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        head_tail_pos = torch.tensor(head_d['pos'] + tail_d['pos'])

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

        dep_head = [k if i - 1 < 0 else i - 1 for k, i in enumerate(self.data_dict['dep_head'][idx])]
        dep_tail = [i for i in range(0, len(dep_head))]
        edge_index = torch.tensor([dep_head, dep_tail], dtype=torch.long)
        edge_index = add_self_loops(edge_index)[0]
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=self.max_seq).squeeze()

        edge_mask = torch.zeros(self.max_seq + self.max_obj_num, self.max_seq + self.max_obj_num)
        edge_mask[:length, :length] = 1
        edge_mask[self.max_seq + self.max_obj_num:, :length] = 1
        edge_mask[self.max_seq:self.max_seq + self.max_obj_num] = 1
        edge_mask[self.max_seq:self.max_seq + self.max_obj_num, self.max_seq:self.max_seq + self.max_obj_num] = 1

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
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                # padding
                aux_mask = torch.tensor([1 for _ in range(len(aux_imgs))] + [0 for _ in range(3 - len(aux_imgs))])
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                return input_ids, pieces2word, attention_mask, token_type_ids, adj_matrix, head_tail_pos, torch.tensor(
                    re_label), image, aux_imgs, aux_mask, edge_mask, vbow_features, tbow_features

            return input_ids, pieces2word, attention_mask, token_type_ids, adj_matrix, head_tail_pos, torch.tensor(
                re_label), image, edge_mask, vbow_features, tbow_features

        return input_ids, pieces2word, attention_mask, token_type_ids, adj_matrix, head_tail_pos, torch.tensor(
            re_label), edge_mask, vbow_features, tbow_features


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

# def collate_fn_padding(batch):
#     data_types = len(batch[0])
#     bsz = len(batch)
#
#     for i in range(data_types):
#         samples = [x for b in range(bsz) for x in batch[b][i]]
#         if not batch[0][i].shape:
#             padded_batch[key] = torch.stack(samples)
#         else:
#             padded_batch[key] = padded_stack([s[key] for s in batch])
#
#     return padded_batch
#
#     padded_batch = dict()
#     keys = batch[0].keys()



