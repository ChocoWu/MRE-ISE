import argparse
import logging
import sys
sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from cores.gene.model import MRE
from transformers import CLIPProcessor, CLIPModel

from transformers import CLIPConfig
from processor.dataset import MMREProcessor, NewMMREDatasetForIB
from cores.gene.model import Trainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASS = {
    'bert': (MMREProcessor, NewMMREDatasetForIB),
}


DATA_PATH = {
        'MRE': {'train': 'data/vsg_tsg/ours_train.json',
                'dev': 'data/vsg_tsg/ours_valid.json',
                'test': 'data/vsg_tsg/ours_test.json',
                'vbow': 'data/vsg_tsg/vbow.pk',
                'tbow': 'data/vsg_tsg/tbow.pk'
                }
}

IMG_PATH = {
        'MRE': {'train': 'data/img_org/train/',
                'dev': 'data/img_org/val/',
                'test': 'data/img_org/test'}}



def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_name', default='openai/clip-vit-base-patch32', type=str, help="The name of pretrained model")
    parser.add_argument('--dataset_name', default='MRE', type=str, help="The name of example_dataset.")
    parser.add_argument('--num_epochs', default=30, type=int, help="Training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr_pretrained', default=2e-5, type=float, help="pre-trained learning rate")
    parser.add_argument('--lr_main', default=2e-4, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int)
    parser.add_argument('--seed', default=1234, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default='ckpt', type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_true')  # , action='store_true'
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--max_seq', default=40, type=int)
    parser.add_argument('--max_obj', default=40, type=int)

    parser.add_argument('--hid_size', default=768, type=int, help="hidden state size")
    parser.add_argument('--num_layers', default=2, type=int, help="number of refine layers")
    parser.add_argument('--beta', default=0.01, type=float, help="Default is 1e-2")
    parser.add_argument("--num_per", type=int, default=16, help="Default is 16")
    parser.add_argument("--feature_denoise", type=bool, default=True, help="Default is False.")
    parser.add_argument("--top_k", type=int, default=10, help="Default is 10.")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Default is 0.3.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Default is 0.1.")
    parser.add_argument("--graph_skip_conn", type=float, default=0.0, help="Default is 0.0.")
    parser.add_argument("--graph_include_self", type=bool, default=True, help="Default is True.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Default is 0.0")
    parser.add_argument("--graph_type", type=str, default="epsilonNN", help="epsilonNN, KNN, prob")
    parser.add_argument("--graph_metric_type", type=str, default="multi_mlp")
    parser.add_argument("--repar", type=bool, default=True, help="Default is True.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Default is 0.2.")
    parser.add_argument("--prior_mode", type=str, default="Gaussian", help="Default is Gaussian.")
    parser.add_argument("--is_IB", type=bool, default=True, help="Default is True.")
    parser.add_argument("--eta1", type=float, default=0.7, help="Default is 0.7")
    parser.add_argument("--eta1", type=float, default=0.9, help="Default is 0.9")
    parser.add_argument("--text_bow_size", type=int, default=2000, help="Default is 2000")
    parser.add_argument("--visual_bow_size", type=int, default=2000, help="Default is 2000")
    parser.add_argument("--neighbor_num", type=int, default=2, help="Default is 2")
    parser.add_argument("--topic_keywords_number", type=int, default=10, help="Default is 10")
    parser.add_argument("--topic_number", type=int, default=10, help="Default is 10")

    args = parser.parse_args()

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name]
    data_process, dataset_class = MODEL_CLASS[args.model_name]
    re_path = 'data/ours_rel2id.json'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed)  # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        args.save_path = os.path.join(args.save_path, args.model_name, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    logdir = "logs/" + args.model_name + "_"+args.dataset_name + "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    writer = SummaryWriter(log_dir=logdir)
    if args.do_train:
        clip_vit, clip_processor, aux_processor, rcnn_processor = None, None, None, None
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        aux_processor = CLIPProcessor.from_pretrained(args.vit_name)
        aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = args.aux_size, args.aux_size
        rcnn_processor = CLIPProcessor.from_pretrained(args.vit_name)
        rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = args.rcnn_size, args.rcnn_size
        clip_model = CLIPModel.from_pretrained(args.vit_name)
        clip_vit = clip_model.vision_model
        clip_text = clip_model.text_model

        processor = data_process(data_path, re_path, args.bert_name, args.vit_name, clip_processor=clip_processor, aux_processor=aux_processor, rcnn_processor=rcnn_processor)
        train_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size, rcnn_size=args.rcnn_size, mode='train', max_obj_num=args.max_obj_num)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dev_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size, rcnn_size=args.rcnn_size, mode='dev', max_obj_num=args.max_obj_num)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size, rcnn_size=args.rcnn_size, mode='test', max_obj_num=args.max_obj_num)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer

        # test
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = CLIPConfig.from_pretrained(args.vit_name).text_config

        model = MRE(args, vision_config, text_config, clip_vit, clip_text, num_labels,
                    args.text_bow_size, args.visual_bow_size, tokenizer, processor)

        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, re_dict=re_dict, model=model, args=args, logger=logger, writer=writer)
        trainer.train()
        torch.cuda.empty_cache()
        writer.close()


if __name__ == "__main__":
    main()
