import numpy as np
import os
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import pickle
from transformers import CLIPModel, CLIPProcessor
import torch
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as stop_words
from scipy.cluster.vq import kmeans, vq

def create_tbow(data, textual_bow_size, target_file, stopwords_language="english"):
    """
    create text bow vocabulary
    """
    stopwords = set(stop_words.words(stopwords_language))
    vectorizer = CountVectorizer(max_features=textual_bow_size, stop_words=stopwords)
    text_for_bow = []
    for d in data:
        text_for_bow.append(' '.join(d['token']))
    vectorizer.fit(text_for_bow)
    vocab = vectorizer.get_feature_names()
    with open(target_file, 'wb') as f:
        pickle.dump([vocab, vectorizer], f)
    return vocab


def create_vbow(data, visual_bow_size, mode, original_img_dir, target_file,
                clip_version="openai/clip-vit-base-patch32"):
    """
    create Visual words
    :param data: input data.
    :param visual_bow_size: the vocabulary size of visual bow.
    :param mode: 'train' / 'val' / 'test'
    :param target_file: the final target file to storage visual words.
    :param clip_version: the vision of pre-trained clip model.
    """
    print('prepare vision features extractor ...')
    vision_model = CLIPModel.from_pretrained(clip_version)
    for name, param in vision_model.named_parameters():
        param.requires_grad = False
    vision_model.eval()
    processor = CLIPProcessor.from_pretrained(clip_version)
    if torch.cuda.is_available():
        vision_model.to(torch.device('cuda'))

    print('extract the visual words')
    with torch.no_grad():
        des_features = []
        for d in tqdm(data, total=len(data)):
            imgid = d['img_id']
            img_path = os.path.join(original_img_dir, mode, imgid)
            bbox = d['VSG']['bbox']
            for b in bbox:
                crop_img = cv2.imread(img_path)
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
                des_features.append(image_features.numpy())
    kmeans = KMeans(n_clusters=visual_bow_size, random_state=0, n_init=10)
    img_cluster = kmeans.fit(np.array(des_features))
    visual_word = img_cluster.cluster_centers_  # ndarray of shape (n_clusters, n_features)
    # labels = img_cluster.labels_
    with open(target_file, 'wb') as f:
        pickle.dump([visual_word, kmeans], f)
    return visual_word


def extract_visual_words(vision_model, processor, data, visual_bow_size, original_img_dir, target_file):
    """
        create Visual words
        :param data: input data.
        :param visual_bow_size: the vocabulary size of visual bow.
        :param target_file: the final target file to storage visual words.
    """
    print('extract the visual words')
    with torch.no_grad():
        des_features = []
        for d in tqdm(data, total=len(data)):
            imgid = d['img_id']
            img_path = os.path.join(original_img_dir, imgid)
            bbox = d['VSG']['bbox']
            for b in bbox:
                crop_img = cv2.imread(img_path)
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
                des_features.append(image_features.numpy())
    kmeans = KMeans(n_clusters=visual_bow_size, random_state=0, n_init=10)
    img_cluster = kmeans.fit(np.array(des_features))
    visual_word = img_cluster.cluster_centers_  # ndarray of shape (n_clusters, n_features)
    # labels = img_cluster.labels_
    id2token = {}
    vocab = []
    for idx in range(visual_word.shape(0)):
        vocab.append('vword_' + str(idx))
    for k, v in zip(range(0, len(vocab)), vocab):
        id2token[k] = v

    with open(target_file, 'wb') as f:
        pickle.dump([img_cluster, vocab, id2token, visual_word], f)
    return img_cluster, vocab, id2token, visual_word


def extract_vbow_features(file_path, visual_word_path, visual_bow_size, original_img_dir,
                          clip_version="openai/clip-vit-base-patch32"):
    """
    extract visual bow features
    :param file_path: the input data file path.
    :param visual_word_path: the visual word file path.
    :param visual_bow_size: the vocabulary size of visual bow.
    :param original_img_dir: the original image directory.
    :param clip_version: the vision of pre-trained clip model.
    :return: 
        visual bow features: the bow features for each bbox,  [the number of bbox, visual_bow_size]
        id2token: dict, the visual word vocabulary.
        vocab: the visual word vocabulary.
        visual_word: the feature of each cluster centers.
    """
    print('prepare vision features extractor ...')
    vision_model = CLIPModel.from_pretrained(clip_version)
    for name, param in vision_model.named_parameters():
        param.requires_grad = False
    vision_model.eval()
    processor = CLIPProcessor.from_pretrained(clip_version)
    if torch.cuda.is_available():
        vision_model.to(torch.device('cuda'))

    print('extract visual bow features .....')
    with open(file_path, 'r') as f:
        data = json.load(f)
    if os.path.exists(visual_word_path):
        with open(visual_word_path, 'rb') as f:
            img_cluster, vocab, id2token, visual_word = pickle.load(f)
    else:
        # data, visual_bow_size, mode, original_img_dir, target_file,
        # clip_version = "openai/clip-vit-base-patch32"
        img_cluster, vocab, id2token, visual_word = extract_visual_words(vision_model, processor,
                                                                         data, visual_bow_size, original_img_dir,
                                                                         visual_word_path)
    des_list = []
    for d in tqdm(data, total=len(data)):
        imgid = d['img_id']
        img_path = os.path.join(original_img_dir, 'train', imgid)
        bbox = d['VSG']['bbox']
        features_list = []
        for b in bbox:
            crop_img = cv2.imread(img_path)
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
            features_list.append(image_features.numpy())
        des_list.append((d['img_id'], d['VSG']['bbox'], np.array(features_list)))

    vbow_features = np.zeros((len(des_list), len(visual_word)), "float32")
    for i in tqdm(range(len(des_list)), total=len(des_list)):
        words, distance = vq(des_list[i][2], visual_word)
        assert len(words) == len(distance) == len(des_list[i][1])
        for w in words:
            vbow_features[i][w] += 1

    return vbow_features, id2token, vocab, visual_word


def extract_text_bow_vocab(train_file_path, target_file, textual_bow_size, stopwords_language="english"):
    with open(train_file_path, 'rb') as f:
        data = json.load(f)
    text_for_bow = []
    for d in tqdm(data, total=len(data)):
        text_for_bow.append(' '.join(d['token']))
    stopwords = set(stop_words.words(stopwords_language))
    vectorizer = CountVectorizer(max_features=textual_bow_size, stop_words=stopwords)
    vectorizer.fit(text_for_bow)
    # train_bow_embeddings = vectorizer.fit_transform(text_for_bow)
    vocab = vectorizer.get_feature_names()
    id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}
    with open(target_file, 'wb') as f:
        pickle.dump([vectorizer, vocab, id2token], f)
    return vectorizer, vocab, id2token


def extract_tbow_features(file_path, textual_word_path, textual_bow_size):
    print('extract textual bow features .....')
    if os.path.exists(textual_word_path):
        with open(textual_word_path, 'rb') as f:
            vectorizer, vocab, id2token = pickle.load(f)
    else:
        vectorizer, vocab, id2token = extract_text_bow_vocab(file_path, textual_word_path, textual_bow_size)
    with open(file_path, 'rb') as f:
        data = json.load(f)
    text_for_bow = []
    for d in tqdm(data, total=len(data)):
        text_for_bow.append(' '.join(d['token']))
    tbow_features = vectorizer.transform(text_for_bow)
    return tbow_features, id2token


if __name__ == '__main__':

    FILE_DIR = '../data/vsg_tsg/'
    with open(os.path.join(FILE_DIR, 'ours_train.json'), 'r') as f:
        data = json.load(f)

    print('create textual bow')
    target_tbow = 'tbow.pt'
    textual_bow_size = 2000
    create_tbow(data, textual_bow_size, os.path.join(FILE_DIR, target_tbow))

    print('create visual bow')
    target_tbow = 'vbow.pt'
    IMG_DIR = '../data/img_org/'
    visual_bow_size = 2000
    create_vbow(data, visual_bow_size, 'train', IMG_DIR, os.path.join(FILE_DIR, target_tbow))




