import os
import sys
import timer
import imgaug
import keras
import random
import zipfile
import urllib.request
import shutil

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import tensorflow as tf
from utils import visualize
from utils.dataset import Dataset
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from nets.mrcnn import get_train_model,get_predict_model
from nets.mrcnn_training import data_generator,load_image_gt
from dataset import ShapesDataset



def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class CocoConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    NUM_CLASSES = 1 + 80


#------------------------------------------------#
#  Dataset
#------------------------------------------------#

class CocoDataset(Dataset):
    def load_coco(self, dataset_dir, subset, year=2017, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        if not class_ids:
            class_ids = sorted(coco.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))
        else:
            image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                if m.max() < 1:
                    continue
                if annotation['iscrowd']:
                    class_id *= -1
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


if __name__ == '__main__':
    config = CocoConfig()
    config.display()

    learning_rate = 1e-5
    init_epoch = 0
    epoch = 100

    MODEL_DIR = "logs"
    COCO_MODEL_PATH = "model_data/mask_rcnn_coco.h5"

    DEFAULT_LOGS_DIR = "logs"
    DEFAULT_DATASET_YEAR = "2014"

    model = get_train_model(config)

    model.load_weights(COCO_MODEL_PATH, by_name=True)

    dataset_train = CocoDataset()
    dataset_train.load_coco(r"E:\dataset\Coco_data\coco_dataset", "train", year=2017, auto_download=False)
    dataset_train.prepare()

    dataset_val = CocoDataset()
    dataset_val.load_coco(r"E:\dataset\Coco_data\coco_dataset", "val", year=2017, auto_download=False)
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    train_generator = data_generator(dataset_train, config, shuffle=True,
                                        batch_size=config.BATCH_SIZE)
    val_generator = data_generator(dataset_val, config, shuffle=True,
                                    batch_size=config.BATCH_SIZE)
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=MODEL_DIR,
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "epoch{epoch:03d}_loss{loss:.3f}_val_loss{val_loss:.3f}.h5"),
                                        verbose=0, save_weights_only=True),
    ]

    log("\nStarting at epoch {}. LR={}\n".format(init_epoch, learning_rate))
    log("Checkpoint Path: {}".format(MODEL_DIR))

    optimizer = keras.optimizers.Adam(lr=learning_rate)

    model._losses = []
    model._per_input_losses = {}
    loss_names = [
        "rpn_class_loss",  "rpn_bbox_loss",
        "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
    for name in loss_names:
        layer = model.get_layer(name)
        if layer.output in model.losses:
            continue
        loss = (
            tf.reduce_mean(layer.output, keepdims=True)
            * config.LOSS_WEIGHTS.get(name, 1.))
        model.add_loss(loss)

    reg_losses = [
        keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
        for w in model.trainable_weights
        if 'gamma' not in w.name and 'beta' not in w.name]
    model.add_loss(tf.add_n(reg_losses))

    model.compile(
        optimizer=optimizer,
        loss=[None] * len(model.outputs)
    )

    for name in loss_names:
        if name in model.metrics_names:
            print(name)
            continue
        layer = model.get_layer(name)
        model.metrics_names.append(name)
        loss = (
            tf.reduce_mean(layer.output, keepdims=True)
            * config.LOSS_WEIGHTS.get(name, 1.))
        model.metrics_tensors.append(loss)


    model.fit_generator(
        train_generator,
        initial_epoch=init_epoch,
        epochs=epoch,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=config.VALIDATION_STEPS,
        max_queue_size=100
    )


