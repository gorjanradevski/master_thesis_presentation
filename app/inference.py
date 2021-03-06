import re
import os
from typing import Dict, List
from tensorflow.contrib.training import HParams
from ruamel.yaml import YAML
from app.lightweightmodel import MultiHopAttentionLightweightModel
import numpy as np
import tensorflow as tf

import absl.logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# https://github.com/abseil/abseil-py/issues/99
absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")


def preprocess_caption(caption: str) -> str:
    """Basic method used around all classes

    Performs pre-processing of the caption in the following way:

    1. Converts the whole caption to lower case.
    2. Removes all characters which are not letters.

    Args:
        caption: A list of words contained in the caption.

    Returns:

    """
    caption = caption.lower()
    caption = re.sub("[^a-z' ]+", "", caption)
    caption = re.sub("\s+", " ", caption).strip()  # NOQA
    caption = caption.strip()

    return caption


class FlickrDataset:
    # Adapted for working with the Flickr8k and Flickr30k dataset.

    def __init__(self, images_path: str, texts_path: str):
        self.img_path_caption = self.parse_captions_filenames(texts_path)
        self.images_path = images_path

    @staticmethod
    def parse_captions_filenames(texts_path: str) -> Dict[str, List[str]]:
        """Creates a dictionary that holds:

        Key: The full path to the image.
        Value: A list of lists where each token in the inner list is a word. The number
        of sublists is 5.

        Args:
            texts_path: Path where the text doc with the descriptions is.

        Returns:
            A dictionary that represents what is explained above.

        """
        img_path_caption: Dict[str, List[str]] = {}
        with open(texts_path, "r") as file:
            for line in file:
                line_parts = line.split("\t")
                image_tag = line_parts[0].partition("#")[0]
                caption = line_parts[1]
                if image_tag not in img_path_caption:
                    img_path_caption[image_tag] = []
                img_path_caption[image_tag].append(preprocess_caption(caption))

        return img_path_caption

    @staticmethod
    def get_data_wrapper(
        imgs_file_path: str,
        img_path_caption: Dict[str, List[str]],
        images_dir_path: str,
    ):
        """Returns the image paths, the captions and the lengths of the captions.

        Args:
            imgs_file_path: A path to a file where all the images belonging to the
            validation part of the dataset are listed.
            img_path_caption: Image name to list of captions dict.
            images_dir_path: A path where all the images are located.

        Returns:
            Image paths, captions and lengths.

        """
        image_paths = []
        captions = []
        with open(imgs_file_path, "r") as file:
            for image_name in file:
                # Remove the newline character at the end
                image_name = image_name[:-1]
                # If there is no specified codec in the name of the image append jpg
                if not image_name.endswith(".jpg"):
                    image_name += ".jpg"
                for i in range(5):
                    image_paths.append(os.path.join(images_dir_path, image_name))
                    captions.append(img_path_caption[image_name][i])

        assert len(image_paths) == len(captions)

        return image_paths, captions

    def get_data(self, images_file_path: str):
        image_paths, captions = self.get_data_wrapper(
            images_file_path, self.img_path_caption, self.images_path
        )

        return image_paths, captions


class YParams(HParams):
    def __init__(self, hparams_path: str):
        super().__init__()
        with open(hparams_path) as fp:
            for k, v in YAML().load(fp).items():
                self.add_hparam(k, v)


hparams = YParams("app/static/hyperparameters/flickr8k.yaml")
dataset = FlickrDataset(
    "app/static/Flickr8k_dataset/Flickr8k_Dataset",
    "app/static/Flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt",
)
test_image_paths, _ = dataset.get_data(
    "app/static/Flickr8k_dataset/Flickr8k_text/Flickr_8k.testImages.txt"
)
images = np.load(open("app/static/embedded_images.pkl", "rb"), allow_pickle=True)
tf.reset_default_graph()
model = MultiHopAttentionLightweightModel(
    hparams.joint_space, hparams.num_layers, hparams.attn_size, hparams.attn_hops
)
sess = tf.Session()
model.init(sess, "app/static/models/siameseRTDUO")


def predict(query: str):
    print(query)
    embedded_query = sess.run(
        model.attended_captions, feed_dict={model.captions: query}
    )
    similarities = np.dot(embedded_query, images.T).flatten()
    indices = np.argsort(similarities)[::-1]
    retrieved_image_paths = [
        "/".join(test_image_paths[index].split("/")[1:]) for index in indices[:25][0::5]
    ]

    return retrieved_image_paths
