from app.SMHA.src.utils.datasets import FlickrDataset
from app.SMHA.src.hyperparameters import YParams
from app.lightweightmodel import MultiHopAttentionLightweightModel
import numpy as np


hparams = YParams("hyperparameters/flickr8k.yaml")
dataset = FlickrDataset(
    "data/Flickr8k_dataset/Flickr8k_Dataset",
    "data/Flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt",
)
test_image_paths, _ = dataset.get_data(
    "data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.testImages.txt"
)
images = np.load(open("data/embedded_images.pkl"))
tf.reset_default_graph()
model = MultiHopAttentionLightweightModel(
    hparams.joint_space, hparams.num_layers, hparams.attn_size, hparams.attn_hops
)
sess = tf.Session()
model.init(sess, "models/siameseRTDUO")


def predict(query: string):
    embedded_query = sess.run(
        model.attended_captions, feed_dict={model.captions: query}
    )
    similarities = np.dot(embedded_query, images.T).flatten()
    indices = np.argsort(similarities)[::-1]
    retrieved_image_path = test_image_paths[indices[0]]

    return retrieved_image_path
