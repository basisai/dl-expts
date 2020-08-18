"""
Script for serving.
"""
import base64
import os

import cv2
import numpy as np
import gluonnlp as nlp
import mxnet as mx
from flask import Flask, request

from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, EOS, BOS
from ocr.utils.denoiser_utils import SequenceGenerator
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import (
    Network as HandwritingRecognitionNet,
    handwriting_recognition_transform,
    decode as decoder_handwriting,
)

MODEL_DIR = "/artefact/"
if os.path.exists("models/"):
    MODEL_DIR = "models/"

WORD_SEGMENTATION_MODEL = MODEL_DIR + "word_segmentation2.params"
HANDWRITING_RECOGNITION_MODEL = MODEL_DIR + "handwriting_line8.params"
DENOISER_MODEL = MODEL_DIR + "denoiser2.params"

FEATURE_LEN = 150
ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()


def load_segmentation_net():
    """Segmentation model."""
    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters(WORD_SEGMENTATION_MODEL)
    word_segmentation_net.hybridize()
    return word_segmentation_net


def load_recognition_net():
    """Recognition model."""
    line_recognition_net = HandwritingRecognitionNet(
        rnn_hidden_states=512, rnn_layers=2, ctx=ctx, max_seq_len=160)
    line_recognition_net.load_parameters(HANDWRITING_RECOGNITION_MODEL, ctx=ctx)
    line_recognition_net.hybridize()
    return line_recognition_net


def load_denoiser():
    """Denoising model."""
    denoiser = Denoiser(alphabet_size=len(ALPHABET),
                        max_src_length=FEATURE_LEN,
                        max_tgt_length=FEATURE_LEN,
                        num_heads=16,
                        embed_size=256,
                        num_layers=2)
    denoiser.load_parameters(DENOISER_MODEL, ctx=ctx)
    denoiser.hybridize(static_alloc=True)

    beam_sampler = nlp.model.BeamSearchSampler(beam_size=20,
                                               decoder=denoiser.decode_logprob,
                                               eos_id=EOS,
                                               scorer=nlp.model.BeamSearchScorer(),
                                               max_length=150)

    language_model, vocab = nlp.model.big_rnn_lm_2048_512(dataset_name="gbw",
                                                          pretrained=True,
                                                          ctx=ctx)

    generator = SequenceGenerator(beam_sampler,
                                  language_model,
                                  vocab,
                                  ctx,
                                  nlp.data.SacreMosesTokenizer(),
                                  nlp.data.SacreMosesDetokenizer())
    return denoiser, generator


word_segmentation_net = load_segmentation_net()
line_recognition_net = load_recognition_net()
denoiser, generator = load_denoiser()


def segment(paragraph_segmented_image):
    """Perform segmentation of lines."""
    # Word segmentation
    min_c = 0.1
    overlap_thres = 0.1
    topk = 600
    predicted_words_bbs = predict_bounding_boxes(
        word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)

    # Line segmentation
    line_bbs = sort_bbs_line_by_line(predicted_words_bbs, y_overlap=0.4)
    line_images = crop_line_images(paragraph_segmented_image, line_bbs)
    return line_images, line_bbs


def get_arg_max(prob):
    """Converts the output of the handwriting recognition network into strings."""
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


def recognize(line_images):
    """Perform recognition for each line image."""
    line_image_size = (60, 800)
    decoded_lines_am = []
    for j, line_image in enumerate(line_images):
        line_image = handwriting_recognition_transform(line_image, line_image_size)
        line_character_probs = line_recognition_net(line_image.as_in_context(ctx))
        decoded_lines_am.append(get_arg_max(line_character_probs))
    return decoded_lines_am


def get_denoised(text, denoiser, generator):
    """Denoise output."""
    src_seq, src_valid_length = encode_char(text)
    src_seq = mx.nd.array([src_seq], ctx=ctx)
    src_valid_length = mx.nd.array(src_valid_length, ctx=ctx)
    encoder_outputs, _ = denoiser.encode(src_seq, valid_length=src_valid_length)
    states = denoiser.decoder.init_state_from_encoder(encoder_outputs,
                                                      encoder_valid_length=src_valid_length)
    inputs = mx.nd.full(shape=(1,), ctx=src_seq.context, dtype=np.float32, val=BOS)
    output = generator.generate_sequences(inputs, states, text)
    return output.strip()

    
def denoise(decoded_lines_am):
    """Perform denoising for each decoded line."""
    decoded_lines = []
    for j, decoded_line_am in enumerate(decoded_lines_am):
        decoded_lines.append(get_denoised(decoded_line_am, denoiser, generator))
    return decoded_lines


def decode_image(field):
    """Decode a base64 encoded image to a list of floats.
    Args:
        field: base64 encoded string
    Returns:
        numpy.array
    """
    array = np.frombuffer(base64.b64decode(field), dtype=np.uint8)
    image_array = cv2.imdecode(array, cv2.IMREAD_ANYCOLOR)
    return image_array


def predict(request_json):
    """Predict function."""
    raw_img = decode_image(request_json["encoded_image"]).reshape(
        request_json["image_shape"])

    image = np.asarray(raw_img.convert("L"))
    line_images, line_bbs = segment(image)
    decoded_lines_am = recognize(line_images)
    decoded_lines = denoise(decoded_lines_am)
    return " ".join(decoded_lines)

            
# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns decoded text."""
    return {"decoded_text": predict(request.json)}


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
