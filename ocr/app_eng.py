import gluonnlp as nlp
import mxnet as mx
import numpy as np
import streamlit as st
import zipfile
from PIL import Image

from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, EOS, BOS
from ocr.utils.denoiser_utils import SequenceGenerator
from ocr.utils.cer_wer import score
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import (
    Network as HandwritingRecognitionNet,
    handwriting_recognition_transform,
    decode as decoder_handwriting,
)


with zipfile.ZipFile("/artefact/models.zip", "r") as zip_ref:
    zip_ref.extractall("/artefact/")

WORD_SEGMENTATION_MODEL = "/artefact/models/word_segmentation2.params"
HANDWRITING_RECOGNITION_MODEL = "/artefact/models/handwriting_line8.params"
DENOISER_MODEL = "/artefact/models/denoiser2.params"

FEATURE_LEN = 150
ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()


@st.cache()
def segment(paragraph_segmented_image):
    # Word segmentation
    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters(WORD_SEGMENTATION_MODEL)
    word_segmentation_net.hybridize()

    min_c = 0.1
    overlap_thres = 0.1
    topk = 600
    predicted_words_bbs = predict_bounding_boxes(
        word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)

    # Line segmentation
    line_bbs = sort_bbs_line_by_line(predicted_words_bbs, y_overlap=0.4)
    line_images = crop_line_images(paragraph_segmented_image, line_bbs)
    return line_images


def get_arg_max(prob):
    """Converts the output of the handwriting recognition network into strings."""
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


@st.cache(suppress_st_warning=True)
def recognize(line_images):
    handwriting_line_recognition_net = HandwritingRecognitionNet(
        rnn_hidden_states=512, rnn_layers=2, ctx=ctx, max_seq_len=160)
    handwriting_line_recognition_net.load_parameters(HANDWRITING_RECOGNITION_MODEL, ctx=ctx)
    handwriting_line_recognition_net.hybridize()

    line_image_size = (60, 800)
    form_character_prob = []
    for i, line_image in enumerate(line_images):
        line_image = handwriting_recognition_transform(line_image, line_image_size)
        line_character_prob = handwriting_line_recognition_net(line_image.as_in_context(ctx))
        form_character_prob.append(line_character_prob)

    progress_bar = st.progress(0)
    decoded_lines_am = []
    for j, line_character_probs in enumerate(form_character_prob):
        decoded_lines_am.append(get_arg_max(line_character_probs))
        progress_bar.progress(int((j + 1) / len(form_character_prob) * 100))
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


@st.cache(suppress_st_warning=True)
def denoise(decoded_lines_am):
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

    progress_bar = st.progress(0)
    decoded_lines = []
    for j, decoded_line_am in enumerate(decoded_lines_am):
        decoded_lines.append(get_denoised(decoded_line_am, denoiser, generator))
        progress_bar.progress(int((j + 1) / len(decoded_lines_am) * 100))
    decoded_text = ' '.join(decoded_lines)
    return decoded_text


st.title("Demo: Handwriting Recognition for English")

uploaded_file = st.file_uploader("Upload an image.")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    image = np.asarray(img.convert('L'))

    st.write("\nSegmentation...")
    line_images = segment(image)

    st.write("\nHandwriting recognition...")
    decoded_lines_am = recognize(line_images)

    st.write("\nDenoising...")
    decoded_text = denoise(decoded_lines_am)

    st.write("Output:")
    st.write(decoded_text)

    reference = st.text_input("Input reference to compute CER and WER")
    st.write("Reference:")
    st.write(reference)

    if reference != "":
        scores = score([reference], [decoded_text])
        st.write(f"CER: {100 * scores[0]:.2f}%")
        st.write(f"WER: {100 * scores[1]:.2f}%")
