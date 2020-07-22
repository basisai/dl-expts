import numpy as np
import gluonnlp as nlp
import mxnet as mx
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, EOS, BOS
from ocr.utils.denoiser_utils import SequenceGenerator
from ocr.utils.cer_wer import cer_scores, wer_scores
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import (
    Network as HandwritingRecognitionNet,
    handwriting_recognition_transform,
    decode as decoder_handwriting,
)

DATA_DIR = ""  # "handwriting_ocr_eng/"

WORD_SEGMENTATION_MODEL = "models/word_segmentation2.params"
HANDWRITING_RECOGNITION_MODEL = "models/handwriting_line8.params"
DENOISER_MODEL = "models/denoiser2.params"

FEATURE_LEN = 150
ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()


@st.cache(allow_output_mutation=True)
def load_segmentation_net():
    """Segmentation model."""
    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters(WORD_SEGMENTATION_MODEL)
    word_segmentation_net.hybridize()
    return word_segmentation_net


@st.cache(allow_output_mutation=True)
def load_recognition_net():
    """Recognition model."""
    line_recognition_net = HandwritingRecognitionNet(
        rnn_hidden_states=512, rnn_layers=2, ctx=ctx, max_seq_len=160)
    line_recognition_net.load_parameters(HANDWRITING_RECOGNITION_MODEL, ctx=ctx)
    line_recognition_net.hybridize()
    return line_recognition_net


@st.cache(allow_output_mutation=True)
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


@st.cache(hash_funcs={WordSegmentationNet: id})
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


@st.cache(hash_funcs={HandwritingRecognitionNet: id}, suppress_st_warning=True)
def recognize(line_images):
    """Perform recognition for each line image."""
    line_image_size = (60, 800)
    progress_bar = st.progress(0)
    decoded_lines_am = []
    for j, line_image in enumerate(line_images):
        line_image = handwriting_recognition_transform(line_image, line_image_size)
        line_character_probs = line_recognition_net(line_image.as_in_context(ctx))
        decoded_lines_am.append(get_arg_max(line_character_probs))
        progress_bar.progress(int((j + 1) / len(line_images) * 100))
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


@st.cache(hash_funcs={Denoiser: id, SequenceGenerator: id}, suppress_st_warning=True)
def denoise(decoded_lines_am):
    """Perform denoising for each decoded line."""
    progress_bar = st.progress(0)
    decoded_lines = []
    for j, decoded_line_am in enumerate(decoded_lines_am):
        decoded_lines.append(get_denoised(decoded_line_am, denoiser, generator))
        progress_bar.progress(int((j + 1) / len(decoded_lines_am) * 100))
    return decoded_lines


def compare(reference, decoded_text):
    cer_s, cer_d, cer_i = cer_scores(reference, decoded_text)
    st.write(f"**Overall CER: `{100 * (cer_s + cer_d + cer_i):.2f}%`**")
    st.write(f"- Substitution error = `{100 * cer_s:.2f}%`"
             f"- Deletion error = `{100 * cer_d:.2f}%`"
             f"- Insertion error = `{100 * cer_i:.2f}%`")

    wer_s, wer_d, wer_i = wer_scores(reference, decoded_text)
    st.write(f"**Overall WER: `{100 * (wer_s + wer_d + wer_i):.2f}%`**")
    st.write(f"- Substitution error = `{100 * wer_s:.2f}%`"
             f"- Deletion error = `{100 * wer_d:.2f}%`"
             f"- Insertion error = `{100 * wer_i:.2f}%`")


def eng_ocr():
    st.title("Demo: Handwriting Recognition for English")

    select = st.selectbox("", ["Select a sample image", "Upload an image"])

    if select == "Select a sample image":
        select_ex = st.selectbox("Select a sample image.", [""] + [f"ex{i}" for i in range(1, 5)])
        uploaded_file = None
        if select_ex != "":
            uploaded_file = DATA_DIR + f"samples/{select_ex}.png"
    else:
        uploaded_file = st.file_uploader("Upload an image.")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        image = np.asarray(img.convert("L"))

        st.write("**Segmentation**")
        line_images, line_bbs = segment(image)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image, cmap='Greys_r')
        ax.axis('off')
        image_h, image_w = image.shape[-2:]
        for line_bb in line_bbs:
            (x, y, w, h) = line_bb
            (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)

            rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
            ax.add_patch(rect)
        st.pyplot()

        st.write("**Handwriting recognition**")
        decoded_lines_am = recognize(line_images)
        decoded_text_am = "\n".join(decoded_lines_am)
        st.text(decoded_text_am)

        st.write("**Denoising**")
        decoded_lines = denoise(decoded_lines_am)
        decoded_text = "\n".join(decoded_lines)

        st.subheader("Output")
        st.text(decoded_text)

        st.subheader("Reference")
        reference = st.text_area("Input reference to compute CER and WER.")
        if reference != "":
            st.text(reference)
            compare(reference, decoded_text)


if __name__ == "__main__":
    eng_ocr()


