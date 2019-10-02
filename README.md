# dl-expts
Contains codes on
- deep convolutional generative adversarial networks (DCGAN)
- named entity recognition with BERT (BERT-NER)

### DCGAN
- Uses TensorFlow & Keras
- Config: `dcgan.hcl`
- Requirements: `requirements_tensorflow.txt`
- Training code: `train_dcgan.py`

### BERT-NER
- Uses PyTorch
- Config: `bert.hcl`
- Requirements: `requirements_pytorch.txt`
- Training code: `train_bertner.py`
- Serving code: `serve_bertner.py`

## Notes
- There is no official PyTorch base image, so use TensorFlow-GPU base image and install PyTorch.
