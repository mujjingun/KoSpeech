import argparse
import torch
import torchaudio
import sys
sys.path.append('..')
from torch import Tensor
from kospeech.data.audio.core import load_audio
from kospeech.model_builder import load_test_model
from kospeech.utils import label_to_string, id2char, EOS_token

import warnings
warnings.filterwarnings(
    action='ignore',
    category=torch.serialization.SourceChangeWarning,
    module=r'.*'
)

def parse_audio(audio_path: str, del_silence: bool = True) -> Tensor:
    signal = load_audio(audio_path, del_silence)

    feature_vector = torchaudio.compliance.kaldi.fbank(Tensor(signal).unsqueeze(0), num_mel_bins=80,
                                                       frame_length=20, frame_shift=10,
                                                       window_type='hamming').transpose(0, 1).numpy()
    feature_vector -= feature_vector.mean()
    feature_vector = Tensor(feature_vector).transpose(0, 1)

    return feature_vector


parser = argparse.ArgumentParser(description='Run Pretrain')
parser.add_argument('--model_path', type=str, default='../pretrain/model.pt')
parser.add_argument('--audio_path', type=str, default='../pretrain/sample_audio.pcm')
parser.add_argument('--device', type=str, default='cuda')
opt = parser.parse_args()

feature_vector = parse_audio(opt.audio_path, del_silence=True)
input_length = torch.IntTensor([len(feature_vector)])

model = load_test_model(opt, opt.device)
model.eval()

print(feature_vector.shape, input_length.shape)

class Proxy(torch.nn.Module):
    def __init__(self, model):
        super(Proxy, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(inputs=x.unsqueeze(0), input_lengths=torch.IntTensor([feature_vector.shape[0]])
, teacher_forcing_ratio=0.0, return_decode_dict=False)

proxy = Proxy(model)
torch.quantization.fuse_modules(proxy.model.encoder.conv.conv, [['0', '1'], ['3', '4'], ['7','8'], ['10', '11']], inplace=True)

proxy_int8 = torch.quantization.quantize_dynamic(proxy, 
        {torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.LSTM}, 
        dtype=torch.qint8)

print(proxy_int8)

traced = torch.jit.trace(proxy_int8, (feature_vector.to(opt.device)))
traced.save('model.zip')
print(traced.code)

output = proxy_int8(feature_vector)
logit = torch.stack(output, dim=1).to(opt.device)
pred = logit.max(-1)[1]

sentence = label_to_string(pred.cpu().detach().numpy(), id2char, EOS_token)
print(sentence)

