---
language:
- en
- zh
- de
- es
- ru
- ko
- fr
- ja
- pt
- tr
- pl
- ca
- nl
- ar
- sv
- it
- id
- hi
- fi
- vi
- he
- uk
- el
- ms
- cs
- ro
- da
- hu
- ta
- 'no'
- th
- ur
- hr
- bg
- lt
- la
- mi
- ml
- cy
- sk
- te
- fa
- lv
- bn
- sr
- az
- sl
- kn
- et
- mk
- br
- eu
- is
- hy
- ne
- mn
- bs
- kk
- sq
- sw
- gl
- mr
- pa
- si
- km
- sn
- yo
- so
- af
- oc
- ka
- be
- tg
- sd
- gu
- am
- yi
- lo
- uz
- fo
- ht
- ps
- tk
- nn
- mt
- sa
- lb
- my
- bo
- tl
- mg
- as
- tt
- haw
- ln
- ha
- ba
- jw
- su
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
- mlx
- speech-to-text
- speech-to-speech
- speech
- speech generation
- stt
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
pipeline_tag: automatic-speech-recognition
license: apache-2.0
library_name: mlx-audio
model-index:
- name: whisper-tiny
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech (clean)
      type: librispeech_asr
      config: clean
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 7.54
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech (other)
      type: librispeech_asr
      config: other
      split: test
      args:
        language: en
    metrics:
    - type: wer
      value: 17.15
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice 11.0
      type: mozilla-foundation/common_voice_11_0
      config: hi
      split: test
      args:
        language: hi
    metrics:
    - type: wer
      value: 141
      name: Test WER
---

# mlx-community/whisper-tiny-asr-fp16
This model was converted to MLX format from [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) using mlx-audio version **0.2.10**.
Refer to the [original model card](https://huggingface.co/openai/whisper-tiny) for more details on the model.

## Use with mlx-audio

```bash
pip install -U mlx-audio
```

### CLI Example:
```bash
python -m mlx_audio.stt.generate --model mlx-community/whisper-tiny-asr-fp16 --audio "audio.wav"
```
### Python Example:
```python
from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription
model = load_model("mlx-community/whisper-tiny-asr-fp16")
transcription = generate_transcription(
    model=model,
    audio_path="path_to_audio.wav",
    output_path="path_to_output.txt",
    format="txt",
    verbose=True,
)
print(transcription.text)
```
