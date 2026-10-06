"""Sample-accurate language epochs for the live worker's ordered PCM stream."""
import math
from language_detection import LANGUAGE_CHOICES

RATE = 16000


class LanguageEpochs:
    def __init__(self, language):
        if not isinstance(language, str) or language not in LANGUAGE_CHOICES:
            raise ValueError('Unsupported live language.')
        self.history = [{'generation': 0, 'language': language, 'start_sample': 0}]

    def register(self, message, received_samples):
        generation, boundary, language = (message.get(key) for key in ('generation', 'start_sample', 'language'))
        if (type(generation) is not int or generation != self.history[-1]['generation']+1
                or type(boundary) is not int or boundary != received_samples
                or boundary < self.history[-1]['start_sample']
                or not isinstance(language, str) or language not in LANGUAGE_CHOICES):
            raise ValueError('Invalid live language update or audio boundary.')
        epoch = {'generation': generation, 'language': language, 'start_sample': boundary}
        self.history.append(epoch)
        return epoch

    def split(self, start_sample, end_sample):
        if end_sample <= start_sample:
            return
        boundaries = sorted({start_sample, end_sample, *(epoch['start_sample'] for epoch in self.history
                              if start_sample < epoch['start_sample'] < end_sample)})
        for start, end in zip(boundaries, boundaries[1:]):
            index = next(i for i in range(len(self.history)-1, -1, -1)
                         if self.history[i]['start_sample'] <= start)
            epoch = self.history[index]
            closes_epoch = index+1 < len(self.history) and self.history[index+1]['start_sample'] == end
            yield start, end, epoch, closes_epoch


def validate_language_segment(segment, history):
    """Reject mismatched results; never relabel old output with current settings."""
    generation = segment.get('language_generation')
    if type(generation) is not int:
        raise ValueError('Live result is missing its language generation.')
    index = next((i for i, epoch in enumerate(history) if epoch['generation'] == generation), None)
    if index is None:
        raise ValueError('Live result has an unknown language generation.')
    epoch = history[index]
    start, end = float(segment['start']), float(segment['end'])
    if (not all(map(math.isfinite, (start, end))) or not start < end
            or round(start*RATE) < epoch['start_sample']
            or (index+1 < len(history) and round(end*RATE) > history[index+1]['start_sample'])
            or segment.get('language_mode') != epoch['language']
            or (epoch['language'] != 'auto' and segment.get('language') != epoch['language'])):
        raise ValueError('Live result does not match its audio language boundary.')
