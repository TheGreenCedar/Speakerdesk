"""Describe saved-profile compatibility without changing or using its embedding."""
from voice_profiles import VoiceModel


def profile_status(profile, *, model=None, available=False):
    if profile is None:
        return dict(voice_compatible=False,voice_profile_state='none',
            voice_profile_message=None,voice_profile_next_action=None)
    if not available or model is None:
        return dict(voice_compatible=False,voice_profile_state='stored_unavailable',
            voice_profile_message='Your saved voice is kept. Voice recognition is currently unavailable.',
            voice_profile_next_action=None)
    try:
        if type(profile['model']) is not dict or set(profile['model'])!=set(model.payload()):
            raise ValueError('Complete original model identity required')
        saved=VoiceModel(**profile['model'])
    except (KeyError,TypeError,ValueError):
        return dict(voice_compatible=False,voice_profile_state='unverified',
            voice_profile_message='Your saved voice is kept, but its compatibility cannot be verified.',
            voice_profile_next_action=None)
    if profile['model']!=model.payload():
        return dict(voice_compatible=False,voice_profile_state='incompatible_runtime',
            voice_profile_message='Remember this voice again before it can recognize this person. Your saved name stays.',
            voice_profile_next_action='remember_voice')
    return dict(voice_compatible=True,voice_profile_state='compatible',
        voice_profile_message=None,voice_profile_next_action=None)
