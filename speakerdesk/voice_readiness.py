"""Public voice-runtime readiness; descriptive only, never an activation gate."""


def readiness(*, available, supported=True, released=False, installed=False):
    if available:
        return dict(readiness_code='runtime_ready', next_action=None,
                    message='Saved voices are recognized once when a new speaker appears. You can correct any name.')
    if not supported:
        return dict(readiness_code='runtime_unsupported', next_action='updates',
                    message='Voice recognition requires the current Speakerdesk on an Apple Silicon Mac with macOS 15 or later. Saved names and voices are kept.')
    if not released:
        return dict(readiness_code='runtime_unqualified', next_action='updates',
                    message='Voice recognition needs a qualified GPU/ANE runtime. Saved names and voices are kept. Install a Speakerdesk update with qualified voice recognition.')
    if not installed:
        return dict(readiness_code='model_missing', next_action='models',
                    message='Finish the voice model download in Settings, then refresh passages. No voice is saved until you give consent.')
    return dict(readiness_code='runtime_unavailable', next_action='updates',
                message='The installed voice runtime could not start. Saved names and voices are kept. Update Speakerdesk or retry voice setup.')
