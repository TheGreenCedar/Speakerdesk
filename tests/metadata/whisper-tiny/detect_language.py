# Verbatim detect_language excerpt from mlx-audio (MIT), revision
# feb25a37b07923bae556e59111995071d66afa0d. CPU test data; not a runtime adapter.
# Upstream: https://github.com/Blaizzy/mlx-audio/blob/feb25a37b07923bae556e59111995071d66afa0d/mlx_audio/stt/models/whisper/decoding.py
# License notices: packaging/licenses/mlx-audio/LICENSE and
# packaging/licenses/Whisper/Apple-MIT.txt, OpenAI-MIT.txt.
def detect_language(
    model: "Whisper", mel: mx.array, tokenizer=None
) -> Tuple[mx.array, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : mx.array, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = model.get_tokenizer()
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel[None]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = mx.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = mx.full(logits.shape[-1], -mx.inf, dtype=mx.float32)
    mask[list(tokenizer.all_language_tokens)] = 0.0
    logits += mask
    language_tokens = mx.argmax(logits, axis=-1)
    language_token_probs = mx.softmax(logits, axis=-1)
    language_token_probs = np.array(language_token_probs)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs
