"""CPU arithmetic experiment, never a production ASR admission predicate.

The only supported calibration authority is independently declared controlled
fixture truth. Quiet historical mic correlation is deliberately unsupported.
Every return has production_admissible=False; there is no source/runtime caller.
"""
import hashlib
import numpy as np

RATE = 16000
MAX_LAG = 8000
MAX_ANCHOR = 392000
WINDOW = 2000
MIN_WINDOWS = 3


def pcm_hash(values):
    return hashlib.sha256(values.astype('<i2').tobytes()).hexdigest()


def diagnose(raw, render, request, prior):
    """Check an exact fixed two-channel delayed copy, including full context.

    All sample arithmetic is int64 in PCM16 LSB units. For a common denominator
    D and fixed coefficients nL,nR, 2*|D*y-nL*xL-nR*xR| <= D+|nL|+|nR| is the
    exact worst-case bound for three unclipped, once-rounded inputs. Unknown
    gain error, interpolation, resampling and additional quantization decline.
    Spectral floating arithmetic only adds a refusal; it cannot widen this bound.
    """
    result = {'scope': 'controlled_fixture_fixed_path_explainability_v2',
              'production_admissible': False, 'explained': False}

    def decline(reason, **evidence):
        return dict(result, reason=reason, **evidence)

    raw, render = np.asarray(raw), np.asarray(render)
    if (raw.dtype.kind != 'i' or raw.dtype.itemsize != 2 or raw.ndim != 1
            or render.dtype.kind != 'i' or render.dtype.itemsize != 2
            or render.ndim != 2 or render.shape[1] != 2):
        return decline('missing_raw_or_stereo_pcm16')
    raw, render = raw.copy(), render.copy()
    if len(raw) != len(render):
        return decline('incomplete_shared_clock')
    if (request.get('raw_sha256') != pcm_hash(raw)
            or request.get('render_sha256') != pcm_hash(render)):
        return decline('pcm_identity_changed')
    if (request.get('clock') != 'recording_samples' or request.get('sample_rate') != RATE
            or request.get('channel_map') != ['render_left', 'render_right']
            or request.get('provenance') != 'controlled_fixture_stereo_raw'
            or request.get('gaps') != []):
        return decline('unknown_clock_channels_or_provenance')
    if (prior.get('origin') != 'controlled_fixture_truth'
            or prior.get('near_calibration_truth') != 'absent_by_construction'):
        return decline('no_independent_calibration_truth')
    for value in (request, prior):
        if (any(not isinstance(value.get(key), str) or not value[key].strip()
                for key in ('recording_id', 'route_id'))
                or any(type(value.get(key)) is not int or value[key] <= 0
                       for key in ('raw_revision', 'render_revision'))):
            return decline('unknown_path_binding')
    for key in ('recording_id', 'route_id', 'raw_revision', 'render_revision'):
        if key not in request or request[key] != prior.get(key):
            return decline('path_binding_changed')
    if (prior.get('stages') != 'single_nearest_rounding_pcm16'
            or prior.get('gain_uncertainty') != 0 or prior.get('lag_uncertainty') != 0):
        return decline('unsupported_quantization_or_path_uncertainty')
    anchor, context = request.get('decode_anchor'), request.get('full_context')
    if (not isinstance(anchor, list) or len(anchor) != 2
            or not isinstance(context, list) or len(context) != 2
            or any(type(v) is not int for v in anchor + context)):
        return decline('unknown_full_decode_context')
    start, end = context
    if not (0 <= start <= anchor[0] < anchor[1] <= end <= len(raw)
            and end - start <= MAX_ANCHOR):
        return decline('incomplete_or_unbounded_context')
    observed = prior.get('calibration_observed_end')
    valid = prior.get('valid_samples')
    if (type(observed) is not int or observed > start
            or not isinstance(valid, list) or len(valid) != 2
            or any(type(v) is not int for v in valid)
            or not 0 <= observed <= valid[0] <= start < end <= valid[1]):
        return decline('fresh_or_expired_calibration')
    lag, coefficients, denominator = prior.get('lag_samples'), prior.get('numerators'), prior.get('denominator')
    if (type(lag) is not int or not 0 <= lag <= MAX_LAG
            or prior.get('lag_candidates') != [lag]
            or not isinstance(coefficients, list) or len(coefficients) != 2
            or any(type(v) is not int for v in coefficients)
            or type(denominator) is not int or not 1 <= denominator <= 1000000
            or not 0 < sum(abs(v) for v in coefficients) <= denominator):
        return decline('unsupported_or_ambiguous_fixed_path')
    # Require all physically bounded causal history, including the ambiguity
    # veto's alternatives. Unknown history is never implicitly padded with zero.
    if start < MAX_LAG:
        return decline('incomplete_causal_history')
    observed_raw = raw[start:end].astype(np.int64)
    reference = render[start-MAX_LAG:end].astype(np.int64)
    if np.any((observed_raw == -32768) | (observed_raw == 32767)) or np.any(
            (reference == -32768) | (reference == 32767)):
        return decline('clipped_pcm')
    n_left, n_right = coefficients
    weighted = reference[:, 0]*n_left + reference[:, 1]*n_right
    offset = MAX_LAG-lag
    prediction = weighted[offset:offset+end-start]
    budget = denominator + abs(n_left) + abs(n_right)
    error_twice = 2*np.abs(denominator*observed_raw-prediction)
    maximum = int(error_twice.max())
    result.update(full_context=context, decode_anchor=anchor,
                  raw_sha256=pcm_hash(raw), render_sha256=pcm_hash(render),
                  maximum_error_twice_numerator=maximum,
                  bound_twice_numerator=budget, denominator=denominator)
    failures = np.flatnonzero(error_twice > budget)
    if len(failures):
        return decline('unexplained_sample', first_unexplained_sample=start+int(failures[0]))
    windows = [i for i in range(0, len(prediction)-WINDOW+1, WINDOW)
               if np.max(np.abs(prediction[i:i+WINDOW])) > 4*denominator]
    if not windows:
        return decline('no_observed_render_excitation')
    # A bounded search is solely an ambiguity veto, never a path/gain fit. An
    # alternative must pass the same exact per-sample bound over the full context.
    probe = windows[0]
    probe_raw = denominator*observed_raw[probe:probe+WINDOW]
    for alternative in range(MAX_LAG+1):
        if alternative == lag:
            continue
        other_offset = MAX_LAG-alternative
        other = weighted[other_offset+probe:other_offset+probe+WINDOW]
        if np.any(2*np.abs(probe_raw-other) > budget):
            continue
        full_other = weighted[other_offset:other_offset+end-start]
        if np.all(2*np.abs(denominator*observed_raw-full_other) <= budget):
            return decline('competing_causal_lag', competing_lag_samples=alternative)
    supported = []
    for i in windows:
        values = prediction[i:i+WINDOW].astype(np.float64)
        values -= values.mean()
        spectrum = np.abs(np.fft.rfft(values*np.hanning(WINDOW)))**2
        if spectrum.sum() and spectrum.max() <= .1*spectrum.sum():
            supported.append([start+i, start+i+WINDOW])
    if len(supported) < MIN_WINDOWS:
        return decline('insufficient_broadband_windows')
    return dict(result, explained=True, reason='bounded_fixed_path_explainability',
                support_windows=supported, fixed_lag_samples=lag,
                fixed_numerators=coefficients)
