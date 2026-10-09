"""Lossless display units for CTC evidence, including mixed unspaced scripts.

These are acoustic scalar units with raw punctuation/marks attached for display,
not linguistic word segmentation. No spelling, whitespace or text is rewritten.
"""
import re
import unicodedata

SUPPORTED_LANGUAGES = ('en','de','fr','it','es','pt','el','nl','pl','vi','zh','ar','ja','ko')
CALIBRATED_LANGUAGES = ('en',)
PROVIDER_ID = 'omnilingual-ctc-mlx-supplied-text-v2'
NORMALIZATION_POLICY_ID = 'omni_ctc_attached_ar_marks_ja_separator_v1'
TIMING_UNIT_POLICY_ID = 'mixed_script_ctc_character_whitespace_v1'
LEGACY_UNIT_POLICY_ID = 'whitespace_runs_v1'
FUNCTIONAL_POLICY_ID = 'omni-ctc-mlx-f32-functional-v2-38ff6a225e75'
UPSTREAM_NORMALIZATION_REVISION = '81f51e224ce9e74b02cc2a3eaf21b2d91d743455'


def cjk_character(character):
    value = ord(character)
    return (0x3400 <= value <= 0x9fff or 0x20000 <= value <= 0x3134f
            or 0x3041 <= value <= 0x3096 or 0x30a1 <= value <= 0x30fa
            or 0xff66 <= value <= 0xff9d)


def kana_character(character):
    value = ord(character)
    return (0x3041 <= value <= 0x3096 or 0x30a1 <= value <= 0x30fa
            or 0xff66 <= value <= 0xff9d) and character != 'ー'


def raw_units(text, policy=LEGACY_UNIT_POLICY_ID):
    if policy not in (LEGACY_UNIT_POLICY_ID, TIMING_UNIT_POLICY_ID):
        raise ValueError('Unknown transcript timing unit policy.')
    units = []
    for match in re.finditer(r'\S+', text):
        begin, end = match.span()
        starts = [(begin, 'whitespace_run')]
        if policy == TIMING_UNIT_POLICY_ID and any(cjk_character(c) and c != 'ー' for c in match.group()):
            starts = []
            for index in range(begin, end):
                character = text[index]
                if unicodedata.category(character)[0] in ('P','M') or character == 'ー':
                    continue  # Raw decoration belongs to the preceding scalar.
                kind = 'ctc_character' if cjk_character(character) else 'whitespace_run'
                if kind == 'ctc_character' or not starts or starts[-1][1] != kind:
                    starts.append((index, kind))
            if not starts:
                starts = [(begin, 'whitespace_run')]
            starts[0] = (begin, starts[0][1])  # Keep any leading punctuation.
        for index, (a, kind) in enumerate(starts):
            b = starts[index+1][0] if index+1 < len(starts) else end
            units.append(dict(text=text[a:b], start_char=a, end_char=b,
                start_utf16=len(text[:a].encode('utf-16-le'))//2,
                end_utf16=len(text[:b].encode('utf-16-le'))//2,
                unit_kind=kind, status='unresolved', start_sample=None, end_sample=None))
    return units
