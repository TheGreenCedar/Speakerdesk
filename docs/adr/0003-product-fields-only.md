# ADR-0003: Product fields only

## Status

Accepted

## Context

0.6.6 writes qualification receipts, implementation-source hashes, GPU receipts, and per-packet evidence journals beside the transcript. Those records exist to prove how a build ran. The product UI, exports, edit rules, and recovery do not read them.

## Decision

The Rust schema stores product fields and the private recovery state the live engine needs to resume. Qualification receipts are not produced. Integrity is checked once, when a model is downloaded and when it is loaded. ASR reuse profiles are replaced by an exact-input memo.

## Alternatives

- Port the receipt journals. They add write traffic and have no product reader.
- Recompute hashes on every poll. 0.6.6 already showed that this spends time without changing what the user can do.

## Consequences

A transcript from the new app does not carry the old receipt objects. Behavior the frontend reads, including review reasons, reading turns, protected edits, and speaker assignments, stays.
