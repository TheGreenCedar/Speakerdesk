# ADR-0002: One-way migration to a new database file

## Status

Accepted

## Context

0.6.x stores jobs in `<storage>/recordings/jobs.sqlite`. That file mixes product fields with recovery journals. A Rust schema that rewrites the same file would destroy the rollback path to 0.6.6.

## Decision

The Rust app opens `<storage>/speakerdesk.sqlite` and migrates into it once. The legacy file is opened read-only, or copied when a hot journal blocks that open. It is never modified. Legacy non-product fields are not copied. If migration fails, only the partial new file is removed and the user is told where the backup is.

## Alternatives

- Migrate in place inside `jobs.sqlite`. Rollback to 0.6.6 would open a database that app cannot read.
- Dual-write both files. The old file would stop being a byte-identical backup.

## Consequences

Rolling back to 0.6.6 opens the pre-migration database. The recordings directory layout and the models directory stay where they are, so existing downloads are reused.
