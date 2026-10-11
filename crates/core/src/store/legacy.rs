//! `core::store::legacy` — one-way migration from the v0.6.6
//! `recordings/jobs.sqlite` into `speakerdesk.sqlite`.
//!
//! The legacy database is never opened directly: the file plus any
//! `-wal`/`-shm`/`-journal` companions are copied to a temp dir and the copy
//! is opened, so SQLite recovery can never touch the original. Writes go to
//! `speakerdesk.sqlite.partial` in one transaction; the file and directory
//! are fsynced, then the file is atomically renamed. On any error only
//! `.partial` is removed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    Result, StoreError, recover_job, write_job_row, write_person_row, write_voice_profile_row,
};
use crate::model::{Job, Person, VoiceProfile, check_opaque_bounds_job};
use crate::transcript::validated_turns;

const LEDGER_TOML: &str = include_str!("../../../../xtask/field-ledger.toml");

/// What the migration produced.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct MigrationOutcome {
    pub jobs: usize,
    pub people: usize,
    pub voice_profiles: usize,
    pub preferences: usize,
    /// Key paths present in the legacy payloads but not in the field ledger —
    /// counted, then dropped.
    pub unknown_keys: BTreeMap<String, u64>,
}

/// The migration failed; the message names the legacy database path.
#[derive(Debug, thiserror::Error)]
#[error("migrating {path}: {message}")]
pub struct MigrateError {
    pub path: PathBuf,
    pub message: String,
    #[source]
    pub source: Option<Box<StoreError>>,
}

impl MigrateError {
    fn wrap(path: &Path, error: StoreError) -> Self {
        Self {
            path: path.to_path_buf(),
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

// ---------------------------------------------------------------------------
// Field-ledger parsing (line-oriented; the ledger file is `[[field]]` blocks
// of `path = "..."` / `class = "..."` lines).

#[derive(Debug, Clone, PartialEq)]
enum Seg {
    Key(String),
    AnyKey,  // {}
    Index,   // []
    Subtree, // terminal *
}

#[derive(Debug, Clone)]
struct LedgerRow {
    segs: Vec<Seg>,
    class: String,
}

fn parse_ledger(toml_text: &str) -> Vec<LedgerRow> {
    let mut rows = Vec::new();
    let mut path: Option<String> = None;
    let mut class: Option<String> = None;
    let mut flush = |path: &mut Option<String>, class: &mut Option<String>| {
        if let (Some(p), Some(c)) = (path.take(), class.take()) {
            let mut segs = Vec::new();
            for seg in p.split('.') {
                match seg {
                    "{}" => segs.push(Seg::AnyKey),
                    "*" => segs.push(Seg::Subtree),
                    s if s.ends_with("[]") => {
                        // `x[]` is key `x` followed by an array element.
                        segs.push(Seg::Key(s[..s.len() - 2].to_string()));
                        segs.push(Seg::Index);
                    }
                    s => segs.push(Seg::Key(s.to_string())),
                }
            }
            rows.push(LedgerRow { segs, class: c });
        }
    };
    for line in toml_text.lines() {
        let line = line.trim();
        if line == "[[field]]" {
            flush(&mut path, &mut class);
        } else if let Some(rest) = line.strip_prefix("path")
            && let Some(v) = rest.trim_start_matches([' ', '=']).strip_prefix('"')
        {
            path = v.strip_suffix('"').map(str::to_string);
        } else if let Some(rest) = line.strip_prefix("class")
            && let Some(v) = rest.trim_start_matches([' ', '=']).strip_prefix('"')
        {
            class = v.strip_suffix('"').map(str::to_string);
        }
    }
    flush(&mut path, &mut class);
    rows
}

/// Observed path segments while walking a payload.
#[derive(Debug, Clone, PartialEq)]
enum Obs {
    Key(String),
    Index,
}

fn seg_matches(pat: &Seg, obs: &Obs) -> bool {
    match (pat, obs) {
        (Seg::Key(p), Obs::Key(o)) => p == o,
        (Seg::AnyKey, Obs::Key(_)) => true,
        (Seg::Index, Obs::Index) => true,
        _ => false,
    }
}

/// The ledger as a coverage set + drop specs.
struct Ledger {
    rows: Vec<LedgerRow>,
    /// Rows that are not a proper prefix of any other row — these cover
    /// their whole subtree (opaque blobs like `segment.provenance`).
    leaf_rows: Vec<LedgerRow>,
    /// (parent pattern, key to delete) derived from drop-class rows.
    drops: Vec<(Vec<Seg>, String)>,
}

impl Ledger {
    fn load() -> Self {
        let rows = parse_ledger(LEDGER_TOML);
        let is_prefix = |a: &[Seg], b: &[Seg]| {
            a.len() < b.len()
                && a.iter().zip(b.iter()).all(|(pa, pb)| match (pa, pb) {
                    (Seg::Key(x), Seg::Key(y)) => x == y,
                    (Seg::AnyKey, Seg::AnyKey) | (Seg::Index, Seg::Index) => true,
                    // a pattern prefix covers a literal key too
                    (Seg::AnyKey, Seg::Key(_)) => true,
                    _ => false,
                })
        };
        let leaf_rows: Vec<LedgerRow> = rows
            .iter()
            .filter(|a| !rows.iter().any(|b| is_prefix(&a.segs, &b.segs)))
            .cloned()
            .collect();
        let mut drops = Vec::new();
        for row in &rows {
            // `drop` rows are removed everywhere. `private` rows under
            // `job.document` are removed too: sealed segments carry product
            // keys only — the engine's private row state (`state`,
            // `machine_versions`, `decode_core_plan`, `last_speech_sample`)
            // lives in `job.rolling_sources`, which is kept. Job-level
            // private keys stay — they persist in the `private` column.
            let under_document =
                row.segs.len() > 1 && matches!(&row.segs[1], Seg::Key(k) if k == "document");
            if row.class != "drop" && !(row.class == "private" && under_document) {
                continue;
            }
            // Ledger paths are scope-prefixed (`job.`); drop patterns apply
            // to the payload root, so remove the leading scope key.
            let mut segs: Vec<Seg> = row.segs.clone();
            if let Some(Seg::Key(_)) = segs.first() {
                segs.remove(0);
            }
            match segs.last() {
                // `path.*`, `path.{}`, `path[]` — remove the parent key whole.
                Some(Seg::Subtree) | Some(Seg::AnyKey) | Some(Seg::Index) => {
                    segs.pop();
                    if let Some(Seg::Key(key)) = segs.pop() {
                        drops.push((segs, key));
                    }
                }
                Some(Seg::Key(key)) => {
                    let key = key.clone();
                    segs.pop();
                    drops.push((segs, key));
                }
                None => {}
            }
        }
        Self {
            rows,
            leaf_rows,
            drops,
        }
    }

    /// Is an observed path covered by the ledger (known)?
    fn covers(&self, path: &[Obs]) -> bool {
        for row in &self.rows {
            if row.segs.len() == path.len()
                && row
                    .segs
                    .iter()
                    .zip(path.iter())
                    .all(|(p, o)| seg_matches(p, o))
            {
                return true;
            }
            // Terminal `*` covers the subtree.
            if let Some(Seg::Subtree) = row.segs.last() {
                let prefix = &row.segs[..row.segs.len() - 1];
                if path.len() >= prefix.len()
                    && prefix
                        .iter()
                        .zip(path.iter())
                        .all(|(p, o)| seg_matches(p, o))
                {
                    return true;
                }
            }
            // A container key is covered when the ledger enumerates rows
            // beneath it (`job.document.diarization` is known because
            // `diarization[].start` etc. are rows).
            if row.segs.len() > path.len()
                && row.segs[..path.len()]
                    .iter()
                    .zip(path.iter())
                    .all(|(p, o)| seg_matches(p, o))
            {
                return true;
            }
        }
        // Leaf (terminal) rows cover their subtree — opaque product/private
        // blobs keep arbitrary contents.
        for row in &self.leaf_rows {
            if path.len() > row.segs.len()
                && row
                    .segs
                    .iter()
                    .zip(path.iter())
                    .all(|(p, o)| seg_matches(p, o))
            {
                return true;
            }
        }
        false
    }

    /// Apply every drop spec to a payload tree.
    fn strip_drops(&self, root: &mut Value) {
        for (parents, key) in &self.drops {
            let mut targets = Vec::new();
            collect_at(root, parents, &mut targets);
            for node in targets {
                if let Value::Object(map) = node {
                    map.remove(key);
                }
            }
        }
    }
}

/// Collect mutable references to nodes at `pattern` below `node`.
fn collect_at<'a>(node: &'a mut Value, pattern: &[Seg], out: &mut Vec<&'a mut Value>) {
    if pattern.is_empty() {
        out.push(node);
        return;
    }
    match (&pattern[0], node) {
        (Seg::Key(k), Value::Object(map)) => {
            if let Some(child) = map.get_mut(k.as_str()) {
                collect_at(child, &pattern[1..], out);
            }
        }
        (Seg::Index, Value::Array(list)) => {
            for child in list.iter_mut() {
                collect_at(child, &pattern[1..], out);
            }
        }
        (Seg::AnyKey, Value::Object(map)) => {
            for child in map.values_mut() {
                collect_at(child, &pattern[1..], out);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------

/// Walk a payload, pushing normalized observed paths.
fn walk(value: &Value, prefix: &mut Vec<Obs>, paths: &mut Vec<Vec<Obs>>) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                prefix.push(Obs::Key(k.clone()));
                paths.push(prefix.clone());
                walk(v, prefix, paths);
                prefix.pop();
            }
        }
        Value::Array(list) => {
            for v in list {
                prefix.push(Obs::Index);
                paths.push(prefix.clone());
                walk(v, prefix, paths);
                prefix.pop();
            }
        }
        _ => {}
    }
}

fn render(path: &[Obs]) -> String {
    let mut out = String::new();
    for seg in path {
        match seg {
            Obs::Key(k) => {
                if !out.is_empty() {
                    out.push('.');
                }
                out.push_str(k);
            }
            Obs::Index => out.push_str("[]"),
        }
    }
    out
}

/// The reading-turns rule: keep `reading_turns` only where the v0.6.6 export
/// gate would have accepted them; otherwise remove the projection so export
/// emits the intact parent passage.
fn apply_reading_turn_gate(document: &mut Value) {
    let Some(segments) = document.get_mut("segments").and_then(Value::as_array_mut) else {
        return;
    };
    for segment in segments.iter_mut() {
        if segment.get("reading_turns").is_some() && validated_turns(segment).is_none() {
            segment.as_object_mut().unwrap().remove("reading_turns");
        }
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

fn fsync_dir(dir: &Path) -> Result<()> {
    fs::File::open(dir)?.sync_all()?;
    Ok(())
}

fn fsync_file(path: &Path) -> Result<()> {
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

/// Run the migration. `Ok(None)` when there is nothing to do (new database
/// already exists, or no legacy database). `now` is the injected clock.
pub fn migrate(
    storage: &Path,
    now: f64,
) -> std::result::Result<Option<MigrationOutcome>, MigrateError> {
    let target = storage.join("speakerdesk.sqlite");
    if target.exists() {
        return Ok(None);
    }
    let legacy = storage.join("recordings").join("jobs.sqlite");
    if !legacy.exists() {
        return Ok(None);
    }
    let partial = storage.join("speakerdesk.sqlite.partial");

    let result = migrate_inner(storage, &legacy, &partial, now).map(Some);
    if result.is_err() {
        // On any error, remove only `.partial` — the legacy database and any
        // pre-existing `speakerdesk.sqlite` are preserved.
        let _ = fs::remove_file(&partial);
    }
    result
}

fn migrate_inner(
    storage: &Path,
    legacy: &Path,
    partial: &Path,
    now: f64,
) -> std::result::Result<MigrationOutcome, MigrateError> {
    let wrap = |e: StoreError| MigrateError::wrap(legacy, e);

    // 1. Copy the legacy database + companions into a temp dir; open the copy.
    let temp = storage.join(format!(
        ".migrate-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir_all(&temp).map_err(|e| wrap(e.into()))?;
    let copy_db = temp.join("jobs.sqlite");
    for suffix in ["", "-wal", "-shm", "-journal"] {
        let src = PathBuf::from(format!("{}{}", legacy.display(), suffix));
        if src.exists() {
            fs::copy(
                &src,
                PathBuf::from(format!("{}{}", copy_db.display(), suffix)),
            )
            .map_err(|e| wrap(e.into()))?;
        }
    }
    let legacy_sha256 = sha256_file(legacy).map_err(wrap)?;

    // 2. Read jobs (with the job_progress overlay), people, voice profiles,
    //    preferences from the copy.
    let source = Connection::open(&copy_db).map_err(|e| wrap(e.into()))?;
    source
        .busy_timeout(std::time::Duration::from_secs(10))
        .map_err(|e| wrap(e.into()))?;

    let mut job_rows: Vec<Value> = Vec::new();
    {
        let mut stmt = source
            .prepare(
                "SELECT j.payload, p.duration, p.updated FROM jobs j
                 LEFT JOIN job_progress p ON p.id=j.id",
            )
            .map_err(|e| wrap(e.into()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<f64>>(1)?,
                    row.get::<_, Option<f64>>(2)?,
                ))
            })
            .map_err(|e| wrap(e.into()))?;
        for row in rows {
            let (payload, progress_duration, progress_updated) = row.map_err(|e| wrap(e.into()))?;
            let mut job: Value = serde_json::from_str(&payload).map_err(|e| wrap(e.into()))?;
            if let Some(duration) = progress_duration {
                let obj = job.as_object_mut().unwrap();
                let base = obj.get("duration").and_then(Value::as_f64).unwrap_or(0.0);
                obj.insert("duration".to_string(), json!(base.max(duration)));
                let updated = obj.get("updated").and_then(Value::as_f64).unwrap_or(0.0);
                obj.insert(
                    "updated".to_string(),
                    json!(updated.max(progress_updated.unwrap_or(0.0))),
                );
            }
            job_rows.push(job);
        }
    }
    let person_rows = read_payloads(&source, "people").map_err(wrap)?;
    let voice_rows = read_payloads(&source, "voice_profiles").map_err(wrap)?;
    let preference_rows = read_pairs(&source, "preferences").map_err(wrap)?;
    drop(source);
    let _ = fs::remove_dir_all(&temp);

    // 3. Map to the product model: unknown-key census, drop strip,
    //    reading-turn gate, recovery rules.
    let ledger = Ledger::load();
    let mut unknown: BTreeMap<String, u64> = BTreeMap::new();
    let census =
        |ledger: &Ledger, root: &str, value: &Value, unknown: &mut BTreeMap<String, u64>| {
            let mut paths = Vec::new();
            walk(value, &mut Vec::new(), &mut paths);
            for path in paths {
                let mut full = vec![Obs::Key(root.to_string())];
                full.extend(path);
                if !ledger.covers(&full) {
                    *unknown.entry(render(&full)).or_insert(0) += 1;
                }
            }
        };

    // Raw `Value` ends here: after the census + ledger strip every payload
    // deserializes strictly into the typed model (`deny_unknown_fields` on
    // persistence-only types surfaces a ledger/type gap as an error).
    let mut jobs: Vec<Job> = Vec::with_capacity(job_rows.len());
    for raw in &mut job_rows {
        census(&ledger, "job", raw, &mut unknown);
        map_payload(raw);
        let mut job: Job = serde_json::from_value(raw.clone()).map_err(|e| wrap(e.into()))?;
        recover_job(&mut job, now).map_err(wrap)?;
        if let Err(msg) = check_opaque_bounds_job(&job) {
            return Err(wrap(StoreError::Store(msg)));
        }
        jobs.push(job);
    }
    let mut people: Vec<Person> = Vec::with_capacity(person_rows.len());
    for raw in &person_rows {
        census(&ledger, "person", raw, &mut unknown);
        people.push(serde_json::from_value(raw.clone()).map_err(|e| wrap(e.into()))?);
    }
    let mut profiles: Vec<(VoiceProfile, Vec<f32>)> = Vec::with_capacity(voice_rows.len());
    for raw in &voice_rows {
        census(&ledger, "voice_profile", raw, &mut unknown);
        let mut raw = raw.clone();
        let centroid: Vec<f32> = raw
            .get("centroid")
            .and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .unwrap_or_default();
        if let Some(obj) = raw.as_object_mut() {
            obj.remove("centroid");
        }
        let profile: VoiceProfile = serde_json::from_value(raw).map_err(|e| wrap(e.into()))?;
        // The model dimension is bounded (model validation 1..=4096) and the
        // centroid must match it exactly.
        if !(1..=4096).contains(&profile.model.dimension)
            || centroid.len() as i64 != profile.model.dimension
        {
            return Err(wrap(StoreError::Store(
                "voice profile centroid does not match the declared dimension".to_string(),
            )));
        }
        profiles.push((profile, centroid));
    }
    for (key, _) in &preference_rows {
        let path = vec![Obs::Key("preference".to_string()), Obs::Key(key.clone())];
        if !ledger.covers(&path) {
            *unknown.entry(render(&path)).or_insert(0) += 1;
        }
    }

    // 4. Write speakerdesk.sqlite.partial in one transaction, fsync, rename.
    {
        let mut conn = Connection::open(partial).map_err(|e| wrap(e.into()))?;
        conn.pragma_update(None, "journal_mode", "DELETE")
            .map_err(|e| wrap(e.into()))?;
        conn.pragma_update(None, "synchronous", "FULL")
            .map_err(|e| wrap(e.into()))?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| wrap(e.into()))?;
        conn.busy_timeout(std::time::Duration::from_secs(10))
            .map_err(|e| wrap(e.into()))?;
        let migrations =
            rusqlite_migration::Migrations::new(vec![rusqlite_migration::M::up(super::SCHEMA)]);
        migrations
            .to_latest(&mut conn)
            .map_err(|e| wrap(e.into()))?;

        let tx = conn.unchecked_transaction().map_err(|e| wrap(e.into()))?;
        for job in &jobs {
            write_job_row(&tx, job).map_err(wrap)?;
        }
        for person in &people {
            write_person_row(&tx, person).map_err(wrap)?;
        }
        for (profile, centroid) in &profiles {
            write_voice_profile_row(&tx, profile, centroid).map_err(wrap)?;
        }
        for (key, value) in &preference_rows {
            tx.execute(
                "INSERT OR REPLACE INTO preferences (key,value) VALUES (?1,?2)",
                params![key, value],
            )
            .map_err(|e| wrap(e.into()))?;
        }
        let counts = json!({
            "jobs": jobs.len(),
            "people": people.len(),
            "voice_profiles": profiles.len(),
            "preferences": preference_rows.len(),
            "unknown_keys": unknown,
        });
        tx.execute(
            "INSERT OR REPLACE INTO migrations
             (id,legacy_path,legacy_sha256,migrated_at,app_version,counts)
             VALUES (1,?1,?2,?3,?4,?5)",
            params![
                legacy.display().to_string(),
                legacy_sha256,
                now,
                env!("CARGO_PKG_VERSION"),
                serde_json::to_string(&counts).unwrap_or_default(),
            ],
        )
        .map_err(|e| wrap(e.into()))?;
        tx.commit().map_err(|e| wrap(e.into()))?;
        conn.close().map_err(|(_, e)| wrap(e.into()))?;
    }
    fsync_file(partial).map_err(wrap)?;
    fsync_dir(storage).map_err(wrap)?;
    fs::rename(partial, target_path(storage)).map_err(|e| wrap(e.into()))?;
    fsync_dir(storage).map_err(wrap)?;

    Ok(MigrationOutcome {
        jobs: jobs.len(),
        people: people.len(),
        voice_profiles: profiles.len(),
        preferences: preference_rows.len(),
        unknown_keys: unknown,
    })
}

fn target_path(storage: &Path) -> PathBuf {
    storage.join("speakerdesk.sqlite")
}

/// Apply the product mapping to one merged legacy job payload without
/// recovery: the reading-turn gate plus drop-class key removal. This is the
/// shape a row has after migration — and the shape the edit-golden harness
/// uses to seed and compare jobs.
pub fn map_payload(payload: &mut Value) {
    if let Some(document) = payload.get_mut("document").filter(|d| d.is_object()) {
        apply_reading_turn_gate(document);
    }
    Ledger::load().strip_drops(payload);
}

fn read_payloads(conn: &Connection, table: &str) -> Result<Vec<Value>> {
    let mut stmt = conn.prepare(&format!("SELECT payload FROM {table}"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    rows.into_iter()
        .map(|payload| serde_json::from_str(&payload).map_err(StoreError::from))
        .collect()
}

fn read_pairs(conn: &Connection, table: &str) -> Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare(&format!("SELECT key,value FROM {table}"))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}
