"""Durable job documents with small, independently written capture progress.

The original two-column jobs table remains authoritative. SQL triggers maintain
its public metadata projection in the same transaction, including legacy SQL
writers. Duration checkpoints are overlaid on reads and folded into full writes.
"""
import json
import math
import time


PRIVATE_FIELDS = ('rolling_sources', 'refinement_history', 'fast_history',
                  'fast_previous_revision', 'boundary_candidate', 'boundary_candidates',
                  'rolling_inflight', 'fast_retained_candidate')
SUMMARY_FIELDS = ('id', 'name', 'kind', 'source_file', 'created', 'updated',
                  'status', 'message', 'language', 'duration', 'revision')


class JobStore:
    def __init__(self, db):
        self.db = db

    def initialize(self):
        paths = ','.join(f"'$.{key}'" for key in ('document',) + PRIVATE_FIELDS)
        projection = f'json_remove(NEW.payload,{paths})'
        with self.db() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS job_metadata '
                         '(id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT, created REAL)')
            conn.execute('CREATE INDEX IF NOT EXISTS job_metadata_status ON job_metadata(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS job_metadata_created ON job_metadata(created)')
            conn.execute('CREATE TABLE IF NOT EXISTS job_progress '
                         '(id TEXT PRIMARY KEY, duration REAL NOT NULL, updated REAL NOT NULL)')
            for event in ('INSERT', 'UPDATE'):
                conn.execute(f'''CREATE TRIGGER IF NOT EXISTS jobs_metadata_{event.lower()}
                    AFTER {event} ON jobs BEGIN
                    INSERT OR REPLACE INTO job_metadata VALUES
                    (NEW.id,{projection},json_extract(NEW.payload,'$.status'),json_extract(NEW.payload,'$.created'));
                    DELETE FROM job_progress WHERE id=NEW.id;
                    END''')
            conn.execute('''CREATE TRIGGER IF NOT EXISTS jobs_metadata_delete AFTER DELETE ON jobs BEGIN
                DELETE FROM job_metadata WHERE id=OLD.id;
                DELETE FROM job_progress WHERE id=OLD.id;
                END''')
            conn.execute(f'''INSERT OR IGNORE INTO job_metadata
                SELECT id,{projection.replace('NEW.payload', 'payload')},
                json_extract(payload,'$.status'),json_extract(payload,'$.created') FROM jobs''')

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        job = json.loads(row['payload'])
        if row['progress_duration'] is not None:
            job['duration'] = max(job.get('duration') or 0, row['progress_duration'])
            job['updated'] = max(job.get('updated') or 0, row['progress_updated'])
        return job

    def get(self, jid, *, metadata=False):
        table = 'job_metadata' if metadata else 'jobs'
        with self.db() as conn:
            row = conn.execute(f'''SELECT j.payload,p.duration AS progress_duration,
                p.updated AS progress_updated FROM {table} j
                LEFT JOIN job_progress p ON p.id=j.id WHERE j.id=?''', (jid,)).fetchone()
        return self._decode(row)

    def summaries(self):
        with self.db() as conn:
            rows = conn.execute('''SELECT j.payload,p.duration AS progress_duration,
                p.updated AS progress_updated FROM job_metadata j
                LEFT JOIN job_progress p ON p.id=j.id ORDER BY j.created DESC''').fetchall()
        return [{key: job[key] for key in SUMMARY_FIELDS if key in job}
                for job in map(self._decode, rows)]

    def count_status(self, statuses):
        placeholders = ','.join('?' for _ in statuses)
        with self.db() as conn:
            return conn.execute(f'SELECT count(*) FROM job_metadata WHERE status IN ({placeholders})',
                                statuses).fetchone()[0]

    def put(self, job, *, default_language=None):
        with self.db() as conn:
            conn.execute('BEGIN IMMEDIATE')
            progress = conn.execute('SELECT duration FROM job_progress WHERE id=?', (job['id'],)).fetchone()
            if progress:
                job['duration'] = max(job.get('duration') or 0, progress['duration'])
            job['updated'] = time.time()
            conn.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)',
                         (job['id'], json.dumps(job, ensure_ascii=False)))
            # The selected future-audio epoch and restart default are one commit.
            # Projection/checkpoint triggers are rolled back too if this fails.
            if default_language is not None:
                from language_preferences import save_default_language
                save_default_language(conn, default_language)

    def checkpoint_duration(self, jid, duration):
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Capture duration must be finite and nonnegative.')
        with self.db() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('''SELECT json_extract(j.payload,'$.duration') AS duration,
                p.duration AS checkpoint FROM job_metadata j
                LEFT JOIN job_progress p ON p.id=j.id WHERE j.id=?''', (jid,)).fetchone()
            if row is None:
                return False
            if duration > max(row['duration'] or 0, row['checkpoint'] or 0):
                conn.execute('INSERT OR REPLACE INTO job_progress VALUES (?,?,?)',
                             (jid, duration, time.time()))
        return True

    def recover(self, active):
        """Fold crash checkpoints before the existing interrupted-work recovery."""
        with self.db() as conn:
            conn.execute('BEGIN IMMEDIATE')
            rows = conn.execute('''SELECT j.payload,p.duration AS progress_duration,
                p.updated AS progress_updated FROM jobs j
                LEFT JOIN job_progress p ON p.id=j.id''').fetchall()
            for row in rows:
                job = self._decode(row)
                changed = row['progress_duration'] is not None
                if job['status'] in active:
                    job.update(status='failed', message=(
                        'The meeting was interrupted. Captured audio and transcript were preserved.'
                        if job.get('kind') == 'meeting' else
                        'The app stopped during processing. Retry this job.'), updated=time.time())
                    changed = True
                if job.get('rolling_refinement') and job.get('refinement_status') in ('waiting', 'refining'):
                    from rolling_refinement import RollingPlan
                    job['rolling_refinement'] = RollingPlan(job['rolling_refinement'], recover=True).snapshot()
                    job['refinement_status'] = 'paused'
                    job.pop('rolling_inflight', None)
                    changed = True
                if changed:
                    conn.execute('UPDATE jobs SET payload=? WHERE id=?',
                                 (json.dumps(job, ensure_ascii=False), job['id']))
