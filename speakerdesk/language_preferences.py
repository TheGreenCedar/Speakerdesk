"""One authoritative local language default, stored with active job changes."""
from language_detection import LANGUAGE_CHOICES


def initialize(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY,value TEXT NOT NULL)')


def default_language(conn):
    row=conn.execute("SELECT value FROM preferences WHERE key='default_language'").fetchone()
    return row[0] if row and row[0] in LANGUAGE_CHOICES else 'auto'


def save_default_language(conn,language):
    if not isinstance(language,str) or language not in LANGUAGE_CHOICES:
        raise ValueError('Choose a supported language mode.')
    conn.execute("INSERT OR REPLACE INTO preferences VALUES ('default_language',?)",(language,))
