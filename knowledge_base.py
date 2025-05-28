import sqlite3
from datetime import datetime
from typing import List, Dict


class KnowledgeBase:
    """SQLite-based knowledge base for structured information storage"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                content TEXT,
                source TEXT,
                confidence REAL,
                timestamp TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1 TEXT,
                relationship TEXT,
                entity2 TEXT,
                strength REAL,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_fact(self, category: str, content: str, source: str = "user", confidence: float = 1.0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO facts (category, content, source, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (category, content, source, confidence, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, content, source, confidence, timestamp
            FROM facts
            WHERE content LIKE ?
            ORDER BY confidence DESC
            LIMIT ?
        ''', (f'%{query}%', limit))
        results = []
        for row in cursor.fetchall():
            results.append({
                'category': row[0],
                'content': row[1],
                'source': row[2],
                'confidence': row[3],
                'timestamp': row[4]
            })
        conn.close()
        return results
