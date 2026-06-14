import sqlite3
import os
from datetime import datetime, date
import json

class DatabaseManager:
    def __init__(self, db_path="database/signverse.db"):
        self.db_path = db_path
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Creates the necessary tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Users & Settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    target_language TEXT DEFAULT 'English',
                    voice_gender TEXT DEFAULT 'Female',
                    voice_rate INTEGER DEFAULT 150,
                    text_size TEXT DEFAULT 'Medium',
                    dark_mode INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. Translation History table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS translation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    original_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    confidence REAL,
                    is_emergency INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 3. Session History table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    session_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_gestures INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0
                )
            """)
            
            # 4. Accuracy Statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gesture_stats (
                    gesture_name TEXT PRIMARY KEY,
                    times_recognized INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    last_recognized TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 5. Learning Progress table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_progress (
                    gesture_name TEXT PRIMARY KEY,
                    best_score REAL DEFAULT 0.0,
                    practice_count INTEGER DEFAULT 0,
                    last_practiced TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 6. Unlocked Achievements table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS achievements (
                    achievement_id TEXT PRIMARY KEY,
                    achievement_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    date_unlocked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert a default user if database is empty
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO users (username, target_language, voice_gender, voice_rate, text_size, dark_mode)
                    VALUES ('Default User', 'English', 'Female', 150, 'Medium', 1)
                """)
            
            conn.commit()

    # User Profile Operations
    def get_user_settings(self, username="Default User"):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_user_settings(self, username, language, voice_gender, voice_rate, text_size, dark_mode):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users
                SET target_language = ?, voice_gender = ?, voice_rate = ?, text_size = ?, dark_mode = ?
                WHERE username = ?
            """, (language, voice_gender, voice_rate, text_size, int(dark_mode), username))
            conn.commit()

    # Translation History Operations
    def add_translation(self, original_text, translated_text, target_lang, confidence, is_emergency=0, session_id="default"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO translation_history (session_id, original_text, translated_text, target_lang, confidence, is_emergency)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, original_text, translated_text, target_lang, confidence, int(is_emergency)))
            conn.commit()

    def get_translation_history(self, limit=50):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT original_text, translated_text, target_lang, confidence, is_emergency, timestamp 
                FROM translation_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_emergency_logs(self, limit=20):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT original_text, translated_text, target_lang, confidence, timestamp 
                FROM translation_history 
                WHERE is_emergency = 1
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # Session Analytics
    def start_session(self, session_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO session_history (session_id, start_time, total_gestures, avg_confidence)
                VALUES (?, ?, 0, 0.0)
            """, (session_id, datetime.now().isoformat()))
            conn.commit()

    def update_session(self, session_id, count_inc=1, new_confidence=0.0):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Get existing stats
            cursor.execute("SELECT total_gestures, avg_confidence FROM session_history WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                total, avg = row
                new_total = total + count_inc
                if new_total > 0:
                    new_avg = ((avg * total) + (new_confidence * count_inc)) / new_total
                else:
                    new_avg = 0.0
                
                cursor.execute("""
                    UPDATE session_history 
                    SET total_gestures = ?, avg_confidence = ?, end_time = ?
                    WHERE session_id = ?
                """, (new_total, new_avg, datetime.now().isoformat(), session_id))
            conn.commit()

    # Gesture Analytics
    def log_gesture_occurrence(self, gesture_name, confidence):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT times_recognized, avg_confidence FROM gesture_stats WHERE gesture_name = ?", (gesture_name,))
            row = cursor.fetchone()
            if row:
                count, avg = row
                new_count = count + 1
                new_avg = ((avg * count) + confidence) / new_count
                cursor.execute("""
                    UPDATE gesture_stats
                    SET times_recognized = ?, avg_confidence = ?, last_recognized = CURRENT_TIMESTAMP
                    WHERE gesture_name = ?
                """, (new_count, new_avg, gesture_name))
            else:
                cursor.execute("""
                    INSERT INTO gesture_stats (gesture_name, times_recognized, avg_confidence)
                    VALUES (?, 1, ?)
                """, (gesture_name, confidence))
            conn.commit()

    def get_gesture_analytics(self):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT gesture_name, times_recognized, avg_confidence FROM gesture_stats ORDER BY times_recognized DESC")
            return [dict(row) for row in cursor.fetchall()]

    def clear_all_history(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM translation_history")
            cursor.execute("DELETE FROM session_history")
            cursor.execute("DELETE FROM gesture_stats")
            cursor.execute("DELETE FROM learning_progress")
            cursor.execute("DELETE FROM achievements")
            conn.commit()

    # Learning Progress Operations
    def save_learning_attempt(self, gesture_name, score):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT best_score, practice_count FROM learning_progress WHERE gesture_name = ?", (gesture_name,))
            row = cursor.fetchone()
            if row:
                best, count = row
                new_best = max(best, score)
                cursor.execute("""
                    UPDATE learning_progress
                    SET best_score = ?, practice_count = ?, last_practiced = CURRENT_TIMESTAMP
                    WHERE gesture_name = ?
                """, (new_best, count + 1, gesture_name))
            else:
                cursor.execute("""
                    INSERT INTO learning_progress (gesture_name, best_score, practice_count)
                    VALUES (?, ?, 1)
                """, (gesture_name, score))
            conn.commit()

    def get_learning_progress(self):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learning_progress")
            return {row["gesture_name"]: dict(row) for row in cursor.fetchall()}

    def unlock_achievement(self, ach_id, name, desc):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO achievements (achievement_id, achievement_name, description)
                VALUES (?, ?, ?)
            """, (ach_id, name, desc))
            conn.commit()

    def get_unlocked_achievements(self):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM achievements ORDER BY date_unlocked DESC")
            return [dict(row) for row in cursor.fetchall()]
