import os
from datetime import datetime
from database.database_manager import DatabaseManager

class AnalyticsReporter:
    """
    Parses translation history logs in SQLite and generates
    structured summaries and text reports for user communication analytics.
    """
    def __init__(self, db_path="database/signverse.db"):
        self.db = DatabaseManager(db_path)

    def generate_summary(self):
        """
        Compiles numerical metrics from database records.
        """
        history = self.db.get_translation_history(limit=5000)
        gestures = self.db.get_gesture_analytics()
        
        total_translations = len(history)
        emergency_count = sum(1 for log in history if log.get("is_emergency", 0) == 1)
        
        # Calculate average confidence
        confidences = [log.get("confidence", 0.0) for log in history if log.get("confidence") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Languages used
        langs = {}
        for log in history:
            lang = log.get("target_lang", "English")
            langs[lang] = langs.get(lang, 0) + 1
            
        # Top gestures
        top_gestures = []
        for g in gestures[:5]:
            top_gestures.append(f"{g['gesture_name']} (x{g['times_recognized']}, {g['avg_confidence']*100:.1f}% acc)")

        summary = {
            "total_translations": total_translations,
            "emergency_count": emergency_count,
            "avg_confidence": avg_confidence,
            "languages_distribution": langs,
            "top_gestures": top_gestures
        }
        return summary

    def write_report_to_file(self, output_path="reports/usage_report.txt"):
        """
        Writes a professional formatted report to a file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stats = self.generate_summary()
        
        report_lines = [
            "="*50,
            "               SIGNVERSE AI REPORT",
            "             Communication & Usage Summary",
            "="*50,
            f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "-"*50,
            f"Total Signed Translations: {stats['total_translations']}",
            f"Average Sign Confidence:  {stats['avg_confidence']*100:.2f}%",
            f"Emergency SOS Triggers:   {stats['emergency_count']}",
            "-"*50,
            "LANGUAGE DISTRIBUTION:",
        ]
        
        for lang, count in stats["languages_distribution"].items():
            report_lines.append(f"  - {lang}: {count} translation(s)")
            
        report_lines.append("-"*50)
        report_lines.append("TOP 5 FREQUENT GESTURES:")
        if stats["top_gestures"]:
            for idx, g in enumerate(stats["top_gestures"], 1):
                report_lines.append(f"  {idx}. {g}")
        else:
            report_lines.append("  No gestures logged yet.")
            
        report_lines.append("="*50)
        
        content = "\n".join(report_lines)
        with open(output_path, "w") as f:
            f.write(content)
            
        print(f"Usage report generated at: {output_path}")
        return output_path
