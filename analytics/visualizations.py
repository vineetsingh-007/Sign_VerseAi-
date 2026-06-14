import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from database.database_manager import DatabaseManager

class AnalyticsVisualizer:
    """
    Generates rich, high-resolution visual plots (frequency charts,
    confidence logs, usage over time) and saves them in the reports directory.
    """
    def __init__(self, db_path="database/signverse.db"):
        self.db = DatabaseManager(db_path)
        # Apply clean styling
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        # Dark styling config
        self.dark_bg = "#0e1117"
        self.primary_color = "#1f77b4"

    def plot_all_analytics(self, output_dir="reports"):
        """Generates all analytics charts and saves them."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig1 = self.plot_gesture_frequencies(os.path.join(output_dir, "gesture_frequencies.png"))
        fig2 = self.plot_confidence_distribution(os.path.join(output_dir, "confidence_distribution.png"))
        fig3 = self.plot_translation_trends(os.path.join(output_dir, "translation_trends.png"))
        
        return {
            "frequencies": fig1,
            "confidence": fig2,
            "trends": fig3
        }

    def plot_gesture_frequencies(self, save_path):
        """Plots horizontal bar chart of recognized gestures."""
        gestures = self.db.get_gesture_analytics()
        if not gestures:
            return None
            
        df = pd.DataFrame(gestures)
        # Limit to top 10 for neatness
        df = df.head(10)
        
        plt.figure(figsize=(10, 6), facecolor=self.dark_bg)
        ax = plt.subplot(111, facecolor=self.dark_bg)
        
        # Plot bars
        sns.barplot(x="times_recognized", y="gesture_name", data=df, palette="viridis", ax=ax)
        
        # Labels and design
        ax.set_title("Top 10 Most Recognized Sign Gestures", color="white", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Number of Times Recognized", color="white", fontsize=11)
        ax.set_ylabel("Sign Gesture", color="white", fontsize=11)
        ax.tick_params(colors="white")
        
        # Grid settings
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor=self.dark_bg)
        plt.close()
        return save_path

    def plot_confidence_distribution(self, save_path):
        """Plots distribution of prediction confidence scores."""
        history = self.db.get_translation_history(limit=1000)
        if not history:
            return None
            
        confidences = [log["confidence"] for log in history if log["confidence"] is not None]
        if not confidences:
            return None
            
        plt.figure(figsize=(10, 6), facecolor=self.dark_bg)
        ax = plt.subplot(111, facecolor=self.dark_bg)
        
        # Draw distribution curve
        sns.histplot(confidences, bins=15, kde=True, color="#00ffcc", edgecolor="black", alpha=0.7, ax=ax)
        
        ax.set_title("Prediction Confidence Distribution", color="white", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Confidence Score", color="white", fontsize=11)
        ax.set_ylabel("Count", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor=self.dark_bg)
        plt.close()
        return save_path

    def plot_translation_trends(self, save_path):
        """Plots translations count over time (grouped by date/hour)."""
        history = self.db.get_translation_history(limit=5000)
        if not history:
            return None
            
        df = pd.DataFrame(history)
        # Extract date
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['datetime'].dt.date
        
        trend = df.groupby('date').size().reset_index(name='count')
        
        plt.figure(figsize=(10, 6), facecolor=self.dark_bg)
        ax = plt.subplot(111, facecolor=self.dark_bg)
        
        # Line graph
        sns.lineplot(x="date", y="count", data=trend, marker="o", color="#ff007f", linewidth=2.5, ax=ax)
        
        ax.set_title("Translation Activity Over Time", color="white", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Date", color="white", fontsize=11)
        ax.set_ylabel("Total Translations", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
        
        # Rotate dates
        plt.xticks(rotation=30)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor=self.dark_bg)
        plt.close()
        return save_path
