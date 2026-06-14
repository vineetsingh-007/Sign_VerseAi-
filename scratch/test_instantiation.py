import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.write("Step 1: Importing modules...")
from database.database_manager import DatabaseManager
from recognition.inference import SignInferenceEngine
from recognition.prediction_engine import PredictionEngine
from speech.speech_generator import SpeechGenerator
from realtime.live_translation import LiveTranslator
from analytics.visualizations import AnalyticsVisualizer

st.write("Step 2: Instantiating Database...")
db = DatabaseManager()

st.write("Step 3: Instantiating LiveTranslator...")
translator = LiveTranslator()

st.write("Step 4: Instantiating SignInferenceEngine...")
inf_engine = SignInferenceEngine()

st.write("Step 5: Instantiating PredictionEngine...")
pred_engine = PredictionEngine()

st.write("Step 6: Instantiating SpeechGenerator...")
speech_gen = SpeechGenerator(voice_gender="Female", rate=150, lang="English")

st.write("Initialization check: SUCCESS")
