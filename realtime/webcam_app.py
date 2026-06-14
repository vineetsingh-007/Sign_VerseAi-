import cv2
import time
import os

# Import components
from database.database_manager import DatabaseManager
from vision.hand_detector import HandDetector
from vision.landmark_extractor import LandmarkExtractor
from recognition.inference import SignInferenceEngine
from recognition.prediction_engine import PredictionEngine
from speech.speech_generator import SpeechGenerator
from realtime.live_translation import LiveTranslator

def run_webcam_application():
    """
    Launches the standalone real-time Sign Language desktop translation app.
    Renders live subtitles, confidence overlays, and vocalizes complete sentences.
    """
    print("\n" + "="*50)
    print("        LAUNCHING SIGNVERSE AI WEBCAM CLIENT")
    print("="*50)
    print("Keyboard Shortcuts:")
    print("  [S] : Speak/Vocalize Current Sentence")
    print("  [C] : Clear Assembled Sentence")
    print("  [Q] : Quit Application")
    print("="*50)

    # Initialize managers
    db = DatabaseManager()
    detector = HandDetector()
    extractor = LandmarkExtractor()
    inf_engine = SignInferenceEngine()
    pred_engine = PredictionEngine()
    translator = LiveTranslator()
    
    # Get user settings
    user_settings = db.get_user_settings()
    target_lang = user_settings.get("target_language", "English")
    voice_gender = user_settings.get("voice_gender", "Female")
    voice_rate = user_settings.get("voice_rate", 150)
    
    # Session start
    session_id = f"desktop_session_{int(time.time())}"
    db.start_session(session_id)
    
    # Speech queue generator
    speech_gen = SpeechGenerator(voice_gender=voice_gender, rate=voice_rate, lang=target_lang)
    
    # Assembled sentence buffer
    active_sentence_words = []
    
    # Start capturing
    # Try different camera indices to find an active camera
    cap = None
    for camera_idx in [0, 1, 2, -1]:
        print(f"Attempting to open camera at index {camera_idx}...")
        try:
            tmp_cap = cv2.VideoCapture(camera_idx)
            if tmp_cap.isOpened():
                ret, test_frame = tmp_cap.read()
                if ret and test_frame is not None:
                    cap = tmp_cap
                    print(f"Successfully connected and read test frame from camera at index {camera_idx}!")
                    break
            tmp_cap.release()
        except Exception as e:
            print(f"Error opening camera index {camera_idx}: {e}")
        cap = None

    if cap is None:
        print("\n" + "!"*70)
        print("ERROR: COULD NOT ACCESS ANY WEBCAM CAMERA FEED.")
        print("\nThis is typically caused by one of two issues on macOS:")
        print("1. CAMERA PRIVACY PERMISSIONS:")
        print("   Go to: System Settings -> Privacy & Security -> Camera")
        print("   Ensure your Terminal (or VS Code / iTerm) is enabled/checked.")
        print("2. CAMERA INDEX CONFLICT:")
        print("   Another app (Zoom, Teams, Chrome) might be locking the webcam.")
        print("!"*70 + "\n")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    
    # Frame stats
    fps = 0.0
    last_word = "None"
    last_conf = 0.0
    
    # Alert state for red warning border
    is_alert = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
            
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Hand tracking and overlay drawing
        results, annotated_frame = detector.find_hands(frame, draw=True)
        
        # Landmark feature extraction
        features = extractor.extract_landmarks(results)
        
        # Inference
        hand_detected = results.multi_hand_landmarks is not None
        raw_pred, raw_conf = inf_engine.predict(features)
        
        # Temporal smoothing
        stable_word, avg_conf, is_new, is_emergency = pred_engine.process_frame(
            raw_pred, raw_conf, hand_detected
        )
        
        # Console diagnostic logging (every 15 frames to prevent spamming)
        frame_count += 1
        if frame_count % 15 == 0:
            if hand_detected:
                print(f"[AI Tracking] Hand Detected! Raw Pred: {raw_pred} ({raw_conf*100:.1f}%) | Stable Lock: {pred_engine.last_stable_prediction}")
            else:
                print("[AI Tracking] Scanning... No Hand Detected in camera frame.")
        
        if stable_word and is_new:
            last_word = stable_word
            last_conf = avg_conf
            active_sentence_words.append(stable_word)
            
            # Log occurrence
            db.log_gesture_occurrence(stable_word, avg_conf)
            db.update_session(session_id, count_inc=1, new_confidence=avg_conf)
            
            if is_emergency:
                is_alert = True
                speech_gen.speak(f"Alert! patient needs {stable_word}!", force=True)
                db.add_translation(stable_word, f"SOS: {stable_word}!", target_lang, avg_conf, is_emergency=1, session_id=session_id)
            else:
                is_alert = False
        
        # Build sentences
        english_sentence = translator.correct_grammar(active_sentence_words)
        translated_sentence = translator.translate(english_sentence, target_lang)
        
        # --- Drawing Desktop HUD ---
        # Draw background bar for subtitles
        hud_height = 120
        cv2.rectangle(annotated_frame, (0, h - hud_height), (w, h), (18, 17, 14), cv2.FILLED)
        
        # Glowing border around screen if Emergency/Alert is active
        if is_alert:
            cv2.rectangle(annotated_frame, (0, 0), (w, h), (0, 0, 255), 15)
            cv2.putText(annotated_frame, "EMERGENCY SOS ACTIVE", (w // 2 - 200, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                        
        # Render Text overlay (HUD)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Active Sign: {last_word} ({last_conf*100:.1f}%)", (w - 350, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
                    
        # Subtitles print
        sub_y = h - 75
        cv2.putText(annotated_frame, f"Sign: {english_sentence}", (30, sub_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Trans ({target_lang}): {translated_sentence}", (30, sub_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                    
        # Render window
        cv2.imshow("SignVerse AI - Desktop Assistive Client", annotated_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            active_sentence_words = []
            last_word = "None"
            last_conf = 0.0
            is_alert = False
            print("Cleared current sentence.")
        elif key == ord('s') or key == ord('S'):
            if translated_sentence:
                print(f"Vocalizing sentence: '{translated_sentence}'")
                speech_gen.speak(translated_sentence, force=True)
                # Save to database log on vocalization
                db.add_translation(english_sentence, translated_sentence, target_lang, last_conf, is_emergency=int(is_alert), session_id=session_id)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    speech_gen.stop()
    print("Standalone application stopped cleanly.")

if __name__ == "__main__":
    run_webcam_application()
