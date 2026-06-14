import os

# Multi-lingual translation lookup table for SIGNVERSE AI vocabulary
TRANSLATION_DICTIONARY = {
    "HELLO": {
        "English": "Hello",
        "Spanish": "Hola",
        "French": "Bonjour",
        "German": "Hallo",
        "Hindi": "नमस्ते",
        "Marathi": "नमस्कार",
        "Arabic": "مرحباً",
        "Japanese": "こんにちは"
    },
    "THANK YOU": {
        "English": "Thank you",
        "Spanish": "Gracias",
        "French": "Merci",
        "German": "Danke",
        "Hindi": "धन्यवाद",
        "Marathi": "धन्यवाद",
        "Arabic": "شكرًا لك",
        "Japanese": "ありがとう"
    },
    "PLEASE": {
        "English": "Please",
        "Spanish": "Por favor",
        "French": "S'il vous plaît",
        "German": "Bitte",
        "Hindi": "कृपया",
        "Marathi": "कृपया",
        "Arabic": "من فضلك",
        "Japanese": "お願いします"
    },
    "YES": {
        "English": "Yes",
        "Spanish": "Sí",
        "French": "Oui",
        "German": "Ja",
        "Hindi": "हाँ",
        "Marathi": "हो",
        "Arabic": "نعم",
        "Japanese": "はい"
    },
    "NO": {
        "English": "No",
        "Spanish": "No",
        "French": "Non",
        "German": "Nein",
        "Hindi": "नहीं",
        "Marathi": "नाही",
        "Arabic": "لا",
        "Japanese": "いいえ"
    },
    "GOOD MORNING": {
        "English": "Good morning",
        "Spanish": "Buenos días",
        "French": "Bonjour",
        "German": "Guten Morgen",
        "Hindi": "शुभ प्रभात",
        "Marathi": "शुभ सकाळ",
        "Arabic": "صباح الخير",
        "Japanese": "おはようございます"
    },
    # Emergency vocabulary
    "HELP": {
        "English": "Help me, please",
        "Spanish": "Ayúdeme, por favor",
        "French": "Aidez-moi, s'il vous plaît",
        "German": "Helfen Sie mir bitte",
        "Hindi": "कृपया मेरी मदद करें",
        "Marathi": "कृपया मला मदत करा",
        "Arabic": "ساعدني من فضلك",
        "Japanese": "助けてください"
    },
    "DOCTOR": {
        "English": "I need a doctor",
        "Spanish": "Necesito un médico",
        "French": "J'ai besoin d'un médecin",
        "German": "Ich brauche einen Arzt",
        "Hindi": "मुझे डॉक्टर की जरूरत है",
        "Marathi": "मला डॉक्टरांची गरज आहे",
        "Arabic": "أحتاج إلى طبيب",
        "Japanese": "医者が必要です"
    },
    "EMERGENCY": {
        "English": "This is an emergency!",
        "Spanish": "¡Esto es una emergencia!",
        "French": "C'est une urgence !",
        "German": "Das ist ein Notfall!",
        "Hindi": "यह एक आपातकालीन स्थिति है!",
        "Marathi": "ही आणीबाणीची परिस्थिती आहे!",
        "Arabic": "هذه حالة طوارئ!",
        "Japanese": "緊急事態です！"
    },
    "WATER": {
        "English": "I need water",
        "Spanish": "Necesito agua",
        "French": "J'ai besoin d'eau",
        "German": "Ich brauche Wasser",
        "Hindi": "मुझे पानी चाहिए",
        "Marathi": "मला पाणी हवे आहे",
        "Arabic": "أحتاج ماء",
        "Japanese": "水が欲しいです"
    },
    "PAIN": {
        "English": "I am in pain",
        "Spanish": "Siento dolor",
        "French": "J'ai mal",
        "German": "Ich habe Schmerzen",
        "Hindi": "मुझे दर्द हो रहा है",
        "Marathi": "मला वेदना होत आहेत",
        "Arabic": "أشعر بالألم",
        "Japanese": "痛みます"
    },
    "MEDICINE": {
        "English": "I need medicine",
        "Spanish": "Necesito medicina",
        "French": "J'ai besoin de médicaments",
        "German": "Ich brauche Medizin",
        "Hindi": "मुझे दवा चाहिए",
        "Marathi": "मला औषध हवे आहे",
        "Arabic": "أحتاج دواء",
        "Japanese": "薬が必要です"
    },
    "AMBULANCE": {
        "English": "Call an ambulance!",
        "Spanish": "¡Llamen a una ambulancia!",
        "French": "Appelez une ambulance !",
        "German": "Rufen Sie einen Krankenwagen!",
        "Hindi": "एम्बुलेंस बुलाओ!",
        "Marathi": "रुग्णवाहिका बोलवा!",
        "Arabic": "اتصل بالإسعاف!",
        "Japanese": "救急車を呼んでください！"
    }
}

class LiveTranslator:
    """
    Manages sentence construction from raw gestural tokens (grammar correction)
    and handles translations into multiple target languages.
    """
    def __init__(self):
        pass

    def correct_grammar(self, word_list):
        """
        Takes a sequence of raw gestures (e.g., ["HELLO", "PLEASE"])
        and merges them into a clean English sentence.
        """
        if not word_list:
            return ""
            
        # Clean sequence
        clean_words = []
        for word in word_list:
            if word and word not in ["NO_HAND", "ERROR", "UNKNOWN"]:
                clean_words.append(word)
                
        if not clean_words:
            return ""

        # Check for matching emergency phrases first
        if len(clean_words) == 1:
            token = clean_words[0]
            if token in TRANSLATION_DICTIONARY:
                return TRANSLATION_DICTIONARY[token]["English"]
            return token.capitalize()
            
        # Heuristic rules for common sign combinations
        combo_str = " ".join(clean_words)
        
        # 1. HELLO + PLEASE
        if combo_str == "HELLO PLEASE":
            return "Hello, please come in."
            
        # 2. HELP + DOCTOR
        if "HELP" in clean_words and "DOCTOR" in clean_words:
            return "Help me! I need a doctor immediately."
            
        # 3. HELP + WATER
        if "HELP" in clean_words and "WATER" in clean_words:
            return "Help, please get me some water."
            
        # 4. EMERGENCY + AMBULANCE
        if "EMERGENCY" in clean_words and "AMBULANCE" in clean_words:
            return "This is a critical emergency, call an ambulance!"
            
        # 5. YES + PLEASE
        if combo_str == "YES PLEASE":
            return "Yes, please."
            
        # 6. NO + THANK YOU
        if combo_str == "NO THANK YOU":
            return "No, thank you."

        # Default: join other words and capitalize
        sentence = " ".join([w.lower() for w in clean_words])
        sentence = sentence[0].upper() + sentence[1:] + "."
        return sentence

    def translate(self, english_sentence, target_lang="English"):
        """
        Translates an English sentence into the target language.
        If a direct lookup exists in the translation dictionary, uses it.
        Otherwise, does token-based lookup or returns the raw sentence (with fallback notation).
        """
        if target_lang == "English":
            return english_sentence
            
        # Find exact matches in dictionary
        for sign_token, translations in TRANSLATION_DICTIONARY.items():
            if translations["English"].lower() in english_sentence.lower():
                return translations.get(target_lang, english_sentence)
                
        # Handle word-by-word simple translating fallback if sentence doesn't match templates
        words = english_sentence.rstrip('.!?').split()
        translated_words = []
        
        for w in words:
            upper_w = w.upper()
            if upper_w in TRANSLATION_DICTIONARY:
                translated_words.append(TRANSLATION_DICTIONARY[upper_w].get(target_lang, w))
            else:
                translated_words.append(w)
                
        # Format spacing based on language type
        if target_lang in ["Japanese"]:
            return "".join(translated_words)
        elif target_lang in ["Arabic"]:
            return " ".join(translated_words) # Arabic RTL is handled by layout
        else:
            return " ".join(translated_words)
