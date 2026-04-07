from flask import Flask, render_template, request
import os
import whisper
from deep_translator import GoogleTranslator
import language_tool_python

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper model
print("Loading Whisper model (this may take a moment on the first run)...")
model = whisper.load_model("small")

# Safely Initialize Grammar Tools
print("Connecting to Grammar Cloud API...")
try:
    grammar_tool_de = language_tool_python.LanguageToolPublicAPI('de-DE')
    grammar_tool_en = language_tool_python.LanguageToolPublicAPI('en-US')
    grammar_enabled = True
    print("Grammar API connected successfully!")
except Exception as e:
    print(f"⚠️ WARNING: Grammar API is rate-limited or unavailable ({e}).")
    print("The app will still run, but grammar recommendations will be skipped.")
    grammar_enabled = False

def check_and_recommend(text, tool):
    """Checks grammar safely. If the API fails, it returns the original text."""
    if not grammar_enabled or not text or not text.strip():
        return text, []

    try:
        matches = tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        
        recommendations = []
        for match in matches:
            if match.replacements:
                original_word = text[match.offset:match.offset + match.errorLength]
                suggestions = ", ".join(match.replacements[:3])
                recommendations.append(f"Instead of '{original_word}', try: {suggestions}")
                
        return corrected_text, recommendations
    except Exception as e:
        print(f"Grammar check failed during processing: {e}")
        return text, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # 1. Transcribe German audio
    print(f"\nTranscribing {file.filename}...")
    result = model.transcribe(filepath, language="de")
    german_text = result["text"]

    # 2. Translate to English
    print("Translating text to English...")
    english_text = GoogleTranslator(source='de', target='en').translate(german_text)

    # 3. Safely Check Grammar
    print("Checking grammar...")
    # Pass the correct tool variable based on whether it initialized successfully
    tool_de = grammar_tool_de if grammar_enabled else None
    tool_en = grammar_tool_en if grammar_enabled else None
    
    german_corrected, german_recs = check_and_recommend(german_text, tool_de)
    english_corrected, english_recs = check_and_recommend(english_text, tool_en)

    print("Done! Sending results to web page.")
    
    # 4. Pass all variables to the frontend
    return render_template('result.html', 
                           german=german_text, 
                           german_corrected=german_corrected,
                           german_recs=german_recs,
                           english=english_text,
                           english_corrected=english_corrected,
                           english_recs=english_recs)

if __name__ == "__main__":
    app.run(debug=True)