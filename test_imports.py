import sys
print("Python version:", sys.version)

try:
    import streamlit
    print("Streamlit: SUCCESS (version)", streamlit.__version__)
except ImportError as e:
    print("Streamlit: FAILED -", e)

try:
    import langchain
    print("Langchain: SUCCESS (version)", langchain.__version__)
except ImportError as e:
    print("Langchain: FAILED -", e)

try:
    import chromadb
    print("ChromaDB: SUCCESS (version)", chromadb.__version__)
except ImportError as e:
    print("ChromaDB: FAILED -", e)

try:
    import PyPDF2
    print("PyPDF2: SUCCESS (version)", PyPDF2.__version__)
except ImportError as e:
    print("PyPDF2: FAILED -", e)

try:
    import sentence_transformers
    print("Sentence Transformers: SUCCESS")
except ImportError as e:
    print("Sentence Transformers: FAILED -", e)

try:
    import whisper
    print("Whisper: SUCCESS")
except ImportError as e:
    print("Whisper: FAILED -", e)

try:
    import sounddevice
    print("Sounddevice: SUCCESS")
except ImportError as e:
    print("Sounddevice: FAILED -", e)

try:
    import soundfile
    print("Soundfile: SUCCESS")
except ImportError as e:
    print("Soundfile: FAILED -", e)
