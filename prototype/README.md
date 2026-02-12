# Voice Bot - ASR Prototype

A React-based voice bot interface that integrates with a custom local ASR backend, LLM (Gemini), and browser-based TTS.

## Features

- 🎤 **Audio Recording**: Hold-to-record with visual feedback using MediaRecorder API
- 👂 **ASR Integration**: Automatic transcription via local backend at `http://localhost:8001/transcribe`
- 🧠 **LLM Integration**: Google Gemini API for intelligent responses
- 🗣️ **Text-to-Speech**: Browser-native speech synthesis
- 💬 **Chat Interface**: Clean, WhatsApp-style chat UI
- ⚙️ **Settings**: Configurable auto-send and TTS options

## Tech Stack

- React 18
- Vite
- Tailwind CSS
- Lucide React (icons)
- Google Generative AI

## Prerequisites

1. **Node.js** (v18 or higher)
2. **ASR Backend**: Your local ASR server must be running at `http://localhost:8001`
   - Endpoint: `POST /transcribe`
   - Expected FormData key: `file`
   - Expected response: `{ "text": "transcribed text" }` or `{ "transcription": "..." }`

## Installation

1. Navigate to the prototype directory:
   ```bash
   cd /Users/chaitanyakartik/Projects/asr-finetuning/prototype
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

1. **Start your ASR backend** (make sure it's running on port 8001):
   ```bash
   # From your ASR project directory
   python inference/asr_server.py
   ```

2. **Start the Voice Bot**:
   ```bash
   npm run dev
   ```

3. The app will open automatically at `http://localhost:3000`

## Configuration

### Gemini API Key (Optional but Recommended)

1. Click the **Settings** (⚙️) icon in the top-right corner
2. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Paste it in the "Gemini API Key" field
4. The key is stored in your browser's localStorage

**Note**: Without an API key, the bot will echo your messages with a mock delay.

### Settings Options

- **Auto-send after recording**: When enabled, audio is sent immediately after recording stops. When disabled, you can review before sending.
- **Enable Text-to-Speech**: Toggle browser TTS for bot responses

## Usage

### Voice Input
1. **Hold** the microphone button to start recording
2. **Release** to stop recording
3. Audio is automatically transcribed and sent to the LLM
4. Bot response plays via TTS (if enabled)

### Text Input
1. Type in the text field at the bottom
2. Press **Enter** or click the **Send** button
3. Message is sent directly to the LLM

## Project Structure

```
prototype/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── src/
│   ├── main.jsx          # Entry point
│   ├── App.jsx           # Root component
│   ├── VoiceBot.jsx      # Main voice bot component
│   └── index.css         # Tailwind imports
└── README.md
```

## Audio Format

The app records audio in **WebM** format using the MediaRecorder API. Ensure your ASR backend can handle this format. If not, you may need to modify the MIME type in the code:

```javascript
// In VoiceBot.jsx, line ~105
const mediaRecorder = new MediaRecorder(stream, {
  mimeType: 'audio/webm'  // Change to 'audio/wav' if needed
});
```

## Troubleshooting

### CORS Errors
If you see CORS errors in the console:
1. Ensure your ASR backend has CORS enabled
2. In Python Flask, add:
   ```python
   from flask_cors import CORS
   CORS(app)
   ```

### Microphone Access
- The browser will request microphone permission on first use
- For HTTPS/production, use a valid SSL certificate

### ASR Backend Not Responding
- Verify the backend is running: `curl http://localhost:8001`
- Check the backend logs for errors
- Ensure the `/transcribe` endpoint accepts FormData with a `file` key

### Gemini API Errors
- Verify your API key is correct
- Check the browser console for detailed error messages
- Ensure you have API quota remaining

## Future Enhancements

- [ ] Add audio waveform visualization during recording
- [ ] Support for multiple audio formats (WAV, MP3)
- [ ] Voice activity detection (VAD) for automatic start/stop
- [ ] Custom TTS integration (e.g., Coqui, ElevenLabs)
- [ ] Export chat history
- [ ] Multi-language support

## License

MIT
