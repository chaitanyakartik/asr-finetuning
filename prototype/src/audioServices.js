// ============================================
// WAV Audio Recorder Helper Class (STT)
// ============================================
export class WAVRecorder {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.audioInput = null;
    this.recorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.recordingStartTime = null;
  }

  async startRecording() {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000 // Request 16kHz directly if possible
        }
      });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      this.originalSampleRate = this.audioContext.sampleRate;
      console.log('🎙️ Recording started at', this.originalSampleRate, 'Hz');

      this.audioInput = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.recorder = this.audioContext.createScriptProcessor(4096, 1, 1);

      this.audioChunks = [];
      this.isRecording = true;
      this.recordingStartTime = Date.now();

      this.recorder.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        this.audioChunks.push(new Float32Array(channelData));
      };

      this.audioInput.connect(this.recorder);
      const silentNode = this.audioContext.createGain();
      silentNode.gain.value = 0;
      this.recorder.connect(silentNode);
      silentNode.connect(this.audioContext.destination);

      return true;
    } catch (error) {
      console.error('❌ Microphone access error:', error);
      throw error;
    }
  }

  async stopRecording() {
    this.isRecording = false;
    const recordingDuration = (Date.now() - this.recordingStartTime) / 1000;

    // Wait for final chunks
    await new Promise(resolve => setTimeout(resolve, 150));

    // Cleanup
    if (this.recorder) {
      this.recorder.disconnect();
      this.recorder = null;
    }
    if (this.audioInput) {
      this.audioInput.disconnect();
      this.audioInput = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    if (this.audioChunks.length === 0) {
      throw new Error('No audio captured');
    }

    // Calculate actual duration
    const totalSamples = this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const actualDuration = totalSamples / this.originalSampleRate;

    console.log(`📊 Recording: ${recordingDuration.toFixed(2)}s wall-clock, ${actualDuration.toFixed(2)}s actual audio`);

    if (actualDuration < 0.3) {
      throw new Error(`Audio too short: ${actualDuration.toFixed(1)}s`);
    }

    return this.createWAVBlob();
  }

  float32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Array;
  }

  createWAVBlob() {
    const totalLength = this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combinedFloat32 = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioChunks) {
      combinedFloat32.set(chunk, offset);
      offset += chunk.length;
    }

    const targetSampleRate = 16000;
    const resampledData = this.resample(combinedFloat32, this.originalSampleRate, targetSampleRate);
    const int16Data = this.float32ToInt16(resampledData);

    const sampleRate = targetSampleRate;
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = int16Data.length * 2;

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    this.writeString(view, 8, 'WAVE');
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    this.writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    const dataView = new Int16Array(buffer, 44);
    dataView.set(int16Data);

    console.log('✅ WAV created:', (int16Data.length / sampleRate).toFixed(2) + 's,', buffer.byteLength, 'bytes');

    return new Blob([buffer], { type: 'audio/wav' });
  }

  resample(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) return audioData;

    const sampleRateRatio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(audioData.length / sampleRateRatio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < result.length; i++) {
      const srcStart = Math.round(i * sampleRateRatio);
      const srcEnd = Math.round((i + 1) * sampleRateRatio);
      let sum = 0;
      let count = 0;
      for (let j = srcStart; j < srcEnd && j < audioData.length; j++) {
        sum += audioData[j];
        count++;
      }
      result[i] = count > 0 ? sum / count : 0;
    }

    return result;
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}

// ============================================
// TTS Helper Functions
// ============================================

// Helper function to find 'data' chunk in WAV
const findDataChunk = (bytes) => {
  const dataStr = 'data';
  for (let i = 0; i < bytes.length - 3; i++) {
    if (
      bytes[i] === dataStr.charCodeAt(0) &&
      bytes[i + 1] === dataStr.charCodeAt(1) &&
      bytes[i + 2] === dataStr.charCodeAt(2) &&
      bytes[i + 3] === dataStr.charCodeAt(3)
    ) {
      return i;
    }
  }
  return -1;
};

// Helper to set uint32 in little-endian
const setUint32LE = (arr, offset, value) => {
  arr[offset] = value & 0xff;
  arr[offset + 1] = (value >> 8) & 0xff;
  arr[offset + 2] = (value >> 16) & 0xff;
  arr[offset + 3] = (value >> 24) & 0xff;
};

// Helper function to calculate WAV duration
export const calculateWAVDuration = (audioBytes) => {
  try {
    // WAV header structure:
    // Bytes 24-27: Sample Rate (little-endian)
    // Bytes 40-43: Data chunk size (little-endian)
    // Bytes 34-35: Bits per sample (little-endian)
    // Bytes 22-23: Number of channels (little-endian)

    const dataView = new DataView(audioBytes.buffer, audioBytes.byteOffset, audioBytes.byteLength);

    const sampleRate = dataView.getUint32(24, true); // true = little-endian
    const bitsPerSample = dataView.getUint16(34, true);
    const numChannels = dataView.getUint16(22, true);
    const dataSize = dataView.getUint32(40, true);

    const bytesPerSample = bitsPerSample / 8;
    const numSamples = dataSize / (bytesPerSample * numChannels);
    const duration = numSamples / sampleRate;

    console.log(`📊 WAV Info: ${sampleRate}Hz, ${bitsPerSample}bit, ${numChannels}ch, ${dataSize}bytes → ${duration.toFixed(2)}s`);

    return duration;
  } catch (error) {
    console.error('❌ Error calculating duration:', error);
    // Fallback: estimate ~0.5s per 8000 bytes (rough estimate for 16kHz mono)
    return audioBytes.length / 16000;
  }
};

// TTS function with chunking and concurrent processing
export const runTTS = async (text, language, ngrokBaseUrl) => {
  if (!text.trim()) {
    return { audio: null, errors: ['Empty text'] };
  }

  const chunkSize = 150; // Slightly larger chunks for better sentence flow
  const apiUrl = `${ngrokBaseUrl}/tts/tts/tts`;

  console.log('🔊 TTS Endpoint:', apiUrl, 'Language:', language);

  // Split into chunks based on punctuation for better cadence
  const parts = text.replace(/([.?!])\s*(?=[A-Z])/g, "$1|").split("|");
  let chunks = [];
  let current = '';

  for (const part of parts) {
    const p = part.trim();
    if (!p) continue;
    if (current.length + p.length < chunkSize) {
      current += (current ? ' ' : '') + p;
    } else {
      if (current) chunks.push(current);
      current = p;
    }
  }
  if (current) chunks.push(current);

  console.log(`📝 Split text into ${chunks.length} chunks`);

  const results = new Array(chunks.length).fill(null);
  const errors = [];

  // Helper to fetch keys
  const callTTS = async (index, chunk) => {
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: chunk })
      });

      if (!response.ok) {
        return { index, audio: null, error: `${response.status}` };
      }

      const data = await response.json();
      const audioBytes = Uint8Array.from(atob(data.audio_base64), c => c.charCodeAt(0));
      return { index, audio: audioBytes, error: null };
    } catch (err) {
      return { index, audio: null, error: err.message };
    }
  };

  // Execution Strategy: Parallel (Promise.all) for speed since we are not streaming
  const promises = chunks.map((chunk, i) => callTTS(i, chunk));
  const responses = await Promise.all(promises);

  // Process responses
  for (const { index, audio, error } of responses) {
    if (error) {
      errors.push(`Chunk ${index + 1}: ${error}`);
    } else {
      results[index] = audio;
    }
  }

  // Collect successful chunks for stitching
  const audioChunks = results.filter(a => a !== null);

  if (audioChunks.length === 0) {
    return { audio: null, errors: errors.length ? errors : ['No audio returned'] };
  }

  if (audioChunks.length === 1) {
    return { audio: audioChunks[0], errors };
  }

  // Find data chunk in first WAV (Stitching logic)
  const first = audioChunks[0];
  const dataIndex = findDataChunk(first);

  if (dataIndex === -1) {
    // Just concatenate if can't find proper WAV structure
    const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of audioChunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    return { audio: combined, errors };
  }

  const start = dataIndex + 8;
  const header = new Uint8Array(first.slice(0, start));

  // Extract data from all chunks
  const bodyParts = [];
  for (const chunk of audioChunks) {
    const chunkDataIndex = findDataChunk(chunk);
    if (chunkDataIndex !== -1) {
      bodyParts.push(chunk.slice(chunkDataIndex + 8));
    } else {
      bodyParts.push(chunk.slice(44)); // Skip standard WAV header
    }
  }

  const bodyLength = bodyParts.reduce((sum, part) => sum + part.length, 0);
  const body = new Uint8Array(bodyLength);
  let offset = 0;
  for (const part of bodyParts) {
    body.set(part, offset);
    offset += part.length;
  }

  // Update header sizes
  const finalHeader = new Uint8Array(header);
  setUint32LE(finalHeader, start - 4, body.length);
  setUint32LE(finalHeader, 4, 36 + body.length);

  const final = new Uint8Array(finalHeader.length + body.length);
  final.set(finalHeader, 0);
  final.set(body, finalHeader.length);

  console.log('✅ Stitched Final WAV:', final.length, 'bytes');
  return { audio: final, errors };
};

// ============================================
// STT/ASR Helper Function
// ============================================

export const transcribeAudio = async (audioBlob, language, ngrokBaseUrl) => {
  const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
  console.log('📤 Sending to ASR:', audioFile.size, 'bytes', 'Language:', language);

  const formData = new FormData();
  formData.append('file', audioFile);
  formData.append('language', language);

  const apiUrl = `${ngrokBaseUrl}/asr/transcribe`;

  console.log('📡 ASR Endpoint:', apiUrl);

  const response = await fetch(apiUrl, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`ASR failed: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  console.log('📥 Raw ASR Response:', data); // Debugging log
  const transcription = data.text || data.transcription || '';

  console.log('✅ Transcription:', transcription);

  return transcription;
};
