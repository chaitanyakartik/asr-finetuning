import React, { useState, useRef, useEffect } from 'react';
import { Mic, Send, Settings, X, Loader2, StopCircle } from 'lucide-react';
// import { GoogleGenerativeAI } from '@google/generative-ai'; // Removed client-side Gemini
import { WAVRecorder, runTTS, calculateWAVDuration, transcribeAudio } from './audioServices';
import { fixBytecodes } from './geminiUtils';

// Chat Bubble Component
const ChatBubble = ({ message, isUser }) => {
  const audioRef = useRef(null);
  const [audioError, setAudioError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  // Simple auto-play when audio becomes available
  useEffect(() => {
    if (!isUser && message.audioUrl && audioRef.current) {
      console.log('🎵 New audio available, attempting play');

      const tryPlay = async () => {
        try {
          await audioRef.current.load();
          await audioRef.current.play();
          console.log('✅ Auto-play successful');
          setIsPlaying(true);
        } catch (err) {
          console.warn('⚠️ Auto-play blocked:', err.message);
        }
      };

      // Small delay to ensure element is ready
      setTimeout(tryPlay, 100);
    }
  }, [message.audioUrl, isUser]);

  const handlePlayManually = async () => {
    if (audioRef.current) {
      try {
        await audioRef.current.play();
        setIsPlaying(true);
      } catch (err) {
        console.error('❌ Play error:', err);
        setAudioError(true);
      }
    }
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[70%] px-4 py-2 rounded-2xl ${isUser ? 'bg-blue-500 text-white rounded-br-none' : 'bg-gray-200 text-gray-800 rounded-bl-none'
        }`}>
        <p className="text-sm whitespace-pre-wrap break-words">{message.text}</p>

        {/* Audio player for bot messages */}
        {!isUser && message.audioUrl && (
          <div className="mt-2 space-y-1">
            <audio
              ref={audioRef}
              src={message.audioUrl}
              controls
              preload="auto"
              className="w-full rounded"
              style={{ maxWidth: '300px', height: '40px' }}
              onError={(e) => {
                console.error('❌ Audio error:', e.target.error);
                setAudioError(true);
              }}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
            />
            {audioError && (
              <p className="text-xs text-red-600">Audio playback error</p>
            )}
          </div>
        )}

        <span className="text-xs opacity-70 mt-1 block">
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>

        {/* Source Reference */}
        {!isUser && message.source_reference && message.source_reference !== 'N/A' && (
          <div className="mt-2 pt-2 border-t border-gray-300 text-xs text-gray-600 italic">
            Source: {message.source_reference}
          </div>
        )}
      </div>
    </div>
  );
};

// Settings Modal Component
const SettingsModal = ({ isOpen, onClose, settings, onSettingsChange }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Settings</h2>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-full">
            <X size={20} />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Backend URL</label>
            <input
              type="text"
              value={settings.backendUrl}
              onChange={(e) => onSettingsChange({ ...settings, backendUrl: e.target.value })}
              placeholder="http://localhost:8000"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              URL of the Gemini wrapper service
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-700">Enable Text-to-Speech</p>
                <p className="text-xs text-gray-500">Play bot responses using backend TTS</p>
              </div>
              <button
                onClick={() => onSettingsChange({ ...settings, useTTS: !settings.useTTS })}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${settings.useTTS ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${settings.useTTS ? 'translate-x-6' : 'translate-x-1'
                  }`} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main VoiceBot Component
const VoiceBot = () => {
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState(''); // 'transcribing', 'llm', 'tts'
  const [textInput, setTextInput] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    useTTS: true,
    backendUrl: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
  });
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [chatHistory, setChatHistory] = useState([]); // Store backend chat context

  // Audio blob storage - keep references to prevent garbage collection
  const audioBlobsRef = useRef(new Map());

  const wavRecorderRef = useRef(null);
  const chatContainerRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const recordingStartTimeRef = useRef(null);
  const isRecordingRef = useRef(false);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    const savedSettings = localStorage.getItem('voiceBotSettings');
    if (savedSettings) setSettings(JSON.parse(savedSettings));

    const savedHistory = localStorage.getItem('voiceBotHistory');
    if (savedHistory) setMessages(JSON.parse(savedHistory));

    const savedLanguage = localStorage.getItem('voiceBotLanguage');
    if (savedLanguage) setSelectedLanguage(savedLanguage);
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('voiceBotHistory', JSON.stringify(messages));
    }
  }, [messages]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      audioBlobsRef.current.forEach(url => URL.revokeObjectURL(url));
      audioBlobsRef.current.clear();
    };
  }, []);

  const handleSettingsChange = (newSettings) => {
    setSettings(newSettings);
    localStorage.setItem('voiceBotSettings', JSON.stringify(newSettings));
  };

  const exportConversation = () => {
    const data = {
      exportDate: new Date().toISOString(),
      totalMessages: messages.length,
      messages: messages.map(m => ({
        text: m.text,
        isUser: m.isUser,
        timestamp: m.timestamp,
        time: new Date(m.timestamp).toLocaleString()
      }))
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voice-bot-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearConversation = () => {
    if (window.confirm('Clear conversation history?')) {
      setMessages([]);
      setChatHistory([]); // Clear backend history context
      localStorage.removeItem('voiceBotHistory');
      // Clean up audio blobs
      audioBlobsRef.current.forEach(url => URL.revokeObjectURL(url));
      audioBlobsRef.current.clear();
    }
  };

  // Toggle recording on/off with click
  const toggleRecording = async () => {
    if (isRecordingRef.current) {
      // Stop recording
      await stopRecording();
    } else {
      // Start recording
      await startRecording();
    }
  };

  const startRecording = async () => {
    if (isRecordingRef.current) return; // Prevent double-start

    try {
      console.log('🎙️ Starting recording...');
      wavRecorderRef.current = new WAVRecorder();
      await wavRecorderRef.current.startRecording();

      isRecordingRef.current = true;
      setIsRecording(true);
      recordingStartTimeRef.current = Date.now();
      setRecordingTime(0);

      // Use requestAnimationFrame for smooth, accurate timer
      const updateTimer = () => {
        if (isRecordingRef.current) {
          const elapsed = (Date.now() - recordingStartTimeRef.current) / 1000;
          setRecordingTime(elapsed);
          recordingTimerRef.current = requestAnimationFrame(updateTimer);
        }
      };
      recordingTimerRef.current = requestAnimationFrame(updateTimer);

    } catch (error) {
      console.error('❌ Recording start failed:', error);
      isRecordingRef.current = false;
      setIsRecording(false);
      alert('Microphone access denied. Please allow microphone permissions.');
    }
  };

  const stopRecording = async () => {
    if (!isRecordingRef.current) return; // Prevent double-stop

    console.log('🛑 Stopping recording...');

    // Cancel timer immediately
    if (recordingTimerRef.current) {
      cancelAnimationFrame(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }

    const finalDuration = (Date.now() - recordingStartTimeRef.current) / 1000;
    isRecordingRef.current = false;
    setIsRecording(false);

    console.log('⏱️ Final duration:', finalDuration.toFixed(2) + 's');

    // Check minimum duration BEFORE trying to stop recorder
    if (finalDuration < 0.5) {
      console.warn('⚠️ Recording too short');

      // Clean up recorder
      if (wavRecorderRef.current) {
        try {
          wavRecorderRef.current.isRecording = false;
          if (wavRecorderRef.current.mediaStream) {
            wavRecorderRef.current.mediaStream.getTracks().forEach(t => t.stop());
          }
          if (wavRecorderRef.current.audioContext) {
            await wavRecorderRef.current.audioContext.close();
          }
        } catch (e) {
          console.error('Cleanup error:', e);
        }
      }

      setMessages(prev => [...prev, {
        text: `⚠️ Recording too short (${finalDuration.toFixed(1)}s). Please hold for at least 0.5 seconds.`,
        isUser: false,
        timestamp: Date.now()
      }]);
      setRecordingTime(0);
      return;
    }

    // Try to process recording
    try {
      const audioBlob = await wavRecorderRef.current.stopRecording();
      setRecordingTime(0);
      await handleSendAudio(audioBlob);
    } catch (error) {
      console.error('❌ Recording stop error:', error);
      setRecordingTime(0);
      setMessages(prev => [...prev, {
        text: `Error: ${error.message}`,
        isUser: false,
        timestamp: Date.now()
      }]);
    }
  };

  const handleSendAudio = async (audioBlob) => {
    setIsProcessing(true);
    setProcessingStage('transcribing');

    try {
      const ngrokBaseUrl = import.meta.env.VITE_NGROK_BASE_URL || 'http://localhost:8001';
      const transcription = await transcribeAudio(audioBlob, selectedLanguage, ngrokBaseUrl);

      setMessages(prev => [...prev, {
        text: transcription,
        isUser: true,
        timestamp: Date.now()
      }]);

      const cleanedTranscription = fixBytecodes(transcription);
      console.log('🧹 Cleaned Transcription:', cleanedTranscription);

      if (!cleanedTranscription || !cleanedTranscription.trim()) {
        console.warn('⚠️ Empty transcription, skipping LLM call.');
        setMessages(prev => [...prev, {
          text: "Could not hear anything. Please try again.",
          isUser: false,
          timestamp: Date.now(),
          source_reference: 'System'
        }]);
        setProcessingStage('');
        return;
      }

      setProcessingStage('llm');
      await handleLLMResponse(cleanedTranscription);

    } catch (error) {
      console.error('❌ ASR error:', error);
      const ngrokBaseUrl = import.meta.env.VITE_NGROK_BASE_URL || 'http://localhost:8001';
      setMessages(prev => [...prev, {
        text: `Error: ${error.message}. Is the ASR server running on ${ngrokBaseUrl}/asr/transcribe?`,
        isUser: false,
        timestamp: Date.now()
      }]);
    } finally {
      setIsProcessing(false);
      setProcessingStage('');
    }
  };

  const callLLM = async (text) => {
    const maxRetries = 2;
    let lastError = null;
    let lastResponse = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        console.log(`🤖 Calling Backend (Attempt ${attempt + 1}/${maxRetries}) with text:`, text);

        const response = await fetch(`${settings.backendUrl}/infomatics_bot/generate_response`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            chat_history: chatHistory,
            metadata: { language: selectedLanguage }
          })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        console.log('✅ Backend Response:', data);
        lastResponse = data;

        if (data.error) throw new Error(data.error);

        // Update chat history from backend if provided
        if (data.chat_history) {
          setChatHistory(data.chat_history);
        }

        // PARSING LOGIC:
        // Case 1: The backend returns the fields directly
        let finalAnswer = data.answer;
        let finalSource = data.source_reference;

        // Clean up markdown code blocks if present in the answer
        if (typeof finalAnswer === 'string') {
          finalAnswer = finalAnswer.replace(/```json\n?/g, '').replace(/```/g, '').trim();
        }

        // Case 2: 'answer' is a JSON string (double-encoded)
        if (typeof finalAnswer === 'string' && finalAnswer.trim().startsWith('{')) {
          try {
            const parsed = JSON.parse(finalAnswer);
            if (parsed.answer) {
              finalAnswer = parsed.answer;
              finalSource = parsed.source_reference || finalSource;
            }
          } catch (e) {
            console.warn('⚠️ Parse warning: answer looked like JSON but failed to parse', e);
          }
        }

        // Validation: Must have an answer field
        if (finalAnswer) {
          return {
            answer: finalAnswer,
            source_reference: finalSource || 'N/A'
          };
        }

        // If we got here, response was "success" but structure was wrong
        console.warn('⚠️ Invalid response structure (missing answer). Retrying...');

      } catch (error) {
        console.error(`❌ Attempt ${attempt + 1} failed:`, error);
        lastError = error;
      }
    }

    console.error('❌ All retries failed. Returning raw response or error.');

    // Fallback: If we have a malformed response, return it as text
    if (lastResponse) {
      return {
        answer: typeof lastResponse === 'string' ? lastResponse : JSON.stringify(lastResponse, null, 2),
        source_reference: 'Error: Invalid Format'
      };
    }

    return {
      answer: `Backend Error: ${lastError ? lastError.message : 'Unknown error'}`,
      source_reference: 'System Error'
    };
  };

  const handleLLMResponse = async (userText) => {
    try {
      const { answer, source_reference } = await callLLM(userText);

      // Add the text message first WITHOUT audio
      const messageId = Date.now();
      const newMessage = {
        text: answer,
        source_reference,
        isUser: false,
        timestamp: messageId,
        audioUrl: null
      };
      setMessages(prev => [...prev, newMessage]);

      // Generate TTS audio in the background
      if (settings.useTTS) {
        setProcessingStage('tts');

        const ngrokBaseUrl = import.meta.env.VITE_NGROK_BASE_URL || 'http://localhost:8001';

        // Use 'answer' for TTS, not the whole JSON object
        const { audio, errors } = await runTTS(answer, selectedLanguage, ngrokBaseUrl);

        if (errors && errors.length > 0) {
          console.warn('⚠️ TTS warnings:', errors);
        }

        if (audio) {
          // Calculate audio duration from WAV file
          const audioDuration = calculateWAVDuration(audio);
          console.log('🕐 Stitched Audio duration:', audioDuration.toFixed(2), 'seconds');

          // Create audio blob for persistence (Replay)
          const blob = new Blob([audio], { type: 'audio/wav' });
          const audioUrl = URL.createObjectURL(blob);

          // Store reference to prevent garbage collection
          audioBlobsRef.current.set(messageId, audioUrl);

          console.log('🔊 Audio blob created, will attach after 1s');

          // Wait for short delay before attaching (allows user to read first)
          const delayMs = 1000;
          setTimeout(() => {
            console.log('⏰ Delay complete, attaching audio URL now');
            setMessages(prev => prev.map(msg =>
              msg.timestamp === messageId && !msg.isUser
                ? { ...msg, audioUrl } // Attach full stitched audio
                : msg
            ));
          }, delayMs);

        } else {
          console.error('❌ TTS failed:', errors);
          // Fallback to browser TTS if backend fails
          if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(answer);
            window.speechSynthesis.speak(utterance);
          }
        }
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        text: `Error: ${error.message}`,
        isUser: false,
        timestamp: Date.now(),
        audioUrl: null
      }]);
    } finally {
      setProcessingStage('');
    }
  };

  const handleSendText = async () => {
    if (!textInput.trim()) return;

    setMessages(prev => [...prev, {
      text: textInput,
      isUser: true,
      timestamp: Date.now()
    }]);
    setTextInput('');

    setIsProcessing(true);
    const cleanedInput = fixBytecodes(textInput);
    await handleLLMResponse(cleanedInput);
    setIsProcessing(false);
  };

  const getStatusText = () => {
    if (isRecording) return 'Listening...';
    if (processingStage === 'transcribing') return 'Audio being sent for transcription...';
    if (processingStage === 'llm') return 'Processing response with Gemini...';
    if (processingStage === 'tts') return 'Generating speech...';
    return 'Click the mic to start talking';
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm px-6 py-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-800">🎙️ Voice Bot</h1>
        <div className="flex items-center gap-3">
          {/* Language Selector */}
          <select
            value={selectedLanguage}
            onChange={(e) => {
              setSelectedLanguage(e.target.value);
              localStorage.setItem('voiceBotLanguage', e.target.value);
            }}
            className="px-4 py-2 border border-gray-300 rounded-lg bg-white hover:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer font-medium text-gray-700"
          >
            <option value="english">🇬🇧 English</option>
            <option value="hindi">🇮🇳 हिन्दी</option>
            <option value="kannada">🇮🇳 ಕನ್ನಡ</option>
          </select>

          <div className="flex gap-2">
            {messages.length > 0 && (
              <>
                <button
                  onClick={exportConversation}
                  className="px-3 py-1 text-sm bg-green-500 text-white rounded-lg hover:bg-green-600"
                >
                  Export
                </button>
                <button
                  onClick={clearConversation}
                  className="px-3 py-1 text-sm bg-red-500 text-white rounded-lg hover:bg-red-600"
                >
                  Clear
                </button>
              </>
            )}
            <button
              onClick={() => setSettingsOpen(true)}
              className="p-2 hover:bg-gray-100 rounded-full"
            >
              <Settings size={20} />
            </button>
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-4 pb-40">
        {messages.length === 0 && !isRecording && !isProcessing && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-gray-400 text-lg mb-2">Start a conversation</p>
              <p className="text-gray-400 text-sm">Click the mic button to begin</p>
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <ChatBubble key={`${msg.timestamp}-${idx}`} message={msg} isUser={msg.isUser} />
        ))}

        {isProcessing && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-200 px-4 py-2 rounded-2xl rounded-bl-none flex items-center gap-2">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm text-gray-600">Processing...</span>
            </div>
          </div>
        )}
      </div>

      {/* Large Centered Mic Button Area */}
      <div className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pb-8 pt-12 pointer-events-none">
        <div className="flex flex-col items-center gap-4">
          {/* Status Text */}
          <div className="text-center min-h-[24px]">
            <p className={`text-sm font-medium transition-colors ${isRecording ? 'text-red-600' :
              isProcessing ? 'text-blue-600' :
                'text-gray-500'
              }`}>
              {getStatusText()}
            </p>
            {isRecording && (
              <p className="text-xs text-gray-400 mt-1">{recordingTime.toFixed(1)}s</p>
            )}
          </div>

          {/* Large Mic Button with Wave Animation */}
          <div className="relative pointer-events-auto">
            {/* Animated waves when recording */}
            {isRecording && (
              <>
                <div className="absolute inset-0 rounded-full bg-red-400 animate-ping opacity-75" style={{ animationDuration: '1.5s' }}></div>
                <div className="absolute inset-0 rounded-full bg-red-300 animate-ping opacity-50" style={{ animationDuration: '2s' }}></div>
              </>
            )}

            {/* Loading spinner when processing */}
            {isProcessing && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-32 h-32 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin"></div>
              </div>
            )}

            {/* Main Button */}
            <button
              onClick={toggleRecording}
              disabled={isProcessing}
              className={`relative w-24 h-24 rounded-full transition-all duration-300 shadow-2xl ${isRecording
                ? 'bg-red-500 hover:bg-red-600 scale-110'
                : 'bg-blue-500 hover:bg-blue-600 hover:scale-105'
                } text-white disabled:bg-gray-300 disabled:cursor-not-allowed disabled:scale-100`}
              title={isRecording ? "Click to stop" : "Click to start"}
            >
              {isRecording ? (
                <StopCircle size={48} className="mx-auto" />
              ) : (
                <Mic size={48} className="mx-auto" />
              )}
            </button>
          </div>

          {/* Text Input Option (smaller, less prominent) */}
          <div className="flex items-center gap-2 w-full max-w-md px-6 pointer-events-auto">
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendText()}
              placeholder="Or type a message..."
              disabled={isProcessing || isRecording}
              className="flex-1 px-4 py-2 text-sm border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            />
            <button
              onClick={handleSendText}
              disabled={isProcessing || !textInput.trim() || isRecording}
              className="p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onSettingsChange={handleSettingsChange}
      />
    </div>
  );
};

export default VoiceBot;