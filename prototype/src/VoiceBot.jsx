import React, { useState, useRef, useEffect } from 'react';
import { Mic, Send, Settings, X, Loader2 } from 'lucide-react';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Chat Bubble Component
const ChatBubble = ({ message, isUser }) => {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[70%] rounded-2xl px-4 py-2 ${
          isUser
            ? 'bg-blue-500 text-white rounded-br-none'
            : 'bg-gray-200 text-gray-800 rounded-bl-none'
        }`}
      >
        <p className="text-sm whitespace-pre-wrap">{message.text}</p>
        <span className="text-xs opacity-70 mt-1 block">
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </span>
      </div>
    </div>
  );
};

// Settings Modal Component
const SettingsModal = ({ isOpen, onClose, settings, onSettingsChange }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-full"
          >
            <X size={20} />
          </button>
        </div>

        <div className="space-y-4">
          {/* Gemini API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Gemini API Key
            </label>
            <input
              type="password"
              value={settings.geminiApiKey}
              onChange={(e) =>
                onSettingsChange({ ...settings, geminiApiKey: e.target.value })
              }
              placeholder="Enter your Gemini API key"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Get your API key from{' '}
              <a
                href="https://makersuite.google.com/app/apikey"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:underline"
              >
                Google AI Studio
              </a>
            </p>
          </div>

          {/* Auto-send toggle */}
          <div className="flex items-center justify-between">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Auto-send after recording
              </label>
              <p className="text-xs text-gray-500">
                Automatically send audio after recording stops
              </p>
            </div>
            <button
              onClick={() =>
                onSettingsChange({ ...settings, autoSend: !settings.autoSend })
              }
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.autoSend ? 'bg-blue-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.autoSend ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          {/* Use native TTS */}
          <div className="flex items-center justify-between">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Enable Text-to-Speech
              </label>
              <p className="text-xs text-gray-500">
                Play bot responses using browser TTS
              </p>
            </div>
            <button
              onClick={() =>
                onSettingsChange({ ...settings, useTTS: !settings.useTTS })
              }
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.useTTS ? 'bg-blue-500' : 'bg-gray-300'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.useTTS ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
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
  const [textInput, setTextInput] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    geminiApiKey: '',
    autoSend: true,
    useTTS: true
  });
  const [recordedAudio, setRecordedAudio] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const chatContainerRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Load settings from localStorage
  useEffect(() => {
    const savedSettings = localStorage.getItem('voiceBotSettings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  // Save settings to localStorage
  const handleSettingsChange = (newSettings) => {
    setSettings(newSettings);
    localStorage.setItem('voiceBotSettings', JSON.stringify(newSettings));
  };

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });

      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setRecordedAudio(audioBlob);
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());

        // Auto-send if enabled
        if (settings.autoSend) {
          handleSendAudio(audioBlob);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Unable to access microphone. Please grant permission.');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Send audio to ASR backend
  const handleSendAudio = async (audioBlob = recordedAudio) => {
    if (!audioBlob) return;

    setIsProcessing(true);
    setRecordedAudio(null);

    try {
      // Convert blob to file
      const audioFile = new File([audioBlob], 'recording.webm', {
        type: 'audio/webm'
      });

      // Send to ASR backend
      const formData = new FormData();
      formData.append('file', audioFile);

      const asrResponse = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData
      });

      if (!asrResponse.ok) {
        throw new Error(`ASR request failed: ${asrResponse.statusText}`);
      }

      const asrData = await asrResponse.json();
      const transcribedText = asrData.text || asrData.transcription || '';

      // Add user message
      const userMessage = {
        text: transcribedText,
        isUser: true,
        timestamp: Date.now()
      };
      setMessages((prev) => [...prev, userMessage]);

      // Get LLM response
      await handleLLMResponse(transcribedText);
    } catch (error) {
      console.error('Error processing audio:', error);
      const errorMessage = {
        text: `Error: ${error.message}. Make sure ASR server is running at http://localhost:8000`,
        isUser: false,
        timestamp: Date.now()
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Call LLM (Gemini API)
  const callLLM = async (text) => {
    if (!settings.geminiApiKey) {
      // Mock response if no API key
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve(`Echo: ${text} (Configure Gemini API key in settings for real responses)`);
        }, 1000);
      });
    }

    try {
      const genAI = new GoogleGenerativeAI(settings.geminiApiKey);
      const model = genAI.getGenerativeModel({ model: 'gemini-pro' });

      const result = await model.generateContent(text);
      const response = await result.response;
      return response.text();
    } catch (error) {
      console.error('Error calling Gemini API:', error);
      throw new Error('Failed to get LLM response. Check your API key.');
    }
  };

  // Handle LLM response
  const handleLLMResponse = async (userText) => {
    try {
      const llmResponse = await callLLM(userText);

      const botMessage = {
        text: llmResponse,
        isUser: false,
        timestamp: Date.now()
      };
      setMessages((prev) => [...prev, botMessage]);

      // Play TTS if enabled
      if (settings.useTTS) {
        playTTS(llmResponse);
      }
    } catch (error) {
      console.error('Error getting LLM response:', error);
      const errorMessage = {
        text: `Error: ${error.message}`,
        isUser: false,
        timestamp: Date.now()
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  // Text-to-Speech using browser API
  const playTTS = (text) => {
    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      window.speechSynthesis.speak(utterance);
    } else {
      console.warn('Text-to-speech not supported in this browser');
    }
  };

  // Send text message
  const handleSendText = async () => {
    if (!textInput.trim()) return;

    const userMessage = {
      text: textInput,
      isUser: true,
      timestamp: Date.now()
    };
    setMessages((prev) => [...prev, userMessage]);
    setTextInput('');
    setIsProcessing(true);

    await handleLLMResponse(textInput);
    setIsProcessing(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 flex justify-between items-center">
        <h1 className="text-xl font-bold text-gray-800">Voice Bot</h1>
        <button
          onClick={() => setSettingsOpen(true)}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
        >
          <Settings size={20} className="text-gray-600" />
        </button>
      </div>

      {/* Chat Container */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto px-4 py-4 space-y-2"
      >
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <Mic size={48} className="mx-auto mb-4 opacity-50" />
              <p>Start a conversation by recording or typing</p>
            </div>
          </div>
        )}
        {messages.map((msg, index) => (
          <ChatBubble key={index} message={msg} isUser={msg.isUser} />
        ))}
        {isProcessing && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-200 rounded-2xl px-4 py-3">
              <Loader2 size={20} className="animate-spin text-gray-600" />
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 px-4 py-3">
        {/* Show pending audio if not auto-send */}
        {!settings.autoSend && recordedAudio && (
          <div className="mb-3 flex items-center justify-between bg-blue-50 px-3 py-2 rounded-lg">
            <span className="text-sm text-blue-700">Audio recorded</span>
            <div className="flex gap-2">
              <button
                onClick={() => handleSendAudio()}
                className="px-3 py-1 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600"
              >
                Send
              </button>
              <button
                onClick={() => setRecordedAudio(null)}
                className="px-3 py-1 bg-gray-300 text-gray-700 text-sm rounded-lg hover:bg-gray-400"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <div className="flex items-center gap-2">
          {/* Microphone Button */}
          <button
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
            disabled={isProcessing}
            className={`p-3 rounded-full transition-all ${
              isRecording
                ? 'bg-red-500 text-white animate-pulse scale-110'
                : 'bg-blue-500 text-white hover:bg-blue-600'
            } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <Mic size={24} />
          </button>

          {/* Text Input */}
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendText()}
            placeholder="Type a message..."
            disabled={isProcessing}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />

          {/* Send Button */}
          <button
            onClick={handleSendText}
            disabled={!textInput.trim() || isProcessing}
            className="p-3 bg-blue-500 text-white rounded-full hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={20} />
          </button>
        </div>

        {isRecording && (
          <p className="text-center text-sm text-red-500 mt-2 animate-pulse">
            Recording... Release to stop
          </p>
        )}
      </div>

      {/* Settings Modal */}
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
