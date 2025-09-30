import React, { useState, useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const VoiceSearchDashboard = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [searchInfo, setSearchInfo] = useState(null);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  
  // Speech recognition
  const {
    transcript,
    listening,
    browserSupportsSpeechRecognition,
    resetTranscript
  } = useSpeechRecognition();

  // Update query when transcript changes
  useEffect(() => {
    if (transcript) {
      setQuery(transcript);
    }
  }, [transcript]);

  // Handle speech recognition start/stop
  const handleVoiceToggle = () => {
    try {
      if (!browserSupportsSpeechRecognition) {
        setError("Speech recognition not supported in this browser.");
        return;
      }
      if (listening) {
        SpeechRecognition.stopListening();
      } else {
        setError(null);
        resetTranscript();
        SpeechRecognition.startListening({ continuous: false });
      }
    } catch (e) {
      setError(`Voice capture failed: ${e?.message || e}`);
    }
  };

  const handleVoiceSearch = async () => {
    if (!query.trim()) {
      setError('Please speak or type a query first.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults([]);
    setSearchInfo(null);

    try {
      // Use the regular vector search endpoint with the transcribed text
      const response = await axios.post(`${BACKEND_URL}/search/vector`, {
        query: query,
        use_nlp: false
      });

      setResults(response.data.results || []);
      setSearchInfo({
        transcribed: query,
        translated: query, // Same as transcribed since we're using speech recognition
        language: 'en', // Speech recognition typically returns English
        confidence: 1.0,
        totalFound: response.data.total_found
      });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Search failed');
      console.error('Voice search error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const clearResults = () => {
    setResults([]);
    setSearchInfo(null);
    setQuery('');
    setError(null);
    resetTranscript();
  };

  const formatAmount = (amount) => {
    if (!amount) return 'N/A';
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(parseFloat(amount));
  };

  return (
    <div style={{ 
      maxWidth: '1200px', 
      margin: '0 auto', 
      padding: '20px',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* Header */}
      <div style={{ 
        textAlign: 'center', 
        marginBottom: '30px',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        padding: '30px',
        borderRadius: '15px',
        boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ margin: '0 0 10px 0', fontSize: '2.5rem', fontWeight: '700' }}>
          🎤 Voice Search
        </h1>
        <p style={{ margin: '0', fontSize: '1.1rem', opacity: 0.9 }}>
          Speak in any language - we'll translate and find what you're looking for
        </p>
      </div>

      {/* Voice Input Section */}
      <div style={{
        background: 'white',
        borderRadius: '15px',
        padding: '30px',
        marginBottom: '30px',
        boxShadow: '0 5px 20px rgba(0,0,0,0.08)',
        border: '1px solid #e5e7eb'
      }}>
        <h2 style={{ margin: '0 0 20px 0', color: '#374151', fontSize: '1.5rem' }}>
          Voice Search
        </h2>
        
        {/* Speech Recognition Status */}
        <div style={{ 
          marginBottom: '20px',
          padding: '15px',
          borderRadius: '10px',
          background: listening ? '#fef3c7' : '#f3f4f6',
          border: `2px solid ${listening ? '#f59e0b' : '#d1d5db'}`,
          textAlign: 'center'
        }}>
          <div style={{ 
            fontSize: '18px', 
            fontWeight: '600',
            color: listening ? '#92400e' : '#374151',
            marginBottom: '8px'
          }}>
            {listening ? '🎤 Listening... Speak now!' : '🎤 Click to start speaking'}
          </div>
          <div style={{ fontSize: '14px', color: listening ? '#92400e' : '#6b7280' }}>
            {listening ? 'Click stop when finished' : 'Your speech will be converted to text and searched'}
          </div>
        </div>

        {/* Query Display */}
        {query && (
          <div style={{
            background: '#f0f9ff',
            padding: '15px',
            borderRadius: '10px',
            marginBottom: '20px',
            border: '1px solid #0ea5e9'
          }}>
            <div style={{ fontSize: '14px', color: '#0369a1', marginBottom: '5px', fontWeight: '600' }}>
              Your Query:
            </div>
            <div style={{ fontSize: '16px', color: '#0c4a6e' }}>
              "{query}"
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: '15px', marginBottom: '20px', flexWrap: 'wrap' }}>
          {/* Voice Toggle Button */}
          <button
            onClick={handleVoiceToggle}
            disabled={!browserSupportsSpeechRecognition}
            style={{
              background: listening ? '#dc2626' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: browserSupportsSpeechRecognition ? 'pointer' : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              opacity: browserSupportsSpeechRecognition ? 1 : 0.6
            }}
          >
            {listening ? '⏹️ Stop Listening' : '🎤 Start Speaking'}
          </button>

          {/* Search Button */}
          <button
            onClick={handleVoiceSearch}
            disabled={!query.trim() || isProcessing}
            style={{
              background: query.trim() && !isProcessing ? '#f59e0b' : '#9ca3af',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: query.trim() && !isProcessing ? 'pointer' : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
            }}
          >
            {isProcessing ? '⏳ Searching...' : '🔍 Search'}
          </button>

          {/* Clear Button */}
          <button
            onClick={clearResults}
            style={{
              background: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
            }}
          >
            🗑️ Clear
          </button>
        </div>

        {/* Browser Support Warning */}
        {!browserSupportsSpeechRecognition && (
          <div style={{
            background: '#fef2f2',
            padding: '15px',
            borderRadius: '8px',
            border: '1px solid #fecaca',
            color: '#dc2626',
            fontSize: '14px'
          }}>
            ⚠️ Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div style={{
            background: '#fef2f2',
            color: '#dc2626',
            padding: '15px',
            borderRadius: '8px',
            border: '1px solid #fecaca',
            marginTop: '15px'
          }}>
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      {/* Search Info */}
      {searchInfo && (
        <div style={{
          background: 'white',
          borderRadius: '15px',
          padding: '25px',
          marginBottom: '30px',
          boxShadow: '0 5px 20px rgba(0,0,0,0.08)',
          border: '1px solid #e5e7eb'
        }}>
          <h3 style={{ margin: '0 0 15px 0', color: '#374151', fontSize: '1.3rem' }}>
            Search Information
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '15px' }}>
            <div>
              <strong>Transcribed:</strong> <span style={{ color: '#6b7280' }}>{searchInfo.transcribed}</span>
            </div>
            <div>
              <strong>Translated:</strong> <span style={{ color: '#059669' }}>{searchInfo.translated}</span>
            </div>
            <div>
              <strong>Language:</strong> <span style={{ color: '#3b82f6' }}>{searchInfo.language?.toUpperCase()}</span>
            </div>
            <div>
              <strong>Confidence:</strong> <span style={{ color: '#f59e0b' }}>{(searchInfo.confidence * 100).toFixed(1)}%</span>
            </div>
            <div>
              <strong>Results Found:</strong> <span style={{ color: '#dc2626' }}>{searchInfo.totalFound}</span>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div style={{
          background: 'white',
          borderRadius: '15px',
          padding: '25px',
          boxShadow: '0 5px 20px rgba(0,0,0,0.08)',
          border: '1px solid #e5e7eb'
        }}>
          <h3 style={{ margin: '0 0 20px 0', color: '#374151', fontSize: '1.3rem' }}>
            Search Results ({results.length})
          </h3>
          <div style={{ display: 'grid', gap: '20px' }}>
            {results.map((item, index) => (
              <div
                key={item.id || index}
                style={{
                  border: '1px solid #e5e7eb',
                  borderRadius: '12px',
                  padding: '20px',
                  background: '#fafafa',
                  transition: 'all 0.2s',
                  cursor: 'pointer'
                }}
                onMouseOver={(e) => {
                  e.target.style.background = '#f3f4f6';
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)';
                }}
                onMouseOut={(e) => {
                  e.target.style.background = '#fafafa';
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '15px' }}>
                  <div>
                    <h4 style={{ margin: '0 0 8px 0', fontSize: '1.2rem', color: '#1f2937' }}>
                      {item.item_name}
                    </h4>
                    <div style={{ color: '#6b7280', fontSize: '0.9rem' }}>
                      Score: <span style={{ color: '#059669', fontWeight: '600' }}>{item.score}</span>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '1.3rem', fontWeight: '700', color: '#dc2626' }}>
                      {formatAmount(item.amount)}
                    </div>
                  </div>
                </div>

                {item.attributes_parsed && Object.keys(item.attributes_parsed).length > 0 && (
                  <div style={{ marginBottom: '15px' }}>
                    <h5 style={{ margin: '0 0 10px 0', color: '#374151', fontSize: '1rem' }}>Attributes:</h5>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '8px' }}>
                      {Object.entries(item.attributes_parsed).map(([key, value]) => (
                        <div key={key} style={{
                          background: '#f3f4f6',
                          padding: '8px 12px',
                          borderRadius: '6px',
                          fontSize: '0.9rem'
                        }}>
                          <strong>{key}:</strong> {value}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {item.image && (
                  <div style={{ textAlign: 'center' }}>
                    <img
                      src={item.image}
                      alt={item.item_name}
                      style={{
                        maxWidth: '200px',
                        maxHeight: '150px',
                        borderRadius: '8px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                      }}
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isProcessing && results.length === 0 && !searchInfo && (
        <div style={{
          textAlign: 'center',
          padding: '60px 20px',
          color: '#6b7280'
        }}>
          <div style={{ fontSize: '4rem', marginBottom: '20px' }}>🎤</div>
          <h3 style={{ margin: '0 0 10px 0', color: '#374151' }}>Ready for Voice Search</h3>
          <p style={{ margin: '0' }}>Record your voice or upload an audio file to get started</p>
        </div>
      )}
    </div>
  );
};

export default VoiceSearchDashboard;
