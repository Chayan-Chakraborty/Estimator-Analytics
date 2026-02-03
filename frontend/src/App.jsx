import { useEffect, useRef, useState } from "react";
import SpeechRecognition, { useSpeechRecognition } from "react-speech-recognition";
import axios from "axios";
import VoiceSearchDashboard from "./VoiceSearchDashboard";

function App() {
  const [query, setQuery] = useState("");
  const [textQuery, setTextQuery] = useState("");
  const [micQuery, setMicQuery] = useState("");
  const [results, setResults] = useState([]);
  const [totalResultsCount, setTotalResultsCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [totalPages, setTotalPages] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [hasPrevious, setHasPrevious] = useState(false);
  const [lastSearchType, setLastSearchType] = useState(null); // 'text' or 'mic'
  const [isSearching, setIsSearching] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState("");

  // Details state (fresh per new requirement)
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState("");
  const [areaAmountStats, setAreaAmountStats] = useState([]); // [{ area, min, max, avg }]
  const [areaStatsAvg, setAreaStatsAvg] = useState(null);

  // Sidebar filter states
  const [filters, setFilters] = useState({
    itemName: "",
    city: "",
    measurement: ""
  });
  const [showSidebar, setShowSidebar] = useState(false);
  const [filteredResults, setFilteredResults] = useState([]);
  const [isFiltering, setIsFiltering] = useState(false);
  const [viewMode, setViewMode] = useState("cards"); // "cards", "table", or "voice"
  const [useNLP, setUseNLP] = useState(true);
  const [extractedFilters, setExtractedFilters] = useState({});

  // Multilingual support states
  const [useMultilingual, setUseMultilingual] = useState(false);
  const [sourceLanguage, setSourceLanguage] = useState("");
  const [detectedLanguage, setDetectedLanguage] = useState("");
  const [translatedQuery, setTranslatedQuery] = useState("");
  const [translationConfidence, setTranslationConfidence] = useState(0);
  const [showTranslation, setShowTranslation] = useState(false);
  
  // Voice recognition language states
  const [voiceLanguage, setVoiceLanguage] = useState("en-US");
  const [detectedVoiceLanguage, setDetectedVoiceLanguage] = useState("");

  const recognitionRef = useRef(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

  const extractImageUrl = (imageVal) => {
    if (!imageVal) return null;
    if (typeof imageVal === 'string') {
      const trimmed = imageVal.trim();
      if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
        try {
          const parsed = JSON.parse(trimmed);
          if (parsed && typeof parsed === 'object' && parsed.default) return parsed.default;
        } catch (e) {
          return null;
        }
      }
      return trimmed;
    }
    if (typeof imageVal === 'object' && imageVal.default) return imageVal.default;
    return null;
  };

  const toSafeNumber = (val) => {
    if (typeof val === 'number') return val;
    if (typeof val === 'string' && val.trim() !== '' && !isNaN(Number(val))) return Number(val);
    return null;
  };

  // Calculate avg amount for same city and item_name
  const calculateAvgAmount = (currentItem, allItems) => {
    if (!currentItem || !allItems || allItems.length === 0) return null;
    
    // Normalize area and item_name for case-insensitive matching
    const normalize = (str) => (str || "").toString().trim().toLowerCase();
    const currentArea = normalize(currentItem.area);
    const currentItemName = normalize(currentItem.item_name);
    
    if (!currentArea || !currentItemName) return null;
    
    const sameCityItemName = allItems.filter(item => {
      const itemArea = normalize(item.area);
      const itemName = normalize(item.item_name);
      return itemArea === currentArea && itemName === currentItemName;
    });
    
    if (sameCityItemName.length <= 1) return null;
    
    const amounts = sameCityItemName
      .map(item => toSafeNumber(item.amount))
      .filter(amount => amount !== null && amount > 0);
    
    if (amounts.length === 0) return null;
    
    return amounts.reduce((sum, amount) => sum + amount, 0) / amounts.length;
  };

  const buildParsedAttributes = (rawAttributes, fallbackMeasurement) => {
    // rawAttributes may be an object or JSON string or null
    let obj = null;
    if (!rawAttributes) return {};
    if (typeof rawAttributes === 'string') {
      try {
        obj = JSON.parse(rawAttributes);
      } catch (e) {
        return {};
      }
    } else if (typeof rawAttributes === 'object') {
      obj = rawAttributes;
    }
    if (!obj || typeof obj !== 'object') return {};

    const from = (key) => (obj[key] && typeof obj[key] === 'object') ? obj[key] : {};

    const mes = from('MES_WD_ATTR');
    const width = mes.width ? String(mes.width) : null;
    const length = mes.length ? String(mes.length) : null;
    const unit = mes.selectedOption ? String(mes.selectedOption) : null;
    const measurement = width && length && unit ? `${width}x${length} ${unit}` : (fallbackMeasurement || null);

    const rat = from('RAT_WD_ATTR');
    const rateVal = rat.value != null ? String(rat.value) : null;
    const rateUnit = rat.selectedOption ? String(rat.selectedOption) : null;
    const rate = rateVal && rateUnit ? `${rateVal} ${rateUnit}` : (rateVal || rateUnit || null);

    const mat = from('MAT_WD_ATTR');
    const material = mat.selectedOption || mat.value || null;

    const fin = from('FIN_WD_ATTR');
    const finish = fin.selectedOption || fin.value || null;

    const out = {};
    if (measurement) out['Measurement'] = measurement;
    if (rate) out['Rate'] = rate;
    if (material) out['Material'] = material;
    if (finish) out['Finish'] = finish;
    return out;
  };

  // Setup native Web Speech recognition (if available)
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      
      // Use automatic language detection
      recognition.lang = "auto";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.continuous = false;

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setMicQuery(transcript);
        
        // Try to detect the language from the transcript
        if (event.results[0][0].confidence) {
          // Some browsers provide language detection
          const detectedLang = detectLanguageFromText(transcript);
          setDetectedVoiceLanguage(detectedLang);
        }
      };
      
      recognition.onend = () => setIsListening(false);
      
      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, []);

  // React Speech Recognition (wrapper around Web Speech API)
  const {
    transcript,
    listening,
    browserSupportsSpeechRecognition,
    resetTranscript
  } = useSpeechRecognition();

  useEffect(() => {
    setIsListening(listening);
  }, [listening]);

  useEffect(() => {
    if (transcript) setMicQuery(transcript);
  }, [transcript]);


  // Perform search using NLP or Vector endpoint based on toggle
  const handleSearch = async () => {
    setIsSearching(true);
    setError("");
    setResults([]);
    setAreaAmountStats([]);
    setAreaStatsAvg(null);
    setTranslatedQuery("");
    setDetectedLanguage("");
    setTranslationConfidence(0);
    
    try {
      let endpoint, requestData;
      // Preprocess romanized or non-English text to English first (client-side assist)
      let finalQuery = query;
      if (useMultilingual && query && query.trim()) {
        try {
          const preprocessRes = await axios.post(`${BACKEND_URL}/speech/text-to-english`, {
            text: query,
            source_language: sourceLanguage || detectedVoiceLanguage || null
          });
          const englishQuery = preprocessRes?.data?.english_query;
          if (englishQuery && typeof englishQuery === 'string' && englishQuery.trim() !== '') {
            finalQuery = englishQuery;
            if (englishQuery !== query) {
              setTranslatedQuery(englishQuery);
              setDetectedLanguage('auto');
              setTranslationConfidence(0);
              setShowTranslation(true);
            }
          }
        } catch (e) {
          // Non-fatal: fall back to raw query
        }
      }
      
      if (useMultilingual) {
        // Use multilingual endpoints
        if (useNLP) {
          endpoint = '/search/multilingual/nlp';
        } else {
          endpoint = '/search/multilingual/vector';
        }
        requestData = {
          query: finalQuery,
          source_language: sourceLanguage || null,
          target_language: "en",
          use_nlp: useNLP
        };
      } else {
        // Use regular endpoints
        endpoint = useNLP ? '/search/nlp' : '/search/vector';
        requestData = {
          query: finalQuery,
          use_nlp: useNLP
        };
      }
      
      const res = await axios.post(`${BACKEND_URL}${endpoint}`, requestData);

      let raw, translationInfo;
      
      if (useMultilingual) {
        // Handle multilingual response
        raw = res?.data?.search_results;
        translationInfo = {
          originalQuery: res?.data?.original_query,
          translatedQuery: res?.data?.translated_query,
          detectedLanguage: res?.data?.detected_language,
          confidence: res?.data?.translation_confidence
        };
        
        setTranslatedQuery(translationInfo.translatedQuery);
        setDetectedLanguage(translationInfo.detectedLanguage);
        setTranslationConfidence(translationInfo.confidence);
        setShowTranslation(translationInfo.translatedQuery !== query);
      } else {
        raw = res?.data?.results;
      }
      
      const safeArray = Array.isArray(raw) ? raw : [];

      const normalized = safeArray.map((item) => {
        const measurementSqft = toSafeNumber(item?.measurement_sqft);
        const fallbackMeasurement = (() => {
          if (measurementSqft && measurementSqft > 0) return `${measurementSqft} sqft`;
          return null;
        })();

        const attrsParsed = item?.attributes_parsed && typeof item.attributes_parsed === 'object'
          ? item.attributes_parsed
          : buildParsedAttributes(item?.attributes, fallbackMeasurement);

        return {
          ...item,
          attributes_parsed: attrsParsed
        };
      });

      setResults(normalized);

      // /search/nlp does not return area_stats
    } catch (e) {
      const message = e?.response?.data?.detail || e?.message || 'Unknown error';
      setError(`Failed to fetch results: ${message}`);
    } finally {
      setIsSearching(false);
    }
  };

  // Plain vector search for text input
  const handleTextVectorSearch = async (page = null) => {
    // page can be a number (for pagination) or null/undefined (for new search)
    const targetPage = (typeof page === 'number' && page > 0) ? page : 1;
    setIsSearching(true);
    setError("");
    setResults([]);
    setCurrentPage(targetPage);
    setLastSearchType('text');
    try {
      const res = await axios.post(`${BACKEND_URL}/search/vector`, {
        query: String(textQuery || ""),
        use_nlp: false,
        is_voice: false,
        page: targetPage,
        page_size: pageSize
      });
      const raw = res?.data?.results;
      const safeArray = Array.isArray(raw) ? raw : [];
      setResults(safeArray);
      setTotalResultsCount(res?.data?.total_found || 0);
      setTotalPages(res?.data?.total_pages || 0);
      setHasNext(res?.data?.has_next || false);
      setHasPrevious(res?.data?.has_previous || false);
    } catch (e) {
      const message = e?.response?.data?.detail || e?.message || 'Unknown error';
      setError(`Failed to fetch results: ${message}`);
    } finally {
      setIsSearching(false);
    }
  };

  // Vector search for mic (voice) input with flag
  const handleMicVectorSearch = async (page = null) => {
    // page can be a number (for pagination) or null/undefined (for new search)
    const targetPage = (typeof page === 'number' && page > 0) ? page : 1;
    setIsSearching(true);
    setError("");
    setResults([]);
    setCurrentPage(targetPage);
    setLastSearchType('mic');
    try {
      const res = await axios.post(`${BACKEND_URL}/search/vector`, {
        query: String(micQuery || ""),
        use_nlp: false,
        is_voice: true,
        page: targetPage,
        page_size: pageSize
      });
      const raw = res?.data?.results;
      const safeArray = Array.isArray(raw) ? raw : [];
      setResults(safeArray);
      setTotalResultsCount(res?.data?.total_found || 0);
      setTotalPages(res?.data?.total_pages || 0);
      setHasNext(res?.data?.has_next || false);
      setHasPrevious(res?.data?.has_previous || false);
    } catch (e) {
      const message = e?.response?.data?.detail || e?.message || 'Unknown error';
      setError(`Failed to fetch results: ${message}`);
    } finally {
      setIsSearching(false);
    }
  };

  // On opening details: use filtered stats endpoint
  useEffect(() => {
    const fetchFilteredStats = async () => {
      if (!selectedItem) return;
      setDetailsLoading(true);
      setDetailsError("");
      setAreaAmountStats([]);
      try {
        // Parse measurement string to number (supports both sqft and 4x4 format)
        const measurementValue = parseMeasurement(filters.measurement);
        
        const res = await axios.post(`${BACKEND_URL}/search/filtered-stats`, {
          item_name: selectedItem.item_name,
          city: filters.city || null,
          measurement: measurementValue
        });

        const areaStats = res?.data?.area_stats || [];
        setAreaAmountStats(areaStats);
      } catch (e) {
        const message = e?.response?.data?.detail || e?.message || 'Unknown error';
        setDetailsError(`Failed to fetch filtered stats: ${message}`);
      } finally {
        setDetailsLoading(false);
      }
    };
    fetchFilteredStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedItem, filters]);


  const handleIngest = async () => {
    setIsIngesting(true);
    setError("");
    try {
      await axios.post(`${BACKEND_URL}/ingest`);
    } catch (e) {
      const message = e?.response?.data?.detail || e?.message || 'Unknown error';
      setError(`Failed to ingest: ${message}`);
    } finally {
      setIsIngesting(false);
    }
  };

  // Test translation function
  const testTranslation = async (text) => {
    try {
      const res = await axios.get(`${BACKEND_URL}/translate`, {
        params: {
          query: text,
          source_language: sourceLanguage || null,
          target_language: "en"
        }
      });
      return res.data;
    } catch (e) {
      console.error("Translation test failed:", e);
      return null;
    }
  };

  // Language detection from text - Indian languages only
  const detectLanguageFromText = (text) => {
    const lowerText = text.toLowerCase();
    
    // Bengali detection (Bengali script)
    if (/[\u0980-\u09FF]/.test(text)) return 'bn';
    
    // Hindi detection (Devanagari script)
    if (/[\u0900-\u097F]/.test(text)) return 'hi';
    
    // Tamil detection (Tamil script)
    if (/[\u0B80-\u0BFF]/.test(text)) return 'ta';
    
    // Telugu detection (Telugu script)
    if (/[\u0C00-\u0C7F]/.test(text)) return 'te';
    
    // Gujarati detection (Gujarati script)
    if (/[\u0A80-\u0AFF]/.test(text)) return 'gu';
    
    // Punjabi detection (Gurmukhi script)
    if (/[\u0A00-\u0A7F]/.test(text)) return 'pa';
    
    // Marathi detection (Devanagari script - same as Hindi but different vocabulary)
    if (/\b(ghar|ghar|kotha|kothay|kemon|kemon ache|bhalo|kharap|sundor|bhalo lagche|ami|tumi|apni|se|ora|tara|amra|tomra|apnara|amader|tomader|apnader|tader|amra|tomra|apnara|kothay|kothay|kemon|kemon ache|bhalo|kharap|sundor|bhalo lagche|ami|tumi|apni|se|ora|tara|amra|tomra|apnara|amader|tomader|apnader|tader|amra|tomra|apnara|kothay|kothay|kemon|kemon ache|bhalo|kharap|sundor|bhalo lagche)\b/.test(lowerText)) return 'mr';
    
    // Bengali detection (transliterated words)
    if (/\b(bichana|khana|ghar|bari|kotha|kothay|kemon|kemon ache|bhalo|kharap|sundor|bhalo lagche|ami|tumi|apni|se|ora|tara|amra|tomra|apnara|amader|tomader|apnader|tader|amra|tomra|apnara|kothay|kothay|kemon|kemon ache|bhalo|kharap|sundor|bhalo lagche|ami|tumi|apni|se|ora|tara|amra|tomra|apnara|amader|tomader|apnader|tader|amra|tomra|apnara|kothay|kothay|kemon|kemon ache|bhalo|kharap|sundor|bhalo lagche)\b/.test(lowerText)) return 'bn';
    
    // Default to English
    return 'en';
  };

  // Language mapping for voice recognition - Indian languages only
  const getVoiceLanguageCode = (langCode) => {
    const languageMap = {
      'bn': 'bn-BD',      // Bengali
      'hi': 'hi-IN',      // Hindi
      'ta': 'ta-IN',      // Tamil
      'te': 'te-IN',      // Telugu
      'gu': 'gu-IN',      // Gujarati
      'pa': 'pa-IN',      // Punjabi
      'mr': 'mr-IN',      // Marathi
      'en': 'en-US'       // English
    };
    return languageMap[langCode] || 'en-US';
  };

  // Auto-detect language from voice input and sync with multilingual mode
  useEffect(() => {
    if (detectedVoiceLanguage && useMultilingual) {
      // Auto-set source language based on detected voice language
      setSourceLanguage(detectedVoiceLanguage);
    }
  }, [detectedVoiceLanguage, useMultilingual]);

  // Auto-detect language from text input when in multilingual mode
  useEffect(() => {
    if (query && useMultilingual && !sourceLanguage) {
      const detectedLang = detectLanguageFromText(query);
      if (detectedLang && detectedLang !== 'en') {
        setSourceLanguage(detectedLang);
        setDetectedLanguage(detectedLang);
      }
    }
  }, [query, useMultilingual, sourceLanguage]);

  const handleVoiceToggle = () => {
    try {
      if (!browserSupportsSpeechRecognition) {
        setError("Speech recognition not supported in this browser.");
        return;
      }
      if (listening) {
        SpeechRecognition.stopListening();
      } else {
        setDetectedVoiceLanguage("");
        resetTranscript();
        SpeechRecognition.startListening({ continuous: false });
      }
    } catch (e) {
      setError(`Voice capture failed: ${e?.message || e}`);
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const parseMeasurement = (measurementStr) => {
    if (!measurementStr || !measurementStr.trim()) {
      return null;
    }
    
    const str = measurementStr.trim().toLowerCase();
    
    // Handle 4x4 format (like "4x4", "3x3", "2x2")
    if (str.includes('x')) {
      const parts = str.split('x');
      if (parts.length === 2) {
        const width = parseFloat(parts[0]);
        const length = parseFloat(parts[1]);
        if (!isNaN(width) && !isNaN(length)) {
          return width * length; // Convert to sqft
        }
      }
    }
    
    // Handle direct sqft value
    const directValue = parseFloat(str);
    if (!isNaN(directValue)) {
      return directValue;
    }
    
    return null;
  };

  const applyFilters = async () => {
    setIsFiltering(true);
    setError("");
    try {
      // Parse measurement string to number (supports both sqft and 4x4 format)
      const measurementValue = parseMeasurement(filters.measurement);
      
      const res = await axios.post(`${BACKEND_URL}/search/filtered-stats/only`, {
        item_name: filters.itemName || null,
        city: filters.city || null,
        measurement: measurementValue
      });
      // New endpoint returns only area_stats
      setFilteredResults([]);
      const areaStats = res?.data?.area_stats || [];
      setAreaAmountStats(areaStats);
      setAreaStatsAvg(areaStats.length > 0 && typeof areaStats[0].avg === 'number' ? areaStats[0].avg : null);
    } catch (e) {
      const message = e?.response?.data?.detail || e?.message || 'Unknown error';
      setError(`Failed to apply filters: ${message}`);
    } finally {
      setIsFiltering(false);
    }
  };

  const clearFilters = () => {
    setFilters({
      itemName: "",
      city: "",
      measurement: ""
    });
    setFilteredResults([]);
    setAreaAmountStats([]);
    setAreaStatsAvg(null);
  };

  const buttonStyle = {
    padding: "8px 12px",
    borderRadius: 8,
    border: "1px solid #e5e7eb",
    background: "#111827",
    color: "#fff",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 6
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f9fafb" }}>
      {/* Top Toolbar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: 16,
        borderBottom: "1px solid #e5e7eb",
        position: "sticky",
        top: 0,
        background: "#ffffff",
        zIndex: 10,
        "@media (max-width: 768px)": {
          padding: "12px 16px",
          flexWrap: "wrap",
          gap: 8
        }
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 28, height: 28, background: "#111827", borderRadius: 6 }} />
          <h2 style={{ margin: 0 }}>Estimator Item Search</h2>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setUseNLP(!useNLP)}
            style={{ ...buttonStyle, background: useNLP ? "#7c3aed" : "#6b7280" }}
            title={useNLP ? "NLP Mode: Extracts filters from natural language" : "Vector Mode: Pure semantic similarity search"}
          >
            {useNLP ? "NLP Mode" : "Vector Mode"}
          </button>
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            style={{ ...buttonStyle, background: showSidebar ? "#dc2626" : "#111827" }}
          >
            {showSidebar ? "Hide Filters" : "Show Filters"}
          </button>
          <button
            onClick={() => {
              if (viewMode === "cards") setViewMode("table");
              else if (viewMode === "table") setViewMode("voice");
              else setViewMode("cards");
            }}
            style={{ ...buttonStyle, background: "#059669" }}
          >
            {viewMode === "cards" ? "Table View" : viewMode === "table" ? "Voice Search" : "Card View"}
          </button>
          <button
            onClick={() => setUseMultilingual(!useMultilingual)}
            style={{ ...buttonStyle, background: useMultilingual ? "#f59e0b" : "#6b7280" }}
            title={useMultilingual ? "Multilingual Mode: Search in any language" : "English Mode: Search in English only"}
          >
            {useMultilingual ? "🌍 Multilingual" : "🇺🇸 English"}
          </button>
          <button onClick={handleIngest} style={buttonStyle} disabled={isIngesting}>
            {isIngesting ? "Ingesting..." : "Ingest Data"}
          </button>
        </div>
      </div>

      {/* Sidebar */}
      {showSidebar && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          height: "100vh",
          width: "320px",
          background: "#fff",
          borderRight: "1px solid #e5e7eb",
          boxShadow: "2px 0 8px rgba(0,0,0,0.12)",
          padding: 16,
          overflowY: "auto",
          zIndex: 20,
          marginTop: "73px",
          "@media (max-width: 1024px)": {
            width: "300px"
          },
          "@media (max-width: 768px)": {
            width: "100vw",
            height: "100vh",
            marginTop: 0,
            zIndex: 30
          }
        }}>
          <h3 style={{ margin: "0 0 16px 0" }}>Filters</h3>

          {/* Search Mode Examples */}
          <div style={{ marginBottom: 16, padding: 12, background: useNLP ? "#f0f9ff" : "#f0fdf4", borderRadius: 6, border: `1px solid ${useNLP ? "#0ea5e9" : "#22c55e"}` }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: useNLP ? "#0369a1" : "#15803d", marginBottom: 8 }}>
              {useMultilingual ? "🌍 Multilingual Examples:" : (useNLP ? "💡 NLP Mode Examples:" : "🔍 Vector Mode Examples:")}
            </div>
            <div style={{ fontSize: 11, color: useNLP ? "#0369a1" : "#15803d", lineHeight: 1.4 }}>
              {useMultilingual ? (
                <>
                  <strong>English:</strong> "wooden door in Mumbai"<br />
                  <strong>বাংলা:</strong> "কাঠের দরজা মুম্বাইতে"<br />
                  <strong>हिन्दी:</strong> "लकड़ी का दरवाजा मुंबई में"<br />
                  <strong>தமிழ்:</strong> "மும்பையில் மர கதவு"<br />
                  <strong>తెలుగు:</strong> "ముంబైలో చెక్క తలుపు"<br />
                  <strong>ગુજરાતી:</strong> "મુંબઈમાં લાકડાના દરવાજા"
                </>
              ) : useNLP ? (
                <>
                  • "TV Wall Unit in Bangalore"<br />
                  • "between 50-100 sqft"<br />
                  • "under ₹50000"<br />
                  • "Console Table from Mumbai"
                </>
              ) : (
                <>
                  • "wooden furniture"<br />
                  • "modern table design"<br />
                  • "bedroom storage"<br />
                  • "dining room furniture"
                </>
              )}
            </div>
          </div>

          {/* Item Name Filter */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>Item Name</label>
            <input
              type="text"
              placeholder={useNLP ? "Try: 'TV Wall Unit' or 'Console Table'" : "Enter item name"}
              value={filters.itemName}
              onChange={(e) => handleFilterChange('itemName', e.target.value)}
              style={{
                width: "100%",
                padding: "8px 12px",
                border: "1px solid #e5e7eb",
                borderRadius: 6,
                fontSize: 14
              }}
            />
          </div>

          {/* City Filter */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>City</label>
            <input
              type="text"
              placeholder={useNLP ? "Try: 'in Bangalore' or 'from Mumbai'" : "Enter city name"}
              value={filters.city}
              onChange={(e) => handleFilterChange('city', e.target.value)}
              style={{
                width: "100%",
                padding: "8px 12px",
                border: "1px solid #e5e7eb",
                borderRadius: 6,
                fontSize: 14
              }}
            />
          </div>

          {/* Measurement Single Input */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>Measurement (sqft)</label>
            <input
              type="text"
              placeholder={useNLP ? "e.g., 120, 4x4, or 3x3" : "Enter sqft (120) or dimensions (4x4)"}
              value={filters.measurement}
              onChange={(e) => handleFilterChange('measurement', e.target.value)}
              style={{
                width: "100%",
                padding: "8px 12px",
                border: "1px solid #e5e7eb",
                borderRadius: 6,
                fontSize: 14
              }}
            />
            {useNLP && (
              <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
                Try: "50-100 sqft" or "200 sqft"
              </div>
            )}
          </div>

          {/* Filter Actions */}
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={applyFilters}
              style={{
                ...buttonStyle,
                flex: 1,
                background: "#111827"
              }}
              disabled={isFiltering}
            >
              {isFiltering ? "Filtering..." : "Apply Filters"}
            </button>
            <button
              onClick={clearFilters}
              style={{
                ...buttonStyle,
                flex: 1,
                background: "#6b7280"
              }}
            >
              Clear
            </button>
          </div>

          {/* Area Stats Average Value */}
          {areaAmountStats.length > 0 && (
            <div style={{ marginTop: 16, padding: 12, background: "#f3f4f6", borderRadius: 6 }}>
              <div style={{ fontSize: 14, fontWeight: 500 }}>
                Area Stats Avg: ₹{areaAmountStats[0]?.avg ? areaAmountStats[0].avg.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 0}
              </div>
            </div>
          )}
        </div>
      )}

      <div style={{ 
        maxWidth: 960, 
        margin: "0 auto", 
        padding: "0 16px", 
        marginLeft: showSidebar ? "336px" : "auto",
        transition: "margin-left 0.3s ease",
        "@media (max-width: 1024px)": {
          marginLeft: showSidebar ? "320px" : "auto"
        },
        "@media (max-width: 768px)": {
          marginLeft: "auto",
          padding: "0 12px"
        }
      }}>
        
        {/* {useMultilingual && (
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
            margin: "16px auto",
            padding: "8px 16px",
            background: "#f0f9ff",
            borderRadius: 8,
            border: "1px solid #0ea5e9",
            maxWidth: 640
          }}>
            <span style={{ fontSize: 14, fontWeight: 600, color: "#0369a1" }}>
              🌍 Auto-detect Indian Languages: বাংলা, हिन्दी, தமிழ், తెలుగు, ગુજરાતી, ਪੰਜਾਬੀ, मराठी
            </span>
          </div>
        )} */}

        {/* Translation Display */}
        {showTranslation && translatedQuery && (
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            margin: "8px auto",
            padding: "8px 16px",
            background: "#f0fdf4",
            borderRadius: 8,
            border: "1px solid #22c55e",
            maxWidth: 640
          }}>
            <span style={{ fontSize: 12, color: "#15803d" }}>
              🌍 Detected: <strong>{detectedLanguage}</strong>
            </span>
            <span style={{ fontSize: 12, color: "#15803d" }}>
              ➡️ Translated: <strong>"{translatedQuery}"</strong>
            </span>
            {translationConfidence > 0 && (
              <span style={{ fontSize: 12, color: "#15803d" }}>
                (Confidence: {Math.round(translationConfidence * 100)}%)
              </span>
            )}
          </div>
        )}

        {/* Two search cards: Text and Mic */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
          margin: "16px auto",
          maxWidth: 960,
          "@media (max-width: 768px)": { gridTemplateColumns: "1fr" }
        }}>
          {/* Text Input Card */}
          <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, background: "#fff", padding: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Text Search</div>
            <textarea
              rows={3}
              placeholder="Type your query..."
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              style={{ width: "100%", border: "1px solid #e5e7eb", borderRadius: 6, padding: 8, resize: "vertical" }}
            />
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 8 }}>
              <button onClick={(e) => { e.preventDefault(); handleTextVectorSearch(); }} style={buttonStyle} disabled={isSearching}>
                {isSearching ? "Searching..." : "Search"}
              </button>
            </div>
          </div>

          {/* Mic Input Card */}
          <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, background: "#fff", padding: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Voice Search</div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <button
                onClick={handleVoiceToggle}
                title="Toggle microphone"
                style={{ ...buttonStyle, background: isListening ? "#dc2626" : "#111827" }}
              >
                {isListening ? "Stop" : "🎤 Mic"}
              </button>
              <input
                type="text"
                readOnly
                placeholder="Converted text will appear here..."
                value={micQuery}
                style={{ flex: 1, border: "1px solid #e5e7eb", borderRadius: 6, padding: 8, background: "#f9fafb" }}
              />
              <button onClick={(e) => { e.preventDefault(); handleMicVectorSearch(); }} style={buttonStyle} disabled={isSearching}>
                {isSearching ? "Searching..." : "Mic Search"}
              </button>
            </div>
          </div>
        </div>

        {/* Auto Language Detection Status */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          margin: "8px auto",
          padding: "6px 12px",
          background: isListening ? "#fef3c7" : (useMultilingual ? "#f0f9ff" : "#f8fafc"),
          borderRadius: 6,
          border: `1px solid ${isListening ? "#f59e0b" : (useMultilingual ? "#0ea5e9" : "#e2e8f0")}`,
          maxWidth: 640
        }}>
          {/* <span style={{ fontSize: 12, color: isListening ? "#92400e" : (useMultilingual ? "#0369a1" : "#64748b"), fontWeight: 500 }}>
            {isListening ? "🎤 Auto-detecting Indian languages..." : (useMultilingual ? "🌍 Multilingual search active" : "🎤 Voice recognition ready for Indian languages")}
          </span> */}
          {detectedVoiceLanguage && (
            <span style={{ fontSize: 10, color: "#10b981", fontWeight: 500 }}>
              🌍 Detected: {detectedVoiceLanguage.toUpperCase()}
            </span>
          )}
          {isListening && (
            <span style={{ fontSize: 10, color: "#dc2626", fontWeight: 600, animation: "pulse 1s infinite" }}>
              ● LISTENING
            </span>
          )}
        </div>

        {/* Area Stats Avg display near search bar
        {areaStatsAvg !== null && (
          <div style={{
            background: "#f3f4f6",
            border: "1px solid #e5e7eb",
            padding: 10,
            borderRadius: 8,
            margin: "0 auto 8px",
            maxWidth: 640,
            color: "#111827"
          }}>
            Area Stats Avg: ₹{Number(areaStatsAvg).toFixed(2)}
          </div>
        )} */}

        {/* Extracted Filters Display - hidden in voice mode */}
        {viewMode !== "voice" && useNLP && Object.keys(extractedFilters).length > 0 && (
          <div style={{
            background: "#f0f9ff",
            border: "1px solid #0ea5e9",
            borderRadius: 8,
            padding: 12,
            margin: "8px auto",
            maxWidth: 640
          }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#0369a1", marginBottom: 8 }}>
              🤖 NLP Extracted Filters:
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {Object.entries(extractedFilters).map(([key, value]) => (
                <div key={key} style={{
                  background: "#fff",
                  border: "1px solid #0ea5e9",
                  borderRadius: 16,
                  padding: "4px 8px",
                  fontSize: 12,
                  color: "#0369a1"
                }}>
                  <strong>{key.replace('_', ' ')}:</strong> {value}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div style={{
            background: "#fef2f2",
            color: "#b91c1c",
            border: "1px solid #fecaca",
            padding: 10,
            borderRadius: 8,
            marginTop: 8
          }}>
            {error}
          </div>
        )}


        {/* Results count and pagination info */}
        {!isSearching && !error && results.length > 0 && (
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 12,
            padding: "8px 16px",
            background: "#f3f4f6",
            borderRadius: 6,
            fontSize: 14,
            flexWrap: "wrap",
            gap: 8
          }}>
            <span style={{ color: "#6b7280" }}>
              Showing {((currentPage - 1) * pageSize) + 1} - {Math.min(currentPage * pageSize, totalResultsCount)} of {totalResultsCount} results
            </span>
            {totalPages > 1 && (
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const newPage = currentPage - 1;
                    if (lastSearchType === 'text') {
                      handleTextVectorSearch(newPage);
                    } else if (lastSearchType === 'mic') {
                      handleMicVectorSearch(newPage);
                    }
                  }}
                  disabled={!hasPrevious || isSearching || !lastSearchType}
                  style={{
                    padding: "6px 12px",
                    border: "1px solid #e5e7eb",
                    borderRadius: 6,
                    background: hasPrevious ? "#fff" : "#f3f4f6",
                    color: hasPrevious ? "#111827" : "#9ca3af",
                    cursor: hasPrevious ? "pointer" : "not-allowed",
                    fontSize: 14
                  }}
                >
                  Previous
                </button>
                <span style={{ color: "#6b7280" }}>
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const newPage = currentPage + 1;
                    if (lastSearchType === 'text') {
                      handleTextVectorSearch(newPage);
                    } else if (lastSearchType === 'mic') {
                      handleMicVectorSearch(newPage);
                    }
                  }}
                  disabled={!hasNext || isSearching || !lastSearchType}
                  style={{
                    padding: "6px 12px",
                    border: "1px solid #e5e7eb",
                    borderRadius: 6,
                    background: hasNext ? "#fff" : "#f3f4f6",
                    color: hasNext ? "#111827" : "#9ca3af",
                    cursor: hasNext ? "pointer" : "not-allowed",
                    fontSize: 14
                  }}
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}

        {/* Empty state */}
        {!isSearching && !error && results.length === 0 && (
          <div style={{ textAlign: "center", color: "#6b7280", marginTop: 24 }}>
            No results yet. Try a search.
          </div>
        )}

        {/* Voice Search Dashboard */}
        {viewMode === "voice" && <VoiceSearchDashboard />}

        {/* Results Display */}
        {viewMode === "cards" ? (
          /* Card View */
          <div style={{
            marginTop: 12,
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: 16,
            "@media (max-width: 768px)": {
              gridTemplateColumns: "1fr",
              gap: 12
            }
          }}>
            {(filteredResults.length > 0 ? filteredResults : results).map((item) => {
              const key = item.item_identifier || item.id || item.item_name;
              const imgUrl = extractImageUrl(item.image);
              const sqft = toSafeNumber(item.measurement_sqft);
              const amountNum = toSafeNumber(item.amount);
              const avgPerSqft = sqft && sqft > 0 && amountNum != null ? (amountNum / sqft) : null;
              const subtitleParts = [item.room_name, item.project_name, item.area].filter(Boolean);
              // Use backend-calculated average_price if available, otherwise fallback to client-side calculation
              const avgAmount = item.average_price !== undefined && item.average_price !== null 
                ? item.average_price 
                : calculateAvgAmount(item, filteredResults.length > 0 ? filteredResults : results);

              const attrs = item.attributes_parsed && typeof item.attributes_parsed === 'object' ? item.attributes_parsed : {};

              return (
                <div key={key}
                  style={{
                    border: "1px solid #e5e7eb",
                    padding: 16,
                    borderRadius: 8,
                    background: "#fff",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
                    cursor: "pointer",
                    transition: "transform 0.2s, box-shadow 0.2s",
                    minHeight: "320px",
                    display: "flex",
                    flexDirection: "column"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-2px)";
                    e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "0 1px 3px rgba(0,0,0,0.1)";
                  }}
                  onClick={() => setSelectedItem(item)}>
                  {imgUrl && (
                    <img src={imgUrl} alt={item.item_name}
                      style={{
                        width: "100%",
                        height: 180,
                        objectFit: "cover",
                        borderRadius: 6,
                        background: "#f3f4f6",
                        marginBottom: 12
                      }} />
                  )}
                  {item.item_name && (
                    <h3 style={{ margin: "0 0 8px 0", fontSize: 16, fontWeight: 600, lineHeight: 1.3 }}>{item.item_name}</h3>
                  )}
                  {subtitleParts.length > 0 && (
                    <p style={{ color: "#6b7280", fontSize: 13, margin: "0 0 12px 0", lineHeight: 1.4 }}>{subtitleParts.join(" • ")}</p>
                  )}
                  {/* Card details table - only show rows that exist */}
                  <div style={{ marginTop: "auto", paddingTop: 8 }}>
                    {item.amount !== undefined && item.amount !== null && item.amount !== '' && (
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                        <span style={{ color: "#6b7280" }}>Amount</span>
                        <span style={{ fontWeight: 600, color: "#059669" }}>₹{Number(item.amount).toFixed(2)}</span>
                      </div>
                    )}
                    {avgAmount !== null && (
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6, padding: "6px 8px", background: "#f0f9ff", borderRadius: 4 }}>
                        <span style={{ color: "#0369a1", fontWeight: 500 }}>Average Price in {item.area}</span>
                        <span style={{ fontWeight: 600, color: "#0369a1" }}>₹{avgAmount.toFixed(2)}</span>
                      </div>
                    )}
                    {sqft && sqft > 0 && (
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                        <span style={{ color: "#6b7280" }}>Measurement (sqft)</span>
                        <span>{sqft.toFixed(2)}</span>
                      </div>
                    )}
                    {avgPerSqft !== null && (
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                        <span style={{ color: "#6b7280" }}>Average Price/ft²</span>
                        <span>₹{avgPerSqft.toFixed(2)}</span>
                      </div>
                    )}
                    {Object.keys(attrs).length > 0 && (
                      <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid #f3f4f6" }}>
                        {Object.entries(attrs).slice(0, 3).map(([k, v]) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                            <span style={{ color: "#6b7280" }}>{k}</span>
                            <span style={{ color: "#111827", fontWeight: 500 }}>{String(v)}</span>
                          </div>
                        ))}
                        {Object.keys(attrs).length > 3 && (
                          <div style={{ fontSize: 11, color: "#9ca3af", textAlign: "center", marginTop: 4 }}>
                            +{Object.keys(attrs).length - 3} more attributes
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          /* Table View */
          <div style={{ 
            marginTop: 12, 
            background: "#fff", 
            borderRadius: 8, 
            overflow: "hidden", 
            border: "1px solid #e5e7eb",
            overflowX: "auto"
          }}>
            <div style={{ minWidth: "800px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ background: "#f9fafb", borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 600, fontSize: 14 }}>Image</th>
                    <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 600, fontSize: 14 }}>Item Name</th>
                    <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 600, fontSize: 14 }}>Room</th>
                    <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 600, fontSize: 14 }}>Project</th>
                    <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 600, fontSize: 14 }}>Area</th>
                    <th style={{ padding: "12px 16px", textAlign: "right", fontWeight: 600, fontSize: 14 }}>Amount</th>
                    <th style={{ padding: "12px 16px", textAlign: "right", fontWeight: 600, fontSize: 14 }}>Measurement (sqft)</th>
                    <th style={{ padding: "12px 16px", textAlign: "right", fontWeight: 600, fontSize: 14 }}>Avg/ft²</th>
                  </tr>
                </thead>
                <tbody>
                  {(filteredResults.length > 0 ? filteredResults : results).map((item, index) => {
                    const imgUrl = extractImageUrl(item.image);
                    const sqft = toSafeNumber(item.measurement_sqft);
                    const amountNum = toSafeNumber(item.amount);
                    const avgPerSqft = sqft && sqft > 0 && amountNum != null ? (amountNum / sqft) : null;

                    return (
                      <tr
                        key={item.item_identifier || item.id || item.item_name || index}
                        style={{
                          borderBottom: "1px solid #f3f4f6",
                          cursor: "pointer",
                          transition: "background-color 0.2s"
                        }}
                        onMouseEnter={(e) => e.target.parentElement.style.background = "#f9fafb"}
                        onMouseLeave={(e) => e.target.parentElement.style.background = "transparent"}
                        onClick={() => setSelectedItem(item)}
                      >
                        <td style={{ padding: "12px 16px" }}>
                          {imgUrl ? (
                            <img
                              src={imgUrl}
                              alt={item.item_name}
                              style={{
                                width: 40,
                                height: 40,
                                objectFit: "cover",
                                borderRadius: 4,
                                background: "#f3f4f6"
                              }}
                            />
                          ) : (
                            <div style={{
                              width: 40,
                              height: 40,
                              background: "#f3f4f6",
                              borderRadius: 4,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              color: "#9ca3af",
                              fontSize: 12
                            }}>
                              No Image
                            </div>
                          )}
                        </td>
                        <td style={{ padding: "12px 16px", fontWeight: 500 }}>{item.item_name || '-'}</td>
                        <td style={{ padding: "12px 16px" }}>{item.room_name || '-'}</td>
                        <td style={{ padding: "12px 16px" }}>{item.project_name || '-'}</td>
                        <td style={{ padding: "12px 16px" }}>{item.area || '-'}</td>
                        <td style={{ padding: "12px 16px", textAlign: "right", fontWeight: 600, color: "#059669" }}>
                          {item.amount ? `₹${item.amount}` : '-'}
                        </td>
                        <td style={{ padding: "12px 16px", textAlign: "right" }}>
                          {sqft ? sqft.toFixed(2) : '-'}
                        </td>
                        <td style={{ padding: "12px 16px", textAlign: "right" }}>
                          {avgPerSqft ? `₹${avgPerSqft.toFixed(2)}` : '-'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Slide-out details panel (left to right) */}
      {selectedItem && (
        <>
          {/* Backdrop */}
          <div
            onClick={() => setSelectedItem(null)}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.35)",
              zIndex: 40
            }}
          />
          {/* Drawer on left */}
          <div style={{
            position: "fixed",
            top: 0,
            left: 0,
            height: "100vh",
            width: "min(560px, 100vw)",
            maxWidth: "100%",
            background: "#fff",
            borderRight: "1px solid #e5e7eb",
            boxShadow: "2px 0 8px rgba(0,0,0,0.12)",
            padding: 16,
            overflowY: "auto",
            zIndex: 50,
            transition: "transform 200ms ease-out"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h3 style={{ margin: 0 }}>Item Details</h3>
              <button
                onClick={() => setSelectedItem(null)}
                style={{
                  border: "none",
                  background: "#ef4444",
                  color: "#fff",
                  borderRadius: 6,
                  padding: "6px 10px",
                  cursor: "pointer"
                }}
              >
                Close
              </button>
            </div>
            <div style={{ marginTop: 12 }}>
              {(() => {
                const imgUrl = extractImageUrl(selectedItem.image);
                return imgUrl ? (
                  <img src={imgUrl} alt={selectedItem.item_name}
                    style={{ width: "100%", height: 180, objectFit: "cover", borderRadius: 8 }} />
                ) : null;
              })()}
              <div style={{ marginTop: 10 }}>
                {selectedItem.item_name && (
                  <div style={{ fontSize: 18, fontWeight: 600 }}>{selectedItem.item_name}</div>
                )}
                <div style={{ color: "#6b7280", marginTop: 4 }}>
                  {[selectedItem.room_name, selectedItem.project_name, selectedItem.area].filter(Boolean).join(" • ")}
                </div>
                {selectedItem.amount !== undefined && selectedItem.amount !== null && (
                  <div style={{ marginTop: 8 }}>
                    <strong>Amount:</strong> {selectedItem.amount}
                  </div>
                )}
              </div>

              {/* Area-wise amount stats table */}
              <div style={{ marginTop: 16 }}>
                <strong>Area-wise Amounts (grouped by city and measurement)</strong>
                {detailsLoading && (
                  <div style={{ color: "#6b7280", marginTop: 6 }}>Loading…</div>
                )}
                {detailsError && (
                  <div style={{
                    background: "#fef2f2",
                    color: "#b91c1c",
                    border: "1px solid #fecaca",
                    padding: 8,
                    borderRadius: 6,
                    marginTop: 6
                  }}>{detailsError}</div>
                )}
                {!detailsLoading && !detailsError && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr 1fr 0.8fr 0.8fr 0.8fr 0.6fr",
                      gap: 6,
                      fontSize: 13,
                      color: "#111827"
                    }}>
                      <div style={{ fontWeight: 600 }}>Area</div>
                      <div style={{ fontWeight: 600 }}>Measurement (sqft)</div>
                      <div style={{ fontWeight: 600 }}>Measurement (2x2)</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Min Amount</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Max Amount</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Avg Amount</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Count</div>
                      {areaAmountStats.map((row, idx) => (
                        <>
                          <div key={`area-${idx}`} style={{ textTransform: "capitalize" }}>{row.area}</div>
                          <div key={`measurement-sqft-${idx}`}>
                            {row.measurement_sqft !== null && row.measurement_sqft !== undefined 
                              ? row.measurement_sqft.toFixed(2)
                              : 'N/A'
                            }
                          </div>
                          <div key={`measurement-2x2-${idx}`}>
                            {row.measurement_2x2 || 'N/A'}
                          </div>
                          <div key={`min-${idx}`} style={{ textAlign: "right" }}>₹{row.min.toFixed(2)}</div>
                          <div key={`max-${idx}`} style={{ textAlign: "right" }}>₹{row.max.toFixed(2)}</div>
                          <div key={`avg-${idx}`} style={{ textAlign: "right" }}>₹{row.avg.toFixed(2)}</div>
                          <div key={`count-${idx}`} style={{ textAlign: "right" }}>{row.count}</div>
                        </>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
