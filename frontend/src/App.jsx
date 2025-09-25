import { useEffect, useRef, useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(12);
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
    measurementMin: "",
    measurementMax: "",
    amountMin: "",
    amountMax: ""
  });
  const [showSidebar, setShowSidebar] = useState(false);
  const [filteredResults, setFilteredResults] = useState([]);
  const [isFiltering, setIsFiltering] = useState(false);
  const [viewMode, setViewMode] = useState("cards"); // "cards" or "table"
  const [useNLP, setUseNLP] = useState(true);
  const [extractedFilters, setExtractedFilters] = useState({});

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
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setQuery(transcript);
      };
      recognition.onend = () => setIsListening(false);

      recognitionRef.current = recognition;
    }
  }, []);

  // Perform search using NLP endpoint
  const handleSearch = async (newPage = page) => {
    setIsSearching(true);
    setError("");
    setResults([]);
    setAreaAmountStats([]);
    setAreaStatsAvg(null);
    try {
      const res = await axios.post(`${BACKEND_URL}/search/nlp`, {
        query,
        use_nlp: true
      });

      const raw = res?.data?.results;
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
      setPage(newPage);

      // /search/nlp does not return area_stats
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
        const res = await axios.post(`${BACKEND_URL}/search/filtered-stats`, {
          item_name: selectedItem.item_name,
          city: filters.city || null,
          measurement_min: filters.measurementMin ? parseFloat(filters.measurementMin) : null,
          measurement_max: filters.measurementMax ? parseFloat(filters.measurementMax) : null,
          amount_min: filters.amountMin ? parseFloat(filters.amountMin) : null,
          amount_max: filters.amountMax ? parseFloat(filters.amountMax) : null
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

  // Re-run search if page changes and query exists
  useEffect(() => {
    if (query.trim()) handleSearch(page);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

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

  const handleVoiceToggle = () => {
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const applyFilters = async () => {
    setIsFiltering(true);
    setError("");
    try {
      const res = await axios.post(`${BACKEND_URL}/search/filtered-stats/only`, {
        item_name: filters.itemName || null,
        city: filters.city || null,
        measurement_min: filters.measurementMin ? parseFloat(filters.measurementMin) : null,
        measurement_max: filters.measurementMax ? parseFloat(filters.measurementMax) : null,
        amount_min: filters.amountMin ? parseFloat(filters.amountMin) : null,
        amount_max: filters.amountMax ? parseFloat(filters.amountMax) : null
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
      measurementMin: "",
      measurementMax: "",
      amountMin: "",
      amountMax: ""
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
        zIndex: 10
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 28, height: 28, background: "#111827", borderRadius: 6 }} />
          <h2 style={{ margin: 0 }}>Estimator Item Search</h2>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setUseNLP(!useNLP)}
            style={{ ...buttonStyle, background: useNLP ? "#7c3aed" : "#6b7280" }}
          >
            {useNLP ? "NLP ON" : "NLP OFF"}
          </button>
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            style={{ ...buttonStyle, background: showSidebar ? "#dc2626" : "#111827" }}
          >
            {showSidebar ? "Hide Filters" : "Show Filters"}
          </button>
          <button
            onClick={() => setViewMode(viewMode === "cards" ? "table" : "cards")}
            style={{ ...buttonStyle, background: "#059669" }}
          >
            {viewMode === "cards" ? "Table View" : "Card View"}
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
          marginTop: "73px"
        }}>
          <h3 style={{ margin: "0 0 16px 0" }}>Filters</h3>

          {/* NLP Examples */}
          {useNLP && (
            <div style={{ marginBottom: 16, padding: 12, background: "#f0f9ff", borderRadius: 6, border: "1px solid #0ea5e9" }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#0369a1", marginBottom: 8 }}>
                💡 NLP Examples:
              </div>
              <div style={{ fontSize: 11, color: "#0369a1", lineHeight: 1.4 }}>
                • "TV Wall Unit in Bangalore"<br />
                • "between 50-100 sqft"<br />
                • "under ₹50000"<br />
                • "Console Table from Mumbai"
              </div>
            </div>
          )}

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

          {/* Measurement Range */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>Measurement (sqft)</label>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                type="number"
                placeholder={useNLP ? "Min" : "Min"}
                value={filters.measurementMin}
                onChange={(e) => handleFilterChange('measurementMin', e.target.value)}
                style={{
                  flex: 1,
                  padding: "8px 12px",
                  border: "1px solid #e5e7eb",
                  borderRadius: 6,
                  fontSize: 14
                }}
              />
              <input
                type="number"
                placeholder={useNLP ? "Max" : "Max"}
                value={filters.measurementMax}
                onChange={(e) => handleFilterChange('measurementMax', e.target.value)}
                style={{
                  flex: 1,
                  padding: "8px 12px",
                  border: "1px solid #e5e7eb",
                  borderRadius: 6,
                  fontSize: 14
                }}
              />
            </div>
            {useNLP && (
              <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
                Try: "between 50-100 sqft" or "over 200 sqft"
              </div>
            )}
          </div>

          {/* Amount Range */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>Amount</label>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                type="number"
                placeholder={useNLP ? "Min" : "Min"}
                value={filters.amountMin}
                onChange={(e) => handleFilterChange('amountMin', e.target.value)}
                style={{
                  flex: 1,
                  padding: "8px 12px",
                  border: "1px solid #e5e7eb",
                  borderRadius: 6,
                  fontSize: 14
                }}
              />
              <input
                type="number"
                placeholder={useNLP ? "Max" : "Max"}
                value={filters.amountMax}
                onChange={(e) => handleFilterChange('amountMax', e.target.value)}
                style={{
                  flex: 1,
                  padding: "8px 12px",
                  border: "1px solid #e5e7eb",
                  borderRadius: 6,
                  fontSize: 14
                }}
              />
            </div>
            {useNLP && (
              <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
                Try: "under ₹50000" or "between ₹10000-50000"
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

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "0 16px", marginLeft: showSidebar ? "336px" : "auto" }}>
        {/* Search bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          border: "1px solid #e5e7eb",
          borderRadius: 999,
          padding: "8px 12px",
          margin: "16px auto",
          maxWidth: 640,
          gap: 8
        }}>
          <input
            type="text"
            placeholder={useNLP ? "Try: 'TV Wall Unit in Bangalore between 50-100 sqft under ₹50000'" : "Search items with natural language..."}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(1); }}
            style={{ flex: 1, border: "none", outline: "none", background: "transparent" }}
          />
          {recognitionRef.current && (
            <button
              onClick={handleVoiceToggle}
              title="Voice search"
              style={{
                border: "none",
                background: isListening ? "#dc2626" : "#111827",
                color: "#fff",
                borderRadius: 999,
                padding: "6px 10px",
                cursor: "pointer"
              }}
            >
              {isListening ? "Stop" : "Mic"}
            </button>
          )}
          <button onClick={() => handleSearch(1)} style={{ ...buttonStyle, marginLeft: 8 }} disabled={isSearching}>
            {isSearching ? "Searching..." : "Search"}
          </button>
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

        {/* Extracted Filters Display */}
        {useNLP && Object.keys(extractedFilters).length > 0 && (
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

        {/* Pagination Info */}
        {results.length > 0 && (
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 8px" }}>
            <div style={{ color: "#6b7280" }}>{`Page ${page}`}</div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={() => page > 1 && setPage(page - 1)}
                style={buttonStyle}
                disabled={page === 1 || isSearching}
              >
                Prev
              </button>
              <button
                onClick={() => setPage(page + 1)}
                style={buttonStyle}
                disabled={isSearching || results.length < pageSize}
              >
                Next
              </button>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isSearching && !error && results.length === 0 && (
          <div style={{ textAlign: "center", color: "#6b7280", marginTop: 24 }}>
            No results yet. Try a search.
          </div>
        )}

        {/* Results Display */}
        {viewMode === "cards" ? (
          /* Card View */
          <div style={{
            marginTop: 12,
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: 12
          }}>
            {(filteredResults.length > 0 ? filteredResults : results).map((item) => {
              const key = item.item_identifier || item.id || item.item_name;
              const imgUrl = extractImageUrl(item.image);
              const sqft = toSafeNumber(item.measurement_sqft);
              const amountNum = toSafeNumber(item.amount);
              const avgPerSqft = sqft && sqft > 0 && amountNum != null ? (amountNum / sqft) : null;
              const subtitleParts = [item.room_name, item.project_name, item.area].filter(Boolean);

              const attrs = item.attributes_parsed && typeof item.attributes_parsed === 'object' ? item.attributes_parsed : {};

              return (
                <div key={key}
                  style={{
                    border: "1px solid #e5e7eb",
                    padding: 12,
                    borderRadius: 8,
                    background: "#fff",
                    boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
                    cursor: "pointer"
                  }}
                  onClick={() => setSelectedItem(item)}>
                  {imgUrl && (
                    <img src={imgUrl} alt={item.item_name}
                      style={{
                        width: "100%",
                        height: 160,
                        objectFit: "cover",
                        borderRadius: 6,
                        background: "#f3f4f6"
                      }} />
                  )}
                  {item.item_name && (
                    <h3 style={{ margin: "10px 0 4px", fontSize: 16 }}>{item.item_name}</h3>
                  )}
                  {subtitleParts.length > 0 && (
                    <p style={{ color: "#6b7280", fontSize: 13 }}>{subtitleParts.join(" • ")}</p>
                  )}
                  {/* Card details table - only show rows that exist */}
                  <div style={{ marginTop: 6 }}>
                    {item.amount !== undefined && item.amount !== null && item.amount !== '' && (
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                        <span style={{ color: "#6b7280" }}>Amount</span>
                        <span style={{ fontWeight: 600, color: "#059669" }}>₹{item.amount}</span>
                      </div>
                    )}
                    {sqft && sqft > 0 && (
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginTop: 4 }}>
                        <span style={{ color: "#6b7280" }}>Measurement (sqft)</span>
                        <span>{sqft.toFixed(2)}</span>
                      </div>
                    )}
                    {avgPerSqft !== null && (
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginTop: 4 }}>
                        <span style={{ color: "#6b7280" }}>Avg/ft²</span>
                        <span>₹{avgPerSqft.toFixed(2)}</span>
                      </div>
                    )}
                    {Object.keys(attrs).length > 0 && (
                      <div style={{ marginTop: 6 }}>
                        {Object.entries(attrs).map(([k, v]) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginTop: 2 }}>
                            <span style={{ color: "#6b7280" }}>{k}</span>
                            <span style={{ color: "#111827" }}>{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          /* Table View */
          <div style={{ marginTop: 12, background: "#fff", borderRadius: 8, overflow: "hidden", border: "1px solid #e5e7eb" }}>
            <div style={{ overflowX: "auto" }}>
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
                      gridTemplateColumns: "1fr 1fr 0.8fr 0.8fr 0.8fr 0.6fr",
                      gap: 6,
                      fontSize: 13,
                      color: "#111827"
                    }}>
                      <div style={{ fontWeight: 600 }}>Area</div>
                      <div style={{ fontWeight: 600 }}>Measurement Group</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Min</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Max</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Avg</div>
                      <div style={{ fontWeight: 600, textAlign: "right" }}>Count</div>
                      {areaAmountStats.map((row, idx) => (
                        <>
                          <div key={`area-${idx}`} style={{ textTransform: "capitalize" }}>{row.area}</div>
                          <div key={`measurement-${idx}`}>{row.measurement_group || 'N/A'}</div>
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
