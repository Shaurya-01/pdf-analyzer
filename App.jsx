import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    setError("");
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file.");
      return;
    }
    setLoading(true);
    setResult(null);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/analyze-file", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        setError(err.detail || "Upload failed.");
        setLoading(false);
        return;
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Network error. Please try again.");
    }
    setLoading(false);
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Utility: Truncate filename if too long (middle)
  const getDisplayFileName = (name) => {
    if (!name) return "";
    if (name.length <= 32) return name;
    const ext = name.lastIndexOf('.') !== -1 ? name.slice(name.lastIndexOf('.')) : '';
    const base = name.slice(0, name.length - ext.length);
    return base.slice(0, 16) + "…" + base.slice(-8) + ext;
  };

  return (
    <div className="bw-bg">
      <main className="bw-main">
        <div className="bw-header">
          <span className="bw-dot" />
          <h1 className="bw-title">File Analyzer</h1>
        </div>
        <form className="bw-form" autoComplete="off" onSubmit={handleAnalyze}>
          <label
            className={`bw-file-drop${file ? " bw-file-drop-selected" : ""}`}
            tabIndex={0}
            onKeyDown={e => { if (e.key === "Enter") fileInputRef.current.click(); }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.png,.jpg,.jpeg"
              onChange={handleFileChange}
              disabled={loading}
              aria-label="Choose file"
            />
            <span className="bw-file-drop-text" title={file ? file.name : undefined}>
              {file ? (
                <>
                  <span className="bw-file-icon">&#128196;</span>
                  <span className="bw-file-name">{getDisplayFileName(file.name)}</span>
                </>
              ) : (
                <>
                  <span className="bw-file-icon">&#8682;</span>
                  Click or drag file here
                </>
              )}
            </span>
            {loading && <span className="bw-loader" />}
          </label>
          <div className="bw-btn-group">
            <button
              type="submit"
              className="bw-analyze-btn"
              disabled={loading || !file}
            >
              {loading ? (
                <span className="bw-loader" />
              ) : (
                <>
                  <svg width="20" height="20" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24" style={{marginRight: "8px"}}>
                    <polyline points="12 19 12 5"></polyline>
                    <polyline points="5 12 12 19 19 12"></polyline>
                  </svg>
                  <span className="bw-analyze-text">Analyze</span>
                </>
              )}
            </button>
            <button
              type="button"
              className="bw-reset-btn"
              onClick={handleReset}
              disabled={loading && !file}
            >
              <svg width="18" height="18" fill="none" stroke="#181818" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <polyline points="1 4 1 10 7 10"></polyline>
                <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
              </svg>
              Reset
            </button>
          </div>
        </form>
        {error && <div className="bw-error">{error}</div>}
        {result && (
          <div className="bw-result-continuous">
            <span>
              <b>Type:</b> <span className="bw-result-value">{result.document_type}</span>
            </span>
            <span>
              <b>Summary:</b> <span className="bw-result-value">{result.summary}</span>
            </span>
            <span>
              <b>Confidence:</b> <span className="bw-result-value">{result.confidence}</span>
            </span>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;