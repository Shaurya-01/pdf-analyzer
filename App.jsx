import React, { useState, useCallback } from 'react';
import './App.css';

const App = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');

  const API_BASE_URL = 'http://localhost:8000';

  // Supported file types
  const SUPPORTED_TYPES = ['.pdf', '.docx', '.pptx', '.png', '.jpg', '.jpeg'];
  const MAX_FILES = 10;
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

  const validateFile = (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_TYPES.includes(ext)) {
      return `File type ${ext} not supported`;
    }
    if (file.size > MAX_FILE_SIZE) {
      return 'File size too large (max 10MB)';
    }
    return null;
  };

  const handleFiles = (fileList) => {
    const newFiles = Array.from(fileList);
    if (files.length + newFiles.length > MAX_FILES) {
      setError(`Maximum ${MAX_FILES} files allowed`);
      return;
    }

    const validFiles = [];
    const errors = [];

    newFiles.forEach(file => {
      const error = validateFile(file);
      if (error) {
        errors.push(`${file.name}: ${error}`);
      } else {
        validFiles.push({
          file,
          id: Date.now() + Math.random(),
          name: file.name,
          size: file.size,
          type: file.type
        });
      }
    });

    if (errors.length > 0) {
      setError(errors.join(', '));
    } else {
      setError('');
    }

    setFiles(prev => [...prev, ...validFiles]);
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragOut = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, [files]);

  const removeFile = (id) => {
    setFiles(prev => prev.filter(f => f.id !== id));
    setError('');
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const analyzeFiles = async () => {
    if (files.length === 0) return;

    setUploading(true);
    setError('');
    setResults(null);

    try {
      const formData = new FormData();
      files.forEach(({ file }) => {
        formData.append('files', file);
      });

      const endpoint = files.length === 1 ? '/analyze-file' : '/analyze-multiple-files';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        body: files.length === 1 ? (() => {
          const singleFormData = new FormData();
          singleFormData.append('file', files[0].file);
          return singleFormData;
        })() : formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis');
    } finally {
      setUploading(false);
    }
  };

  const resetApp = () => {
    setFiles([]);
    setResults(null);
    setError('');
  };

  return (
    <div className="app-root">
      <div className="professional-bg"></div>
      <div className="main-content">
        <header className="header">
          <h1>
            <span className="brand-gradient">Document Analyzer</span>
          </h1>
          <p>Upload and analyze your documents securely and intelligently</p>
        </header>

        {!results ? (
          <div className="upload-section">
            <div
              className={`drop-zone ${dragActive ? 'drag-active' : ''} ${files.length > 0 ? 'has-files' : ''}`}
              onDragEnter={handleDragIn}
              onDragLeave={handleDragOut}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="drop-zone-content">
                <div className="upload-icon">
                  <svg width="56" height="56" viewBox="0 0 24 24" fill="none">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <line x1="12" y1="18" x2="12" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <polyline points="9,15 12,12 15,15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <h3>Drag &amp; drop or&nbsp;
                  <label className="browse-btn">
                    browse
                    <input
                      type="file"
                      multiple
                      accept=".pdf,.docx,.pptx,.png,.jpg,.jpeg"
                      onChange={(e) => handleFiles(e.target.files)}
                      className="file-input"
                    />
                  </label>
                </h3>
                <p>PDF, DOCX, PPTX, PNG, JPG, JPEG &middot; Max 10MB each</p>
              </div>
            </div>

            {error && (
              <div className="error-message">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                  <line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" strokeWidth="2"/>
                  <line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" strokeWidth="2"/>
                </svg>
                {error}
              </div>
            )}

            {files.length > 0 && (
              <div className="file-list">
                <h3>Selected Files ({files.length}/{MAX_FILES})</h3>
                {files.map(file => (
                  <div key={file.id} className="file-item">
                    <div className="file-info">
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{formatFileSize(file.size)}</span>
                    </div>
                    <button
                      onClick={() => removeFile(file.id)}
                      className="remove-btn"
                      title="Remove file"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2"/>
                        <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2"/>
                      </svg>
                    </button>
                  </div>
                ))}
                
                <div className="action-buttons">
                  <button
                    onClick={analyzeFiles}
                    disabled={uploading || files.length === 0}
                    className="analyze-btn primary"
                  >
                    {uploading ? 'Analyzing...' : 'Analyze'}
                  </button>
                  <button onClick={resetApp} className="clear-btn secondary">
                    Clear
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="results-section">
            <div className="results-header">
              <h2>Results</h2>
              <button onClick={resetApp} className="new-analysis-btn">
                New Analysis
              </button>
            </div>

            {results.analyses ? (
              <div className="results-summary">
                <div className="summary-stats">
                  <div className="stat">
                    <span className="stat-number">{results.total_files}</span>
                    <span className="stat-label">Total</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">{results.successful_analyses}</span>
                    <span className="stat-label">Analyzed</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">{results.failed_files.length}</span>
                    <span className="stat-label">Failed</span>
                  </div>
                </div>

                {results.failed_files.length > 0 && (
                  <div className="failed-files">
                    <h4>Failed Files:</h4>
                    {results.failed_files.map((error, index) => (
                      <div key={index} className="failed-file">{error}</div>
                    ))}
                  </div>
                )}

                <div className="analysis-results">
                  {results.analyses.map((analysis, index) => (
                    <div key={index} className="result-card">
                      <div className="result-header">
                        <h4>{analysis.filename}</h4>
                        <span className={`confidence ${analysis.confidence.toLowerCase()}`}>
                          {analysis.confidence} Confidence
                        </span>
                      </div>
                      <div className="result-body">
                        <div className="document-type">
                          <strong>Type:</strong> {analysis.document_type}
                        </div>
                        <div className="summary">
                          <strong>Summary:</strong> {analysis.summary}
                        </div>
                        <div className="text-length">
                          <strong>Text Length:</strong> {analysis.extracted_text_length.toLocaleString()} chars
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="single-result">
                <div className="result-card">
                  <div className="result-header">
                    <h4>{results.filename}</h4>
                    <span className={`confidence ${results.confidence.toLowerCase()}`}>
                      {results.confidence} Confidence
                    </span>
                  </div>
                  <div className="result-body">
                    <div className="document-type">
                      <strong>Type:</strong> {results.document_type}
                    </div>
                    <div className="summary">
                      <strong>Summary:</strong> {results.summary}
                    </div>
                    <div className="text-length">
                      <strong>Text Length:</strong> {results.extracted_text_length.toLocaleString()} chars
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
