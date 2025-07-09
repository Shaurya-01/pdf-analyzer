import React, { useState, useCallback } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:3000';

const SUPPORTED_TYPES = ['.pdf', '.docx', '.pptx', '.png', '.jpg', '.jpeg'];
const MAX_FILES = 10;
const MAX_FILE_SIZE = 10 * 1024 * 1024;

const App = () => {
  const [mode, setMode] = useState("analysis");

  // Document Analysis state
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');

  // Resume Scoring state
  const [jdText, setJdText] = useState('');
  const [jdFile, setJdFile] = useState(null);
  const [resumeFiles, setResumeFiles] = useState([]);
  const [scoringResult, setScoringResult] = useState(null);
  const [scoringError, setScoringError] = useState('');
  const [scoringUploading, setScoringUploading] = useState(false);

  // --- Utility functions ---
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

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // --- Document Analysis handlers ---
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

  // --- Resume Scoring handlers ---
  const handleJdFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setJdFile(file);
      setJdText('');
    }
  };

  const handleResumeFilesChange = (e) => {
    const newResumes = Array.from(e.target.files);
    const validFiles = [];
    const errors = [];
    newResumes.forEach(file => {
      const err = validateFile(file);
      if (err) {
        errors.push(`${file.name}: ${err}`);
      } else {
        validFiles.push(file);
      }
    });
    if (errors.length > 0) setScoringError(errors.join(', '));
    else setScoringError('');
    setResumeFiles(prev => [...prev, ...validFiles]);
  };

  const removeResumeFile = (idx) => {
    setResumeFiles(prev => prev.filter((_, i) => i !== idx));
    setScoringError('');
  };

  // --- UI mode switching/clearing ---
  const switchToAnalysis = () => {
    setMode("analysis");
    setFiles([]);
    setUploading(false);
    setResults(null);
    setDragActive(false);
    setError('');
  };

  const switchToScoring = () => {
    setMode("scoring");
    setJdText('');
    setJdFile(null);
    setResumeFiles([]);
    setScoringResult(null);
    setScoringError('');
    setScoringUploading(false);
  };

  // --- Resume scoring logic ---
  const scoreResumes = async () => {
    if (!jdFile && !jdText) {
      setScoringError('Please provide a job description (file or text)');
      return;
    }
    if (resumeFiles.length === 0) {
      setScoringError('Please upload at least one resume');
      return;
    }
    setScoringUploading(true);
    setScoringResult(null);
    setScoringError('');
    try {
      const formData = new FormData();
      if (jdFile) formData.append('jd_file', jdFile);
      if (jdText && !jdFile) formData.append('jd_text', jdText);
      resumeFiles.forEach(f => formData.append('resumes', f));
      const response = await fetch(`${API_BASE_URL}/score-resumes`, {
        method: 'POST',
        body: formData
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Scoring failed');
      }
      const data = await response.json();
      setScoringResult(data);
    } catch (err) {
      setScoringError(err.message || 'An error occurred during resume scoring');
    } finally {
      setScoringUploading(false);
    }
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

        <div style={{ display: "flex", gap: "1.5rem", marginBottom: 24 }}>
          <button
            className={mode === "analysis" ? "tab-btn active" : "tab-btn"}
            onClick={switchToAnalysis}
          >
            Document Analysis
          </button>
          <button
            className={mode === "scoring" ? "tab-btn active" : "tab-btn"}
            onClick={switchToScoring}
          >
            Resume Scoring
          </button>
        </div>

        {/* Resume Scoring UI */}
        {mode === "scoring" && (
          <div className="resume-scoring-section">
            <h2>Resume Scoring</h2>
            <div className="scoring-card">
              <div className="jd-section">
                <h3>Job Description</h3>
                <div>
                  <label className="browse-btn">
                    Upload JD File
                    <input
                      type="file"
                      accept=".pdf,.docx,.pptx,.png,.jpg,.jpeg"
                      onChange={handleJdFileChange}
                      style={{ display: 'none' }}
                    />
                  </label>
                  <span style={{ marginLeft: 10, color: "#555" }}>
                    {jdFile && jdFile.name}
                  </span>
                  <span style={{ margin: "0 10px" }}>|</span>
                  <label>
                    Or paste JD text:
                    <textarea
                      rows={4}
                      value={jdText}
                      disabled={!!jdFile}
                      onChange={e => setJdText(e.target.value)}
                      placeholder="Paste job description here"
                      style={{ width: "100%", marginTop: 5 }}
                    />
                  </label>
                </div>
              </div>
              <div className="resume-upload-section" style={{ marginTop: 16 }}>
                <h3>Resumes</h3>
                <label className="browse-btn">
                  Upload Resumes
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.docx,.pptx,.png,.jpg,.jpeg"
                    onChange={handleResumeFilesChange}
                    style={{ display: 'none' }}
                  />
                </label>
                <div>
                  {resumeFiles.length > 0 && (
                    <div className="file-list">
                      {resumeFiles.map((f, idx) => (
                        <div key={idx} className="file-item">
                          <span className="file-name">{f.name}</span>
                          <span className="file-size">{formatFileSize(f.size)}</span>
                          <button
                            onClick={() => removeResumeFile(idx)}
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
                    </div>
                  )}
                </div>
              </div>
              <div className="action-buttons" style={{ marginTop: 24 }}>
                <button
                  onClick={scoreResumes}
                  disabled={scoringUploading || (!jdFile && !jdText) || resumeFiles.length === 0}
                  className="analyze-btn primary"
                >
                  {scoringUploading ? 'Scoring...' : 'Score Resumes'}
                </button>
                <button onClick={switchToScoring} className="clear-btn secondary">
                  Clear
                </button>
              </div>
              {scoringError && (
                <div className="error-message" style={{ marginTop: 12 }}>
                  {scoringError}
                </div>
              )}
            </div>
            {scoringResult && (
              <div className="results-section" style={{ marginTop: 32 }}>
                <h3>Scoring Results</h3>
                <div className="jd-preview">
                  <strong>Job Description Preview:</strong>
                  <div className="jd-text">{scoringResult.job_description.length > 600 ? scoringResult.job_description.slice(0,600) + "..." : scoringResult.job_description}</div>
                </div>
                {scoringResult.failed_files && scoringResult.failed_files.length > 0 && (
                  <div className="failed-files" style={{ marginTop: 12 }}>
                    <strong>Failed Resumes:</strong>
                    <ul>
                      {scoringResult.failed_files.map((err, idx) => (
                        <li key={idx}>{err}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="resume-score-results">
                  {scoringResult.results.map((res, idx) => (
                    <div key={idx} className="result-card" style={{ marginBottom: 18 }}>
                      <div className="result-header">
                        <h4>{res.filename}</h4>
                        <span className="score-badge">Total Score: {res.score}/100</span>
                        <span className={res.qualification_status === "Qualified" ? "confidence high" : "confidence low"}>
                          {res.qualification_status}
                        </span>
                      </div>
                      <div className="result-body">
                        <div>
                          <strong>Experience Score:</strong> {res.experience_score}/50
                        </div>
                        <div>
                          <strong>Skills Score:</strong> {res.skills_score}/50
                        </div>
                        <div className="summary"><strong>Summary:</strong> {res.summary}</div>
                        <div className="strengths"><strong>Strengths:</strong>
                          <ul>
                            {res.strengths.split('-').filter(Boolean).map((s, i) => (
                              <li key={i}>{s.trim()}</li>
                            ))}
                          </ul>
                        </div>
                        <div className="weaknesses"><strong>Weaknesses:</strong>
                          <ul>
                            {res.weaknesses.split('-').filter(Boolean).map((s, i) => (
                              <li key={i}>{s.trim()}</li>
                            ))}
                          </ul>
                        </div>
                        <div className="skills-matched"><strong>Skills Matched:</strong> {res.skills_matched || "None"}</div>
                        <div className="skills-partial"><strong>Partial/Related Skills:</strong> {res.skills_partial || "None"}</div>
                        <div className="skills-missing"><strong>Skills Missing:</strong> {res.skills_missing || "None"}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Document Analysis UI */}
        {mode === "analysis" && (
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
                  <button onClick={switchToAnalysis} className="clear-btn secondary">
                    Clear
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results for Document Analysis */}
        {mode === "analysis" && results && (
          <div className="results-section">
            <div className="results-header">
              <h2>Results</h2>
              <button onClick={switchToAnalysis} className="new-analysis-btn">
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
