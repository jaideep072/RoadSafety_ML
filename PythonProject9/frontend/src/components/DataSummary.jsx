import { useState, useEffect } from 'react'
import axios from 'axios'
import Card from './Card'
import './DataSummary.css'

export default function DataSummary() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/data-summary')
      .then(res => {
        setData(res.data.data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.response?.data?.error || err.message)
        setLoading(false)
      })
  }, [])

  if (loading) return <div className="spinner"></div>
  if (error) return <div className="error-box">Error loading data: {error}</div>
  if (!data) return null

  return (
    <div className="animate-fade-in data-summary-page">
      <h2 style={{ marginBottom: '2rem' }}>Data Loading & Overview</h2>
      
      <div className="metrics-grid">
        <Card className="metric-card">
          <div className="metric-icon rows"></div>
          <div className="metric-content">
            <h4>Total Rows</h4>
            <div className="metric-value">{data.n_rows.toLocaleString()}</div>
          </div>
        </Card>
        
        <Card className="metric-card">
          <div className="metric-icon cols"></div>
          <div className="metric-content">
            <h4>Total Columns</h4>
            <div className="metric-value">{data.n_cols}</div>
          </div>
        </Card>
      </div>

      <Card title="Data Schema & Missing Values">
        <div className="table-responsive">
          <table className="modern-table">
            <thead>
              <tr>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Missing Values</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data.columns.map(col => {
                const missing = data.missing_counts[col]
                const missingPct = ((missing / data.n_rows) * 100).toFixed(1)
                const statusClass = missingPct > 20 ? 'status-danger' : missingPct > 0 ? 'status-warning' : 'status-success'
                
                return (
                  <tr key={col}>
                    <td className="font-medium">{col}</td>
                    <td><span className="dtype-badge">{data.dtypes[col]}</span></td>
                    <td>{missing.toLocaleString()} ({missingPct}%)</td>
                    <td><span className={`status-dot ${statusClass}`}></span></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
      
      <Card title="Dataset Preview (Sample)">
        <div className="table-responsive">
          <table className="modern-table">
            <thead>
              <tr>
                {data.columns.map(col => <th key={col}>{col}</th>)}
              </tr>
            </thead>
            <tbody>
              {data.preview.map((row, idx) => (
                <tr key={idx}>
                  {data.columns.map(col => (
                    <td key={col} className="truncate-cell" title={row[col]}>{row[col] !== null ? String(row[col]) : 'NaN'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
