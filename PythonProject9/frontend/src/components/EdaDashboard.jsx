import { useState, useEffect } from 'react'
import axios from 'axios'
import Card from './Card'
import './EdaDashboard.css'

export default function EdaDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`http://127.0.0.1:5000/api/eda?t=${new Date().getTime()}`)
      .then(res => {
        setData(res.data.data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.response?.data?.error || err.message)
        setLoading(false)
      })
  }, [])

  if (loading) return (
    <div className="loading-container">
      <div className="spinner"></div>
      <p>Generating Exploratory Data Analysis... This may take a moment.</p>
    </div>
  )
  if (error) return <div className="error-box">Error loading EDA: {error}</div>
  if (!data) return null

  return (
    <div className="animate-fade-in eda-page">
      <div className="eda-header">
        <h2>Exploratory Data Analysis Dashboard</h2>
        <div className="badge-primary">15 Analytical Tasks</div>
      </div>
      
      <p className="eda-subtitle">
        Visualizing and analyzing the US Accidents dataset structure, severity distributions, temporal trends, and environmental factors.
      </p>

      {/* KPI Row */}
      <div className="kpi-grid">
        <Card className="kpi-card">
          <div className="kpi-label">Analyzed Rows</div>
          <div className="kpi-value">{data.n_rows.toLocaleString()}</div>
        </Card>
        <Card className="kpi-card">
          <div className="kpi-label">Features</div>
          <div className="kpi-value">{data.n_cols}</div>
        </Card>
        <Card className="kpi-card">
          <div className="kpi-label">Duplicates</div>
          <div className="kpi-value">{data.duplicate_count.toLocaleString()}</div>
        </Card>
      </div>

      {/* Charts Grid */}
      <div className="charts-masonry">
        {data.charts && data.charts.map((chart, idx) => (
          <Card key={idx} title={chart.title} className="chart-card">
            <div className="chart-image-container">
              <img 
                src={`http://127.0.0.1:5000/api/charts/${chart.filename}?t=${new Date().getTime()}`} 
                alt={chart.title} 
                className="chart-image"
                loading="lazy"
              />
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
