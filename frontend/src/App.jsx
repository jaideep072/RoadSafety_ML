import { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import DataSummary from './components/DataSummary'
import EdaDashboard from './components/EdaDashboard'
import './App.css'

function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <header className="topbar">
          <h1>Project Dashboard</h1>
          <div className="user-profile">
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>System Version 2.0</span>
          </div>
        </header>
        <div className="page-container">
          <Routes>
            <Route path="/" element={<Navigate to="/data-loading" replace />} />
            <Route path="/data-loading" element={<DataSummary />} />
            <Route path="/eda" element={<EdaDashboard />} />
            <Route path="*" element={<Navigate to="/data-loading" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
