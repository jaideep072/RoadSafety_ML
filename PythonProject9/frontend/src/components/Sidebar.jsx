import { NavLink } from 'react-router-dom'
import { Database, BarChart2, Settings, Archive, FileText } from 'lucide-react'
import './Sidebar.css'

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <Archive className="logo-icon" size={22} />
        <span>Road Safety Analysis</span>
      </div>
      
      <div className="sidebar-menu">
        <p className="menu-label">Data Modules</p>
        <NavLink to="/data-loading" className={({isActive}) => isActive ? "menu-item active" : "menu-item"}>
          <Database size={18} />
          <span>Dataset Summary</span>
        </NavLink>
        <NavLink to="/eda" className={({isActive}) => isActive ? "menu-item active" : "menu-item"}>
          <BarChart2 size={18} />
          <span>Exploratory Analysis</span>
        </NavLink>
        
        <p className="menu-label" style={{ marginTop: '2rem' }}>Modeling</p>
        <div className="menu-item disabled">
          <Settings size={18} />
          <span>Preprocessing</span>
          <span className="badge">Soon</span>
        </div>
        <div className="menu-item disabled">
          <FileText size={18} />
          <span>Model Training</span>
          <span className="badge">Soon</span>
        </div>
      </div>
    </nav>
  )
}
