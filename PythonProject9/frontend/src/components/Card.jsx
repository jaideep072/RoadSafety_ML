export default function Card({ title, children, className = '' }) {
  return (
    <div className={`glass-panel p-6 ${className}`} style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
      {title && <h3 style={{ marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>{title}</h3>}
      <div>{children}</div>
    </div>
  )
}
