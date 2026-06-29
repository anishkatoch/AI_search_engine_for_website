import { useState, useEffect, useRef, useCallback } from 'react'

const SUGGESTIONS = [
  'wireless headphones',
  'fitness smartwatch',
  'gaming laptop',
  'home audio speakers',
  '4k tv',
]

function hueFromString(str = '') {
  let h = 0
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) % 360
  return h
}

async function fetchResults(query, page) {
  const res = await fetch(`/api/search?query=${encodeURIComponent(query)}&page=${page}`)
  if (!res.ok) throw new Error(`Search failed (${res.status})`)
  return res.json()
}

export default function App() {
  const [query, setQuery] = useState('')
  const [committed, setCommitted] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [showDrop, setShowDrop] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  const [trending, setTrending] = useState([])
  const [detailSku, setDetailSku] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [dataset, setDataset] = useState(null)
  const [showUpload, setShowUpload] = useState(false)
  const [showSql, setShowSql] = useState(false)
  const debounce = useRef()

  const refreshTrending = useCallback(() => {
    fetch('/api/trending')
      .then((r) => r.json())
      .then((d) => setTrending(d.products || []))
      .catch(() => setTrending([]))
  }, [])

  useEffect(() => {
    fetch('/api/health').then((r) => r.json()).then(setHealth).catch(() => {})
    fetch('/api/dataset').then((r) => r.json()).then(setDataset).catch(() => {})
    refreshTrending()
  }, [refreshTrending])

  // Live autocomplete.
  useEffect(() => {
    clearTimeout(debounce.current)
    setActiveIndex(-1)
    if (!query.trim() || query === committed) {
      setSuggestions([])
      setShowDrop(false)
      return
    }
    debounce.current = setTimeout(async () => {
      try {
        const d = await fetchResults(query, 1)
        setSuggestions(d.products.slice(0, 5))
        setShowDrop(true)
      } catch {
        setSuggestions([])
        setShowDrop(false)
      }
    }, 200)
    return () => clearTimeout(debounce.current)
  }, [query, committed])

  const loadPage = useCallback(async (q, p) => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchResults(q, p))
    } catch (e) {
      setError(e.message)
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const commit = useCallback((text) => {
    const q = (text || '').trim()
    if (!q) return
    setShowDrop(false)
    setActiveIndex(-1)
    setDetailSku(null)
    setDetail(null)
    setQuery(q)
    setCommitted(q)
    loadPage(q, 1)
  }, [loadPage])

  const selectProduct = useCallback(async (sku) => {
    setDetailSku(sku)
    setDetailLoading(true)
    setShowDrop(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
    try {
      const res = await fetch(`/api/product/${encodeURIComponent(sku)}`)
      if (!res.ok) throw new Error('Not found')
      setDetail(await res.json())
    } catch {
      setDetail(null)
    } finally {
      setDetailLoading(false)
    }
  }, [])

  const back = () => {
    setDetailSku(null)
    setDetail(null)
  }

  const goToPage = (p) => {
    loadPage(committed, p)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  // Apply a new active dataset (after upload-build or reset).
  const applyDataset = useCallback((info) => {
    setDataset(info)
    setQuery('')
    setCommitted('')
    setData(null)
    setDetailSku(null)
    setDetail(null)
    setError(null)
    fetch('/api/health').then((r) => r.json()).then(setHealth).catch(() => {})
    refreshTrending()
  }, [refreshTrending])

  const resetDataset = async () => {
    try {
      const info = await (await fetch('/api/dataset/reset', { method: 'POST' })).json()
      applyDataset(info)
    } catch { /* ignore */ }
  }

  const onKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (suggestions.length) {
        setShowDrop(true)
        setActiveIndex((i) => Math.min(i + 1, suggestions.length - 1))
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex((i) => Math.max(i - 1, -1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (activeIndex >= 0 && suggestions[activeIndex]) commit(suggestions[activeIndex].title)
      else commit(query)
    } else if (e.key === 'Escape') {
      setShowDrop(false)
      setActiveIndex(-1)
    }
  }

  const uploaded = dataset?.source === 'uploaded'

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <span className="logo" onClick={back} style={{ cursor: 'pointer' }}>◆</span>
          <span>ProductSearch</span>
          <StatusChip health={health} />
        </div>
        <h1>Find products by meaning, not just keywords</h1>
        <p className="subtitle">
          Hybrid search with recommendations — on our data or yours.
        </p>
      </header>

      <DatasetBar
        dataset={dataset}
        onUpload={() => setShowUpload(true)}
        onSql={() => setShowSql(true)}
        onReset={resetDataset}
      />

      <div className="search-card">
        <div className="search-wrap">
          <span className="search-icon">⌕</span>
          <input
            className="search-input"
            type="text"
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            onFocus={() => suggestions.length && setShowDrop(true)}
            onBlur={() => setTimeout(() => setShowDrop(false), 150)}
            placeholder={uploaded
              ? 'Search your data…'
              : 'Search products… e.g. wireless headphones, TV & Display, speakers, printers'}
          />
          {loading && <span className="spinner" aria-label="loading" />}

          {showDrop && suggestions.length > 0 && (
            <ul className="dropdown">
              {suggestions.map((s, i) => (
                <li
                  key={s.sku}
                  className={`dd-item ${i === activeIndex ? 'active' : ''}`}
                  onMouseDown={(e) => { e.preventDefault(); commit(s.title) }}
                  onMouseEnter={() => setActiveIndex(i)}
                >
                  <span className="dd-icon">⌕</span>
                  <span className="dd-title">{s.title}</span>
                  {s.category && <span className="dd-cat">{s.category}</span>}
                </li>
              ))}
            </ul>
          )}
        </div>

        {!query && !detailSku && !uploaded && (
          <div className="suggestions">
            <span className="suggestions-label">Try:</span>
            {SUGGESTIONS.map((s) => (
              <button key={s} className="chip" onClick={() => commit(s)}>{s}</button>
            ))}
          </div>
        )}
      </div>

      {error && <p className="error">⚠️ {error}</p>}

      {detailSku ? (
        <DetailView detail={detail} loading={detailLoading} onBack={back} onSelect={selectProduct} />
      ) : data && query ? (
        <>
          <div className="meta-bar">
            <span>
              {data.total_results > 0
                ? `${data.total_results} result${data.total_results > 1 ? 's' : ''} for “${data.query}”`
                : `No matches above the similarity threshold for “${data.query}”`}
            </span>
            {data.total_pages > 1 && (
              <span className="page-indicator">Page {data.page} of {data.total_pages}</span>
            )}
          </div>
          <ProductGrid products={data.products} onSelect={selectProduct} />
          {data.total_pages > 1 && (
            <Pagination page={data.page} totalPages={data.total_pages} onChange={goToPage} />
          )}
        </>
      ) : trending.length > 0 ? (
        <Section title={dataset?.has_trending ? '🔥 Trending now' : '📄 Sample of your data'}>
          <ProductGrid products={trending} onSelect={selectProduct} showScore={false} />
        </Section>
      ) : null}

      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onComplete={(info) => { setShowUpload(false); applyDataset(info) }}
        />
      )}
      {showSql && (
        <SqlModal
          onClose={() => setShowSql(false)}
          onComplete={(info) => { setShowSql(false); applyDataset(info) }}
        />
      )}
    </div>
  )
}

/* ---------- dataset bar ---------- */
function DatasetBar({ dataset, onUpload, onSql, onReset }) {
  if (!dataset) return null
  const uploaded = dataset.source === 'uploaded'
  return (
    <div className="dataset-bar">
      <span className="ds-label">
        <span className="ds-icon">{uploaded ? '📄' : '🛒'}</span>
        Searching <b>{dataset.name}</b>
        <span className="ds-sub">
          · {dataset.total.toLocaleString()} rows
        </span>
      </span>
      <span className="ds-actions">
        <button className="ds-btn primary" onClick={onUpload}>⬆ Upload file</button>
        <button className="ds-btn" onClick={onSql}>🗄 Connect SQL</button>
        {uploaded && (
          <a className="ds-btn" href="/api/dataset/download" download>⬇ Download CSV</a>
        )}
        {uploaded && <button className="ds-btn" onClick={onReset}>Reset to default</button>}
      </span>
    </div>
  )
}

/* ---------- shared column picker (used by file + SQL flows) ---------- */
function ColumnPicker({ res, onComplete }) {
  const [selected, setSelected] = useState(() => new Set())
  const [rows, setRows] = useState('')
  const [building, setBuilding] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [error, setError] = useState(null)

  const toggle = (col) => setSelected((prev) => {
    const next = new Set(prev)
    next.has(col) ? next.delete(col) : next.add(col)
    return next
  })

  const build = async () => {
    setError(null)
    setBuilding(true)
    try {
      const r = await fetch(`/api/upload/${res.upload_id}/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          search_columns: [...selected],
          max_rows: rows ? Number(rows) : null,
        }),
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail || 'Build failed')
      onComplete(body)
    } catch (err) {
      setError(err.message)
    } finally {
      setBuilding(false)
    }
  }

  return (
    <>
      <p className="upload-info">
        <b>{res.filename}</b> · {res.n_rows.toLocaleString()} rows · {res.columns.length} columns
      </p>

      {res.truncated && (
        <p className="dropped-note">
          ⚠️ Truncation done — your data has {res.total_rows.toLocaleString()} rows,
          but the limit is {res.max_rows.toLocaleString()}. The search will run on the
          first {res.max_rows.toLocaleString()} rows.
        </p>
      )}

      {res.dropped_columns?.length > 0 && (
        <p className="dropped-note">
          ⚠️ Dropped {res.dropped_columns.length} empty column(s) (all values were
          null): <b>{res.dropped_columns.join(', ')}</b>
        </p>
      )}

      <div className="picker-actions">
        <button className="link-btn" onClick={() => setShowPreview((v) => !v)}>
          {showPreview ? '🙈 Hide data' : '👁 Preview data'}
        </button>
        <a className="link-btn" href={`/api/upload/${res.upload_id}/download`} download>
          ⬇ Download CSV
        </a>
      </div>

      {showPreview && <PreviewTable columns={res.columns} rows={res.preview} />}

      <p className="upload-q">Which column(s) should the search run on?</p>
      <div className="col-list">
        {res.columns.map((col) => (
          <label key={col} className={`col-item ${selected.has(col) ? 'on' : ''}`}>
            <input type="checkbox" checked={selected.has(col)} onChange={() => toggle(col)} />
            {col}
          </label>
        ))}
      </div>
      <p className="upload-hint">
        Multiple columns are concatenated (separated by “,”) into one text field,
        then embedded.
      </p>

      <label className="rows-fld">
        <span>Rows to include (optional — default all, max {res.max_rows.toLocaleString()})</span>
        <input
          type="number"
          min="1"
          max={res.n_rows}
          value={rows}
          onChange={(e) => setRows(e.target.value)}
          placeholder={`all (${res.n_rows.toLocaleString()})`}
        />
      </label>

      {error && <p className="error modal-error">⚠️ {error}</p>}
      <div className="modal-foot">
        <button
          className="ds-btn primary"
          disabled={selected.size === 0 || building}
          onClick={build}
        >
          {building ? 'Building index…' : `Build & search (${selected.size})`}
        </button>
      </div>
    </>
  )
}

function PreviewTable({ columns, rows }) {
  if (!rows?.length) return <p className="upload-hint">No preview rows.</p>
  return (
    <div className="preview-wrap">
      <table className="preview-table">
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{columns.map((c) => <td key={c}>{r[c]}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ---------- file upload modal ---------- */
function UploadModal({ onClose, onComplete }) {
  const [uploading, setUploading] = useState(false)
  const [res, setRes] = useState(null)
  const [error, setError] = useState(null)

  const handleFile = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    setRes(null)
    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const r = await fetch('/api/upload', { method: 'POST', body: fd })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail || 'Upload failed')
      setRes(body)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Upload your data</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {!res ? (
          <>
            <div className="upload-drop">
              <p>Choose a <b>.csv</b>, <b>.xls</b> or <b>.xlsx</b> file.</p>
              <label className="file-btn">
                {uploading ? 'Reading…' : 'Select file'}
                <input type="file" accept=".csv,.xls,.xlsx" onChange={handleFile} disabled={uploading} hidden />
              </label>
            </div>
            {error && <p className="error modal-error">⚠️ {error}</p>}
            <div className="modal-foot">
              <button className="ds-btn" onClick={onClose}>Cancel</button>
            </div>
          </>
        ) : (
          <ColumnPicker res={res} onComplete={onComplete} />
        )}
      </div>
    </div>
  )
}

/* ---------- SQL connect modal ---------- */
const DIALECTS = [
  { value: 'postgresql', label: 'PostgreSQL', port: 5432 },
  { value: 'mysql', label: 'MySQL', port: 3306 },
  { value: 'mssql', label: 'SQL Server', port: 1433 },
  { value: 'snowflake', label: 'Snowflake', port: '' },
]

function SqlModal({ onClose, onComplete }) {
  const [form, setForm] = useState({
    dialect: 'postgresql', host: '', port: '', user: '', password: '',
    database: '', schema: '', warehouse: '', role: '',
    query: 'SELECT * FROM your_table LIMIT 1000',
  })
  const [running, setRunning] = useState(false)
  const [res, setRes] = useState(null)
  const [error, setError] = useState(null)

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))
  const isSnowflake = form.dialect === 'snowflake'

  const runQuery = async () => {
    setError(null)
    setRunning(true)
    try {
      const payload = {
        dialect: form.dialect,
        host: form.host.trim(),
        port: form.port ? Number(form.port) : null,
        user: form.user,
        password: form.password,
        database: form.database,
        query: form.query,
        schema: form.schema || null,
        warehouse: form.warehouse || null,
        role: form.role || null,
      }
      const r = await fetch('/api/sql/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail || 'Query failed')
      setForm((f) => ({ ...f, password: '' })) // clear password from memory once used
      setRes(body)
    } catch (err) {
      setError(err.message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Connect a SQL database</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        {!res ? (
          <>
            <div className="sql-grid">
              <label className="fld fld-wide">
                <span>Database type</span>
                <select value={form.dialect} onChange={set('dialect')}>
                  {DIALECTS.map((d) => <option key={d.value} value={d.value}>{d.label}</option>)}
                </select>
              </label>
              <label className="fld">
                <span>{isSnowflake ? 'Account' : 'Host'}</span>
                <input value={form.host} onChange={set('host')}
                  placeholder={isSnowflake ? 'xy12345.eu-west-1' : 'db.example.com'} />
              </label>
              {!isSnowflake && (
                <label className="fld">
                  <span>Port</span>
                  <input value={form.port} onChange={set('port')}
                    placeholder={String(DIALECTS.find((d) => d.value === form.dialect)?.port || '')} />
                </label>
              )}
              <label className="fld">
                <span>User</span>
                <input value={form.user} onChange={set('user')} autoComplete="off" />
              </label>
              <label className="fld">
                <span>Password</span>
                <input type="password" value={form.password} onChange={set('password')} autoComplete="new-password" />
              </label>
              <label className="fld">
                <span>Database</span>
                <input value={form.database} onChange={set('database')} />
              </label>
              {isSnowflake && (
                <>
                  <label className="fld"><span>Schema</span>
                    <input value={form.schema} onChange={set('schema')} /></label>
                  <label className="fld"><span>Warehouse</span>
                    <input value={form.warehouse} onChange={set('warehouse')} /></label>
                  <label className="fld"><span>Role</span>
                    <input value={form.role} onChange={set('role')} /></label>
                </>
              )}
            </div>

            <label className="fld fld-wide">
              <span>Query (read-only — SELECT / WITH; single table or joins)</span>
              <textarea className="sql-query" value={form.query} onChange={set('query')} rows={4} />
            </label>

            <p className="sql-secure">
              🔒 Your password is used once to run this query, then discarded — it
              is never saved or logged.
            </p>
            {error && <p className="error modal-error">⚠️ {error}</p>}
            <div className="modal-foot">
              <button className="ds-btn" onClick={onClose}>Cancel</button>
              <button className="ds-btn primary" disabled={running || !form.host || !form.query}
                onClick={runQuery}>
                {running ? 'Running query…' : 'Run query'}
              </button>
            </div>
          </>
        ) : (
          <ColumnPicker res={res} onComplete={onComplete} />
        )}
      </div>
    </div>
  )
}

/* ---------- reusable pieces ---------- */
function ProductGrid({ products, onSelect, showScore = true }) {
  return (
    <div className="product-grid">
      {products.map((p) => (
        <ProductCard key={p.sku} p={p} onSelect={onSelect} showScore={showScore} />
      ))}
    </div>
  )
}

function ProductCard({ p, onSelect, showScore = true }) {
  const hue = hueFromString(p.category || p.title)
  const hasMeta = p.category || p.total_reviews > 0
  // For uploaded data with no category/reviews, show a couple of raw fields.
  const extra = hasMeta
    ? []
    : Object.entries(p.fields || {}).filter(([, v]) => v && v !== p.title).slice(0, 2)
  const showSku = p.sku && /\D/.test(p.sku) // hide pure-numeric row indices

  return (
    <article className="product-card clickable" onClick={() => onSelect?.(p.sku)}>
      <div
        className="thumb"
        style={{
          background: `linear-gradient(135deg, hsl(${hue} 70% 92%), hsl(${hue + 40} 70% 84%))`,
          color: `hsl(${hue} 45% 35%)`,
        }}
      >
        <span className="thumb-letter">
          {(p.category || p.title || '?').trim().charAt(0).toUpperCase()}
        </span>
        <span className="rank-badge">#{p.rank}</span>
        {showScore && p.score > 0 && <ScoreBadge value={p.score} />}
      </div>
      <div className="pc-body">
        <h3 className="pc-title">{p.title}</h3>
        {extra.map(([k, v]) => (
          <div key={k} className="pc-field"><span>{k}:</span> {v}</div>
        ))}
        <div className="pc-foot">
          <span className="pc-left">
            {p.category && <span className="tag">{p.category}</span>}
            {p.total_reviews > 0 && (
              <span className="pc-reviews">★ {p.total_reviews.toLocaleString()}</span>
            )}
          </span>
          {showSku && <span className="pc-sku">{p.sku}</span>}
        </div>
      </div>
    </article>
  )
}

function Section({ title, children }) {
  return (
    <section className="rec-section">
      <h2 className="section-title">{title}</h2>
      {children}
    </section>
  )
}

function DetailView({ detail, loading, onBack, onSelect }) {
  if (loading || !detail) {
    return <div className="detail-loading"><span className="spinner" /></div>
  }
  const { product, similar, category } = detail
  const hue = hueFromString(product.category || product.title)
  return (
    <div className="detail">
      <button className="back-btn" onClick={onBack}>← Back</button>

      <div className="detail-hero">
        <div
          className="detail-thumb"
          style={{
            background: `linear-gradient(135deg, hsl(${hue} 70% 92%), hsl(${hue + 40} 70% 84%))`,
            color: `hsl(${hue} 45% 35%)`,
          }}
        >
          {(product.category || product.title || '?').trim().charAt(0).toUpperCase()}
        </div>
        <div className="detail-info">
          {product.category && <div className="tag">{product.category}</div>}
          <h2 className="detail-title">{product.title}</h2>
          <dl className="detail-fields">
            {Object.entries(product.fields || {}).map(([k, v]) => (
              <div key={k} className="df-row">
                <dt>{k}</dt>
                <dd>{v || '—'}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {similar.length > 0 && (
        <Section title="You may also like">
          <ProductGrid products={similar} onSelect={onSelect} />
        </Section>
      )}
      {category.length > 0 && (
        <Section title={`More from ${product.category}`}>
          <ProductGrid products={category} onSelect={onSelect} />
        </Section>
      )}
    </div>
  )
}

function ScoreBadge({ value }) {
  const pct = Math.max(0, Math.min(1, value))
  const hue = Math.round(pct * 120)
  return (
    <span
      className="score-badge"
      style={{
        color: `hsl(${hue} 70% 30%)`,
        background: `hsl(${hue} 80% 96%)`,
        borderColor: `hsl(${hue} 60% 78%)`,
      }}
      title="hybrid match score"
    >
      {(pct * 100).toFixed(0)}% match
    </span>
  )
}

function Pagination({ page, totalPages, onChange }) {
  const pages = []
  const span = 2
  const start = Math.max(1, page - span)
  const end = Math.min(totalPages, page + span)
  for (let i = start; i <= end; i++) pages.push(i)

  return (
    <nav className="pagination">
      <button disabled={page <= 1} onClick={() => onChange(page - 1)}>‹ Prev</button>
      {start > 1 && (
        <>
          <button onClick={() => onChange(1)}>1</button>
          {start > 2 && <span className="ellipsis">…</span>}
        </>
      )}
      {pages.map((p) => (
        <button key={p} className={p === page ? 'active' : ''} onClick={() => onChange(p)}>{p}</button>
      ))}
      {end < totalPages && (
        <>
          {end < totalPages - 1 && <span className="ellipsis">…</span>}
          <button onClick={() => onChange(totalPages)}>{totalPages}</button>
        </>
      )}
      <button disabled={page >= totalPages} onClick={() => onChange(page + 1)}>Next ›</button>
    </nav>
  )
}

function StatusChip({ health }) {
  if (!health) return <span className="status status-off">offline</span>
  if (!health.ready) return <span className="status status-warn">starting…</span>
  return (
    <span className="status status-on">
      {health.total_products.toLocaleString()} items
    </span>
  )
}
