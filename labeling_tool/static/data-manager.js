// ══════════════════════════════════════════════════════════════
// Enterprise Data Management Module v2.0 — Enhanced UI
// ══════════════════════════════════════════════════════════════

(function injectDMStyles() {
    if (document.getElementById('dm-v2-styles')) return;
    const s = document.createElement('style');
    s.id = 'dm-v2-styles';
    s.textContent = `
@keyframes dm-shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
.dm-skel {
  border-radius: var(--radius-lg);
  background: linear-gradient(90deg, var(--bg-3) 25%, var(--bg-2) 50%, var(--bg-3) 75%);
  background-size: 200% 100%;
  animation: dm-shimmer 1.2s ease-in-out infinite;
}
.dm-card {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
}
.dm-card--glow:hover {
  border-color: rgba(122,162,255,0.35);
  box-shadow: 0 0 0 1px rgba(122,162,255,0.08), 0 8px 32px rgba(0,0,0,0.35), 0 0 24px rgba(122,162,255,0.06);
}
.dm-stat-card {
  border-radius: 10px;
  padding: 16px;
  border: 1px solid var(--border);
  background: linear-gradient(145deg, var(--bg-3) 0%, var(--bg-2) 50%, rgba(28,34,54,0.9) 100%);
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, border-color 0.2s ease;
}
.dm-stat-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 80% 50% at 100% 0%, var(--accent-dim), transparent 55%);
  pointer-events: none;
}
.dm-stat-card:hover { transform: translateY(-2px); border-color: rgba(122,162,255,0.25); }
.dm-stat-card__inner { position: relative; z-index: 1; }
.dm-stat-card__icon {
  width: 32px; height: 32px; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; margin-bottom: 8px;
  background: var(--accent-dim); color: var(--accent);
}
.dm-progress-track {
  height: 6px; border-radius: 999px;
  background: var(--bg-3);
  overflow: hidden; margin-top: 8px;
}
.dm-progress-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-hover));
  transition: width 0.45s ease;
}
.dm-btn-primary {
  display: inline-flex; align-items: center; justify-content: center;
  gap: 6px;
  padding: 6px 14px;
  font-size: 11px; font-weight: 600;
  color: var(--bg-0) !important;
  background: linear-gradient(180deg, var(--accent-hover), var(--accent)) !important;
  border: 1px solid rgba(122,162,255,0.4);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: filter 0.2s ease, box-shadow 0.2s ease;
}
.dm-btn-primary:hover {
  filter: brightness(1.08);
  box-shadow: 0 0 16px rgba(122,162,255,0.35);
}
.dm-warn-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 999px;
  font-size: 9px; font-weight: 600;
  background: rgba(251,191,36,0.12);
  color: var(--warn);
  border: 1px solid rgba(251,191,36,0.25);
}
.dm-expand-panel {
  max-height: 0; overflow: hidden;
  transition: max-height 0.35s ease, opacity 0.25s ease;
  opacity: 0;
}
.dm-dataset-card.dm-card--expanded .dm-expand-panel {
  max-height: 480px; opacity: 1; margin-top: 12px; padding-top: 12px;
  border-top: 1px solid var(--border);
}
.dm-chevron { transition: transform 0.25s ease; display: inline-block; }
.dm-dataset-card.dm-card--expanded .dm-chevron { transform: rotate(90deg); }
.dm-ring-wrap {
  width: 128px; height: 128px; position: relative; margin: 0 auto;
}
.dm-ring {
  width: 100%; height: 100%; border-radius: 50%;
  background: conic-gradient(var(--accent) calc(var(--dm-pct, 0) * 1%), var(--bg-3) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #000 calc(100% - 11px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #000 calc(100% - 11px));
}
.dm-ring-label {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; pointer-events: none;
}
.dm-project-card {
  border-radius: 10px;
  padding: 16px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  position: relative;
  transition: box-shadow 0.25s ease, transform 0.2s ease;
}
.dm-project-card::before {
  content: ''; position: absolute; inset: 0; border-radius: 10px; padding: 1px;
  background: linear-gradient(135deg, transparent, transparent);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
  transition: background 0.3s ease;
}
.dm-project-card:hover::before {
  background: linear-gradient(135deg, rgba(122,162,255,0.5), rgba(52,211,153,0.15), rgba(122,162,255,0.35));
}
.dm-project-card:hover {
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
  transform: translateY(-1px);
}
.dm-radio-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
.dm-radio-item {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 10px 12px;
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: border-color 0.2s ease, background 0.2s ease;
}
.dm-radio-item:hover { border-color: rgba(122,162,255,0.35); background: var(--bg-2); }
.dm-radio-item input[type="radio"] {
  accent-color: var(--accent); margin-top: 3px; flex-shrink: 0;
}
.dm-audit-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.dm-audit-table thead tr {
  background: var(--bg-3);
  border-bottom: 1px solid var(--border);
}
.dm-audit-table th {
  text-align: left; padding: 10px 12px;
  color: var(--text-2); font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.dm-audit-table tbody tr {
  border-bottom: 1px solid rgba(37,45,66,0.6);
  transition: background 0.15s ease;
}
.dm-audit-table tbody tr:nth-child(even) { background: rgba(21,25,41,0.45); }
.dm-audit-table tbody tr:nth-child(odd) { background: rgba(16,19,31,0.25); }
.dm-audit-table tbody tr:hover { background: rgba(122,162,255,0.06); }
.dm-audit-table td { padding: 8px 12px; vertical-align: middle; }
.dm-empty {
  text-align: center; padding: 48px 24px; color: var(--text-2);
  border: 1px dashed var(--border); border-radius: 10px;
  background: linear-gradient(180deg, var(--bg-2), var(--bg-1));
}
.dm-empty__art {
  font-family: ui-monospace, monospace; font-size: 11px; line-height: 1.4;
  color: var(--text-2); margin-bottom: 16px; white-space: pre;
  opacity: 0.85;
}
.dm-section-title {
  font-size: 13px; font-weight: 700; color: var(--text-0);
  letter-spacing: 0.02em; margin-bottom: 12px;
}
.dm-input, .dm-select {
  background: var(--bg-3); border: 1px solid var(--border); color: var(--text-0);
  border-radius: var(--radius-md); padding: 6px 10px; font-size: 11px;
  transition: border-color 0.2s ease;
}
.dm-input:focus, .dm-select:focus {
  outline: none; border-color: var(--accent);
}
.dm-browser-item {
  display: flex; align-items: center; gap: 4px;
  padding: 5px 10px; border-radius: var(--radius-md);
  font-size: 11px; cursor: pointer; transition: background 0.15s;
}
.dm-browser-item:hover { background: rgba(107,147,255,0.12); }
.dm-bc-item:hover { text-decoration: underline; }
.dm-metric-grid { display: grid; gap: 12px; }
@media (min-width: 720px) {
  .dm-metric-grid--4 { grid-template-columns: repeat(4, 1fr); }
  .dm-metric-grid--3 { grid-template-columns: repeat(3, 1fr); }
}
`;
    document.head.appendChild(s);
})();

const DM_API = '/api';
const DM_BAR_COLORS = ['#7aa2ff', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#fb923c', '#22d3ee', '#e879f9', '#38bdf8', '#4ade80'];

let _dmDatasetStatsCache = {};
let _dmGroupsCache = null;

function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function dmWarningText(w) {
    if (w == null) return '';
    if (typeof w === 'string') return w;
    return w.message || JSON.stringify(w);
}

function dmEstimateSizeBuckets(stats) {
    const total = stats.total_annotations || 0;
    const warnings = stats.quality_warnings || [];
    let tiny = 0;
    for (const w of warnings) {
        if (w && w.type === 'tiny_annotation') tiny = w.count || 0;
    }
    tiny = Math.min(tiny, total);
    const other = Math.max(0, total - tiny);
    const mean = (stats.bbox_size_stats && stats.bbox_size_stats.mean_area) || 0;
    const cSmall = 0.01, cMed = 0.05, cLarge = 0.15;
    const wSmall = 1 / (1 + Math.abs(mean - cSmall) * 80);
    const wMed = 1 / (1 + Math.abs(mean - cMed) * 80);
    const wLarge = 1 / (1 + Math.abs(mean - cLarge) * 80);
    const sumW = wSmall + wMed + wLarge || 1;
    let s = Math.round((other * wSmall) / sumW);
    let m = Math.round((other * wMed) / sumW);
    let l = other - s - m;
    if (l < 0) { m += l; l = 0; }
    if (m < 0) { s += m; m = 0; }
    return {
        tiny,
        small: s,
        medium: m,
        large: l,
        note: '小/中/大 为基于均值面积的估算，精确直方图需服务端扩展',
    };
}

function dmSkeletonSummaryRow() {
    return `
<div class="dm-metric-grid dm-metric-grid--4" style="margin-bottom:20px;">
  ${[1, 2, 3, 4].map(() => '<div class="dm-skel" style="height:88px;"></div>').join('')}
</div>`;
}

function dmSkeletonDatasetList(n = 4) {
    return `<div style="display:flex;flex-direction:column;gap:10px;">
  ${Array.from({ length: n }, () => '<div class="dm-skel" style="height:96px;"></div>').join('')}
</div>`;
}

const DatasetManager = {
    _initialized: false,
    init() {
        if (!this._initialized) {
            switchDMTab('datasets');
            this._initialized = true;
        }
    },
    show() {
        const modal = document.getElementById('dataset-manager-modal');
        if (modal) modal.style.display = 'flex';
        this.init();
    },
    hide() {
        const modal = document.getElementById('dataset-manager-modal');
        if (modal) modal.style.display = 'none';
    },
    refresh() {
        this._initialized = false;
        this.init();
    },
};

function switchDMTab(tab) {
    const tabs = ['datasets', 'projects', 'stats', 'evaluate', 'export', 'audit'];
    tabs.forEach(t => {
        const btn = document.getElementById(`dm-tab-${t}`);
        if (btn) btn.className = t === tab ? 'dm-tab-btn active' : 'dm-tab-btn';
    });
    const content = document.getElementById('dm-content');
    if (!content) return;

    switch (tab) {
        case 'datasets': renderDMDatasets(content); break;
        case 'projects': renderDMProjects(content); break;
        case 'stats': renderDMStats(content); break;
        case 'evaluate': renderDMEvaluate(content); break;
        case 'export': renderDMExport(content); break;
        case 'audit': renderDMAudit(content); break;
        default: renderDMDatasets(content);
    }
}

// ── Datasets Tab ──────────────────────────────────────────────

async function renderDMDatasets(container) {
    _dmDatasetStatsCache = {};
    container.innerHTML = `
<div class="dm-section-title">数据集</div>
${dmSkeletonSummaryRow()}
${dmSkeletonDatasetList(5)}`;

    try {
        const [groupsRes, overviewRes] = await Promise.all([
            fetch(`${DM_API}/groups`),
            fetch(`${DM_API}/stats/overview`),
        ]);
        const groups = await groupsRes.json();
        const overview = await overviewRes.json();
        _dmGroupsCache = groups;

        try {
            const statsAllRes = await fetch(`${DM_API}/datasets/stats_all`, { method: 'POST' });
            if (statsAllRes.ok) {
                const allStats = await statsAllRes.json();
                for (const s of allStats) {
                    if (s.group_id) _dmDatasetStatsCache[s.group_id] = s;
                }
            }
        } catch {}

        const pct = v => Math.min(100, Math.round((v || 0) * 100));
        const overviewPct = Math.min(100, overview.label_percentage || 0);

        let html = `
<div class="dm-metric-grid dm-metric-grid--4" style="margin-bottom:20px;">
  <div class="dm-stat-card dm-card--glow">
    <div class="dm-stat-card__inner">
      <div class="dm-stat-card__icon">◇</div>
      <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;letter-spacing:0.06em;">数据集</div>
      <div style="font-size:28px;font-weight:800;color:var(--accent);margin-top:4px;">${overview.total_datasets}</div>
    </div>
  </div>
  <div class="dm-stat-card dm-card--glow">
    <div class="dm-stat-card__inner">
      <div class="dm-stat-card__icon">▣</div>
      <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;letter-spacing:0.06em;">总图像</div>
      <div style="font-size:28px;font-weight:800;color:var(--text-0);margin-top:4px;">${(overview.total_images || 0).toLocaleString()}</div>
    </div>
  </div>
  <div class="dm-stat-card dm-card--glow">
    <div class="dm-stat-card__inner">
      <div class="dm-stat-card__icon">✓</div>
      <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;letter-spacing:0.06em;">已标注</div>
      <div style="font-size:28px;font-weight:800;color:var(--success);margin-top:4px;">${(overview.total_labeled || 0).toLocaleString()}</div>
    </div>
  </div>
  <div class="dm-stat-card dm-card--glow">
    <div class="dm-stat-card__inner">
      <div class="dm-stat-card__icon">%</div>
      <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;letter-spacing:0.06em;">标注率</div>
      <div style="font-size:28px;font-weight:800;margin-top:4px;color:${overviewPct > 80 ? 'var(--success)' : overviewPct > 50 ? 'var(--warn)' : 'var(--danger)'};">${overviewPct}%</div>
      <div class="dm-progress-track"><div class="dm-progress-fill" style="width:${overviewPct}%;background:linear-gradient(90deg,var(--accent),var(--success));"></div></div>
    </div>
  </div>
</div>

<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:10px;">
  <div class="dm-section-title" style="margin:0;">数据集列表</div>
  <div style="display:flex;align-items:center;gap:8px;">
    <button type="button" class="dm-btn-primary" onclick="dmShowAddDataset()" style="font-size:11px;padding:5px 14px;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      添加数据集
    </button>
    <input type="text" id="dm-search" class="dm-input" placeholder="搜索数据集..." style="width:220px;"
      oninput="filterDMDatasets()">
  </div>
</div>

<div id="dm-add-dataset-panel" style="display:none;margin-bottom:16px;">
  <div class="dm-card" style="border-color:rgba(107,147,255,0.3);">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <div style="font-size:13px;font-weight:700;color:var(--text-0);">添加新数据集</div>
      <button type="button" onclick="document.getElementById('dm-add-dataset-panel').style.display='none'" style="background:none;border:none;color:var(--text-2);cursor:pointer;font-size:16px;">✕</button>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px;">
      <div>
        <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">① 选择目录 <span style="color:var(--danger);">*</span></label>
        <div style="display:flex;gap:6px;">
          <input type="text" id="dm-add-ds-path" class="dm-input" placeholder="点击 浏览 选择目录，或手动输入路径" style="flex:1;">
          <button type="button" class="dm-btn-primary" style="font-size:11px;padding:5px 12px;white-space:nowrap;" onclick="dmToggleBrowser('dm-add-ds-path')">📂 浏览</button>
        </div>
      </div>
      <div id="dm-browser-dm-add-ds-path" class="dm-dir-browser" style="display:none;"></div>
      <div>
        <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">② 名称（可选）</label>
        <input type="text" id="dm-add-ds-name" class="dm-input" placeholder="自定义名称，留空使用目录名" style="width:100%;">
      </div>
      <div id="dm-add-ds-preview" style="display:none;"></div>
      <div style="display:flex;gap:8px;justify-content:flex-end;">
        <button type="button" class="btn btn-ghost" style="font-size:11px;padding:5px 14px;" onclick="dmPreviewDataset()">③ 检测结构</button>
        <button type="button" class="dm-btn-primary" style="font-size:11px;padding:5px 14px;" onclick="dmAddDataset()">④ 确认添加</button>
      </div>
    </div>
  </div>
</div>

<div id="dm-dataset-list" style="display:flex;flex-direction:column;gap:10px;">`;

        for (const g of groups) {
            const trainCount = g.train_count || 0;
            const valCount = g.val_count || 0;
            const totalCount = trainCount + valCount;
            const lsCount = g.label_sets?.length || 0;
            const classCount = Object.keys(g.names || {}).length;
            const classNames = Object.values(g.names || {}).slice(0, 5).join(', ');
            const st = _dmDatasetStatsCache[g.group_id];
            const prog = st ? pct(st.label_progress) : null;
            const warnCount = st && Array.isArray(st.quality_warnings) ? st.quality_warnings.length : 0;
            const encId = encodeURIComponent(g.group_id);
            const expandId = encId.replace(/%/g, '_');
            const userDsId = g._user_dataset_id || '';
            const safeUserDsId = escapeHtml(userDsId).replace(/'/g, "\\'");

            html += `
<div class="dm-dataset-card dm-card dm-card--glow" data-name="${escapeHtml(g.group_name.toLowerCase())}" data-dm-gid="${encId}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
    <div style="flex:1;min-width:0;">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">
        <span style="font-size:14px;font-weight:700;color:var(--text-0);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%;">${escapeHtml(g.group_name)}</span>
        ${g.has_val ? '<span class="tag">Train+Val</span>' : '<span class="tag" style="background:rgba(255,123,134,0.1);color:var(--danger);border-color:rgba(255,123,134,0.2);">Train Only</span>'}
        ${userDsId ? '<span class="tag" style="background:rgba(107,147,255,0.1);color:var(--accent);border-color:rgba(107,147,255,0.2);font-size:8px;">用户添加</span>' : ''}
        ${warnCount > 0 ? `<span class="dm-warn-badge" title="质量提示">⚠ ${warnCount}</span>` : ''}
      </div>
      <div style="font-size:10px;color:var(--text-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(g.group_id)}">${escapeHtml(g.group_id)}</div>
      ${prog != null ? `
      <div style="margin-top:10px;">
        <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-2);margin-bottom:4px;">
          <span>标注覆盖</span><span style="color:var(--text-1);font-weight:600;">${prog}%</span>
        </div>
        <div class="dm-progress-track"><div class="dm-progress-fill" style="width:${prog}%;"></div></div>
      </div>` : '<div style="margin-top:8px;font-size:10px;color:var(--text-2);">统计加载中或不可用</div>'}
    </div>
    <div style="display:flex;flex-direction:column;gap:6px;flex-shrink:0;align-items:flex-end;">
      <button type="button" class="dm-btn-primary" onclick="dmSelectDataset('${encId.replace(/'/g, "\\'")}')">打开</button>
      <div style="display:flex;gap:4px;flex-wrap:wrap;">
        <button type="button" class="btn btn-ghost" style="font-size:10px;padding:4px 8px;"
          onclick="dmToggleDatasetExpand('${encId.replace(/'/g, "\\'")}')"><span class="dm-chevron">▸</span> 分布</button>
        <button type="button" class="btn btn-ghost" style="font-size:10px;padding:4px 8px;"
          onclick="dmExportDataset('${encId.replace(/'/g, "\\'")}')">导出</button>
        ${userDsId ? `<button type="button" class="btn btn-ghost" style="font-size:10px;padding:4px 8px;"
          onclick="dmRenameDataset('${safeUserDsId}','${escapeHtml(g.group_name).replace(/'/g, "\\'")}')">重命名</button>` : ''}
        <button type="button" class="btn btn-ghost" style="font-size:10px;padding:4px 8px;color:var(--danger);"
          onclick="dmHideDataset('${encId.replace(/'/g, "\\'")}','${escapeHtml(g.group_name).replace(/'/g, "\\'")}','${safeUserDsId}')">${userDsId ? '移除' : '隐藏'}</button>
      </div>
    </div>
  </div>
  <div style="display:flex;gap:16px;margin-top:10px;font-size:11px;flex-wrap:wrap;align-items:center;">
    <div style="color:var(--text-1);"><span style="color:var(--text-2);">图像</span> <strong>${totalCount.toLocaleString()}</strong>${g.has_val ? ` <span style="color:var(--text-2);font-size:9px;">(${trainCount}+${valCount})</span>` : ''}</div>
    <div style="color:var(--text-1);"><span style="color:var(--text-2);">标注集</span> <strong>${lsCount}</strong>
      <span style="font-size:9px;color:var(--text-2);">(${(g.label_sets||[]).map(ls=>ls.set_name).join(', ')})</span>
    </div>
    <div style="color:var(--text-1);"><span style="color:var(--text-2);">类别</span> <strong>${classCount}</strong> <span style="color:var(--text-2);font-size:9px;" title="${escapeHtml(classNames)}">(${escapeHtml(classNames)})</span></div>
    <div style="margin-left:auto;display:flex;gap:4px;">
      <button type="button" class="btn btn-ghost" style="font-size:9px;padding:2px 8px;" onclick="dmCreateLabelSet('${encId.replace(/'/g, "\\'")}')">+ 标注集</button>
      <button type="button" class="btn btn-ghost" style="font-size:9px;padding:2px 8px;" onclick="dmToggleFileManager('${encId.replace(/'/g, "\\'")}')">📁 文件</button>
    </div>
  </div>
  <div class="dm-expand-panel" id="dm-expand-${encId.replace(/%/g, '_')}">
    <div id="dm-expand-inner-${encId.replace(/%/g, '_')}" style="font-size:11px;color:var(--text-2);">点击「分布」展开类别统计</div>
  </div>
  <div id="dm-files-${encId.replace(/%/g, '_')}" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid var(--border);"></div>
</div>`;
        }

        html += '</div>';
        if (groups.length === 0) {
            html += `<div class="dm-empty" style="margin-top:16px;">
<div class="dm-empty__art">   ┌──────────┐
   │  ∅ 无数据 │
   └──────────┘</div>
<div>未找到数据集配置，请检查 data*.yaml</div>
</div>`;
        }

        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<div class="dm-empty"><div class="dm-empty__art">!!!</div><div style="color:var(--danger);">加载失败: ${escapeHtml(e.message)}</div></div>`;
    }
}

function dmToggleDatasetExpand(encodedId) {
    const gid = decodeURIComponent(encodedId);
    const safe = encodeURIComponent(gid).replace(/%/g, '_');
    const encId = encodeURIComponent(gid);
    const target = document.querySelector(`.dm-dataset-card[data-dm-gid="${encId}"]`);
    if (!target) return;
    const isOpen = target.classList.toggle('dm-card--expanded');
    const inner = document.getElementById(`dm-expand-inner-${safe}`);
    if (!inner || !isOpen) return;
    const st = _dmDatasetStatsCache[gid];
    if (!st) {
        inner.innerHTML = '<span style="color:var(--text-2);">暂无统计数据</span>';
        return;
    }
    const dist = st.class_distribution || {};
    const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) {
        inner.innerHTML = '<span style="color:var(--text-2);">该数据集尚无类别分布</span>';
        return;
    }
    const total = entries.reduce((s, [, v]) => s + v, 0);
    let h = '<div style="font-weight:600;color:var(--text-1);margin-bottom:8px;">类别分布</div><div style="display:flex;flex-direction:column;gap:6px;">';
    entries.slice(0, 12).forEach(([name, count], i) => {
        const p = total ? ((count / total) * 100).toFixed(1) : '0';
        const color = DM_BAR_COLORS[i % DM_BAR_COLORS.length];
        h += `<div style="display:flex;align-items:center;gap:8px;">
<span style="width:72px;text-align:right;font-size:10px;color:var(--text-1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
<div style="flex:1;height:16px;background:var(--bg-3);border-radius:4px;overflow:hidden;">
<div style="width:${p}%;height:100%;background:${color};min-width:2px;transition:width .4s ease;"></div>
</div>
<span style="min-width:52px;text-align:right;font-size:10px;color:var(--text-2);">${count} (${p}%)</span>
</div>`;
    });
    h += '</div>';
    inner.innerHTML = h;
}

function filterDMDatasets() {
    const q = (document.getElementById('dm-search')?.value || '').toLowerCase();
    document.querySelectorAll('.dm-dataset-card').forEach(card => {
        const name = card.dataset.name || '';
        card.style.display = name.includes(q) ? '' : 'none';
    });
}

async function dmSelectDataset(encodedId) {
    const groupId = decodeURIComponent(encodedId);
    const modal = document.getElementById('dataset-manager-modal');
    if (modal) modal.style.display = 'none';

    // Access state via window to avoid const/let cross-script scope issues
    const appState = window.state;
    if (!appState) {
        alert('标注页面尚未初始化，请刷新页面后重试 (Ctrl+F5)');
        return;
    }

    const overlay = _dmShowLoadingOverlay('正在加载数据集...');
    try {
        // Step 1: Refresh groups
        _dmSetLoadingStep(overlay, 1, '正在同步数据集列表...');
        const freshRes = await fetch(`${DM_API}/groups`);
        if (!freshRes.ok) throw new Error(`获取数据集列表失败 (HTTP ${freshRes.status})`);
        const freshGroups = await freshRes.json();
        appState.groups = freshGroups;
        const groupSelectEl = document.getElementById('group-select');
        if (groupSelectEl) {
            groupSelectEl.innerHTML = freshGroups.map(g =>
                `<option value="${g.group_id}">${g.group_name}</option>`
            ).join('');
        }

        // Step 2: Find and set group
        _dmSetLoadingStep(overlay, 2, '正在定位数据集...');
        const group = appState.groups.find(g => g.group_id === groupId);
        if (!group) {
            throw new Error('未找到匹配的数据集: ' + groupId.slice(-40));
        }
        appState.dirty = false;
        appState.currentGroup = group;
        if (groupSelectEl) groupSelectEl.value = groupId;
        if (typeof window.resetFewshotState === 'function') window.resetFewshotState();

        const gInfoEl = document.getElementById('group-info');
        if (gInfoEl) gInfoEl.textContent = `${group.nc}类 · Train:${group.train_count} Val:${group.val_count}`;

        // Populate label set
        const lsEl = document.getElementById('labelset-select');
        if (lsEl) {
            lsEl.innerHTML = group.label_sets.map(ls =>
                `<option value="${ls.set_id}">${ls.set_name} (${ls.label_format})</option>`
            ).join('');
        }
        if (group.label_sets.length > 0) {
            appState.currentLabelSet = group.label_sets[0];
        }

        // Step 3: Load classes
        _dmSetLoadingStep(overlay, 3, '正在加载类别信息...');
        if (typeof window.loadClasses === 'function') await window.loadClasses();

        // Step 4: Load images
        _dmSetLoadingStep(overlay, 4, '正在加载图片列表...');
        if (typeof window.loadImages === 'function') await window.loadImages();

        // Step 5: Finalize
        _dmSetLoadingStep(overlay, 5, '正在初始化标注环境...');
        if (typeof window.loadSavedSupports === 'function') await window.loadSavedSupports();
        if (typeof window.loadHybridPromptNames === 'function') window.loadHybridPromptNames();
        if (typeof window.warmupPreviewCache === 'function') window.warmupPreviewCache();
        if (typeof window._updateEmptyState === 'function') window._updateEmptyState();
        if (typeof window.updateStatusBar === 'function') window.updateStatusBar();

        _dmHideLoadingOverlay(overlay);
        if (typeof window.showStatus === 'function') {
            window.showStatus('已加载数据集: ' + (group.group_name || groupId));
        }
    } catch (err) {
        console.error('[dmSelectDataset] error:', err);
        _dmHideLoadingOverlay(overlay);
        const msg = '加载数据集出错: ' + (err.message || '未知错误');
        if (typeof window.showStatus === 'function') {
            window.showStatus(msg, true);
        } else {
            alert(msg);
        }
    }
}

function _dmShowLoadingOverlay(title) {
    let overlay = document.getElementById('dm-loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'dm-loading-overlay';
        overlay.style.cssText = 'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.6);backdrop-filter:blur(6px);transition:opacity 0.2s;';
        overlay.innerHTML = `
        <div style="background:var(--bg-1);border:1px solid var(--border);border-radius:var(--radius-xl);padding:32px 40px;max-width:400px;width:90%;box-shadow:0 16px 64px rgba(0,0,0,0.5);text-align:center;">
            <div id="dm-load-spinner" style="width:40px;height:40px;margin:0 auto 16px;border:3px solid var(--bg-3);border-top-color:var(--accent);border-radius:50%;animation:dm-load-spin 0.8s linear infinite;"></div>
            <div id="dm-load-title" style="font-size:15px;font-weight:700;color:var(--text-0);margin-bottom:12px;">${title || '加载中...'}</div>
            <div id="dm-load-steps" style="display:flex;flex-direction:column;gap:6px;text-align:left;"></div>
        </div>`;
        const spin = document.createElement('style');
        spin.textContent = '@keyframes dm-load-spin{to{transform:rotate(360deg)}}';
        overlay.appendChild(spin);
        document.body.appendChild(overlay);
    } else {
        overlay.style.display = 'flex';
        overlay.style.opacity = '1';
        const titleEl = overlay.querySelector('#dm-load-title');
        if (titleEl) titleEl.textContent = title || '加载中...';
    }
    const stepsEl = overlay.querySelector('#dm-load-steps');
    if (stepsEl) stepsEl.innerHTML = '';
    return overlay;
}

function _dmSetLoadingStep(overlay, stepNum, text) {
    if (!overlay) return;
    const stepsEl = overlay.querySelector('#dm-load-steps');
    if (!stepsEl) return;
    const totalSteps = 5;
    // Mark previous steps as done
    stepsEl.querySelectorAll('.dm-load-step').forEach(el => {
        const n = parseInt(el.dataset.step);
        if (n < stepNum) {
            el.querySelector('.dm-load-step-icon').textContent = '✓';
            el.querySelector('.dm-load-step-icon').style.color = 'var(--success)';
            el.querySelector('.dm-load-step-text').style.color = 'var(--text-2)';
        }
    });
    let stepEl = stepsEl.querySelector(`[data-step="${stepNum}"]`);
    if (!stepEl) {
        stepEl = document.createElement('div');
        stepEl.className = 'dm-load-step';
        stepEl.dataset.step = stepNum;
        stepEl.style.cssText = 'display:flex;align-items:center;gap:8px;padding:4px 0;font-size:12px;';
        stepEl.innerHTML = `<span class="dm-load-step-icon" style="width:16px;text-align:center;font-size:10px;color:var(--accent);flex-shrink:0;">●</span><span class="dm-load-step-text" style="color:var(--text-1);flex:1;">${text}</span><span style="color:var(--text-2);font-size:10px;">${stepNum}/${totalSteps}</span>`;
        stepsEl.appendChild(stepEl);
    }
}

function _dmHideLoadingOverlay(overlay) {
    if (!overlay) overlay = document.getElementById('dm-loading-overlay');
    if (!overlay) return;
    overlay.style.opacity = '0';
    setTimeout(() => { overlay.style.display = 'none'; }, 200);
}

function dmExportDataset(encodedId) {
    switchDMTab('export');
    setTimeout(() => {
        const sel = document.getElementById('dm-export-group');
        if (sel) sel.value = decodeURIComponent(encodedId);
        if (_dmGroupsCache) updateExportLabelSets(_dmGroupsCache);
    }, 120);
}

// ══════════════════════════════════════════════════════════════
// Statistics Tab
// ══════════════════════════════════════════════════════════════

async function renderDMStats(container) {
    container.innerHTML = dmSkeletonSummaryRow() + dmSkeletonDatasetList(3);
    try {
        const [summaryRes, dailyRes, overviewRes] = await Promise.all([
            fetch(`${DM_API}/stats/summary?days=30`),
            fetch(`${DM_API}/stats/daily?days=14`),
            fetch(`${DM_API}/stats/overview`),
        ]);
        const summary = await summaryRes.json();
        const daily = await dailyRes.json();
        const overview = await overviewRes.json();

        let html = `
<div class="dm-section-title">30天活动总览</div>
<div class="dm-metric-grid dm-metric-grid--4" style="margin-bottom:24px;">
  <div class="dm-stat-card"><div class="dm-stat-card__inner">
    <div class="dm-stat-card__icon" style="background:var(--accent-dim);color:var(--accent);">📷</div>
    <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;">标注图像</div>
    <div style="font-size:24px;font-weight:800;color:var(--accent);margin-top:4px;">${summary.images_annotated}</div>
    <div style="font-size:10px;color:var(--text-2);margin-top:2px;">日均 ${summary.avg_daily_images}</div>
  </div></div>
  <div class="dm-stat-card"><div class="dm-stat-card__inner">
    <div class="dm-stat-card__icon" style="background:var(--success-dim);color:var(--success);">✎</div>
    <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;">创建标注</div>
    <div style="font-size:24px;font-weight:800;color:var(--success);margin-top:4px;">${summary.annotations_created}</div>
    <div style="font-size:10px;color:var(--text-2);margin-top:2px;">日均 ${summary.avg_daily_annotations}</div>
  </div></div>
  <div class="dm-stat-card"><div class="dm-stat-card__inner">
    <div class="dm-stat-card__icon" style="background:var(--warn-dim);color:var(--warn);">🤖</div>
    <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;">AI辅助</div>
    <div style="font-size:24px;font-weight:800;color:var(--warn);margin-top:4px;">${summary.ai_assists_used}</div>
  </div></div>
  <div class="dm-stat-card"><div class="dm-stat-card__inner">
    <div class="dm-stat-card__icon" style="background:rgba(167,139,250,0.12);color:#a78bfa;">📅</div>
    <div style="font-size:10px;color:var(--text-2);text-transform:uppercase;">活跃天数</div>
    <div style="font-size:24px;font-weight:800;color:var(--text-0);margin-top:4px;">${summary.active_days}</div>
  </div></div>
</div>

<div class="dm-section-title">每日活动</div>
<div class="dm-card" style="padding:20px;margin-bottom:24px;">`;

        if (daily.length === 0) {
            html += '<div style="text-align:center;padding:24px;color:var(--text-2);">暂无活动数据</div>';
        } else {
            const maxVal = Math.max(1, ...daily.map(d => d.annotations_created));
            const sorted = [...daily].sort((a, b) => a.date.localeCompare(b.date)).slice(-14);
            html += '<div style="display:flex;align-items:flex-end;gap:6px;height:140px;padding-bottom:24px;position:relative;">';
            for (const d of sorted) {
                const pct = (d.annotations_created / maxVal) * 100;
                html += `<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;min-width:0;">
                    <div style="font-size:9px;color:var(--text-1);font-weight:600;">${d.annotations_created}</div>
                    <div style="width:100%;background:linear-gradient(180deg,var(--accent),rgba(107,147,255,0.4));border-radius:4px 4px 0 0;
                        height:${Math.max(4, pct)}%;min-height:4px;transition:height .3s ease;"></div>
                    <div style="font-size:8px;color:var(--text-2);white-space:nowrap;transform:rotate(-45deg);margin-top:4px;">${d.date.slice(5)}</div>
                </div>`;
            }
            html += '</div>';
        }

        html += `</div>
<div class="dm-section-title">系统概览</div>
<div class="dm-metric-grid dm-metric-grid--3" style="margin-bottom:24px;">
  <div class="dm-stat-card"><div class="dm-stat-card__inner">
    <div style="font-size:10px;color:var(--text-2);">总数据集</div>
    <div style="font-size:22px;font-weight:700;color:var(--text-0);margin-top:4px;">${overview.total_datasets}</div>
  </div></div>
  <div class="dm-stat-card"><div class="dm-stat-card__inner">
    <div style="font-size:10px;color:var(--text-2);">总图像</div>
    <div style="font-size:22px;font-weight:700;color:var(--text-0);margin-top:4px;">${(overview.total_images || 0).toLocaleString()}</div>
  </div></div>
  <div class="dm-stat-card"><div class="dm-stat-card__inner">
    <div style="font-size:10px;color:var(--text-2);">标注覆盖率</div>
    <div style="font-size:22px;font-weight:700;margin-top:4px;color:${(overview.label_percentage||0)>80?'var(--success)':'var(--warn)'};">${Math.min(100,overview.label_percentage||0)}%</div>
    <div class="dm-progress-track" style="margin-top:6px;"><div class="dm-progress-fill" style="width:${Math.min(100,overview.label_percentage||0)}%;"></div></div>
  </div></div>
</div>

<div class="dm-section-title">类别分布统计</div>
<div class="dm-card" style="padding:20px;">
  <div style="margin-bottom:12px;">
    <select id="dm-stats-group-select" class="dm-select" style="width:260px;"
        onchange="_loadClassDistribution()">
        <option value="">选择数据集查看详情...</option>
    </select>
  </div>
  <div id="dm-class-dist-chart" style="color:var(--text-2);text-align:center;padding:24px;font-size:11px;">
    请选择一个数据集查看类别分布
  </div>
</div>`;

        container.innerHTML = html;

        try {
            const gRes = await fetch(`${DM_API}/groups`);
            const groups = await gRes.json();
            const sel = document.getElementById('dm-stats-group-select');
            if (sel) groups.forEach(g => {
                const opt = document.createElement('option');
                opt.value = g.group_id;
                opt.textContent = g.group_name;
                sel.appendChild(opt);
            });
        } catch {}
    } catch (e) {
        container.innerHTML = `<div class="dm-empty"><div style="color:var(--danger);">加载失败: ${escapeHtml(e.message)}</div></div>`;
    }
}

async function _loadClassDistribution() {
    const groupId = document.getElementById('dm-stats-group-select')?.value;
    const chartDiv = document.getElementById('dm-class-dist-chart');
    if (!chartDiv) return;
    if (!groupId) { chartDiv.innerHTML = '<div style="padding:24px;color:var(--text-2);text-align:center;">请选择数据集</div>'; return; }
    chartDiv.innerHTML = '<div style="padding:24px;color:var(--text-2);text-align:center;">加载中...</div>';
    try {
        const res = await fetch(`${DM_API}/datasets/stats?group_id=${encodeURIComponent(groupId)}`);
        const stats = await res.json();
        const dist = stats.class_distribution || {};
        const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
        if (entries.length === 0) { chartDiv.innerHTML = '<div style="padding:24px;color:var(--text-2);text-align:center;">该数据集暂无标注数据</div>'; return; }

        const total = entries.reduce((s, [, v]) => s + v, 0);
        let html = '<div style="display:flex;gap:20px;align-items:flex-start;">';
        html += '<div style="flex:1;display:flex;flex-direction:column;gap:6px;">';
        entries.forEach(([name, count], i) => {
            const pct = (count / total * 100).toFixed(1);
            const color = DM_BAR_COLORS[i % DM_BAR_COLORS.length];
            html += `<div style="display:flex;align-items:center;gap:8px;">
                <div style="width:80px;text-align:right;font-size:10px;color:var(--text-1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
                <div style="flex:1;height:20px;background:var(--bg-3);border-radius:4px;overflow:hidden;">
                    <div style="width:${pct}%;height:100%;background:linear-gradient(90deg,${color},${color}88);border-radius:4px;min-width:2px;transition:width .4s;"></div>
                </div>
                <div style="min-width:60px;text-align:right;font-size:10px;color:var(--text-1);">${count} (${pct}%)</div>
            </div>`;
        });
        html += '</div>';

        const gradStops = [];
        let cum = 0;
        entries.forEach(([, count], i) => {
            const pct = count / total * 100;
            gradStops.push(`${DM_BAR_COLORS[i % DM_BAR_COLORS.length]} ${cum}% ${cum + pct}%`);
            cum += pct;
        });
        html += `<div style="flex-shrink:0;width:140px;text-align:center;">
            <div style="width:120px;height:120px;border-radius:50%;margin:0 auto 10px;
                background:conic-gradient(${gradStops.join(',')});
                box-shadow:0 0 0 4px var(--bg-2),0 4px 16px rgba(0,0,0,0.3);"></div>
            <div style="font-size:11px;color:var(--text-2);">总计 <strong style="color:var(--text-0);">${total.toLocaleString()}</strong> 标注</div>
        </div></div>
        <div style="margin-top:14px;font-size:10px;color:var(--text-2);border-top:1px solid var(--border);padding-top:10px;">
            标注图像: <strong>${stats.labeled_images||0}/${stats.total_images||0}</strong>
            · 总标注: <strong>${(stats.total_annotations||0).toLocaleString()}</strong>
            · 均标注/图: <strong>${(stats.avg_annotations_per_image||0).toFixed(1)}</strong>
            ${(stats.quality_warnings||[]).length > 0 ? `<br/>质量提示: <span style="color:var(--warn);">${stats.quality_warnings.map(w=>dmWarningText(w)).join('; ')}</span>` : ''}
        </div>`;
        chartDiv.innerHTML = html;
    } catch (e) {
        chartDiv.innerHTML = `<div style="padding:20px;color:var(--danger);text-align:center;">加载失败: ${escapeHtml(e.message)}</div>`;
    }
}

// ══════════════════════════════════════════════════════════════
// Evaluate Tab — Simplified research evaluation
// ══════════════════════════════════════════════════════════════

async function renderDMEvaluate(container) {
    let groupOpts = '<option value="">-- 手动输入路径 --</option>';
    try {
        if (!_dmGroupsCache) {
            const gRes = await fetch(`${DM_API}/groups`);
            _dmGroupsCache = await gRes.json();
        }
        for (const g of (_dmGroupsCache || [])) {
            groupOpts += `<option value="${escapeHtml(g.group_id)}">${escapeHtml(g.group_name)}</option>`;
        }
    } catch {}

    container.innerHTML = `
<div class="dm-section-title">标注对比评估</div>
<div style="font-size:11px;color:var(--text-2);margin-bottom:16px;">
    选择数据集自动填充路径，或手动输入目录路径。支持分割和分类两种评估模式。
</div>

<div class="dm-card" style="border-color:rgba(107,147,255,0.3);">
  <div style="display:flex;flex-direction:column;gap:14px;">

    <div>
      <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;font-weight:600;">快速选择数据集</label>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
        <select id="dm-eval-group" class="dm-select" onchange="dmEvalGroupChanged()">
          ${groupOpts}
        </select>
        <select id="dm-eval-gold-ls" class="dm-select" onchange="dmEvalLsChanged()">
          <option value="">① 金标准标注集</option>
        </select>
        <select id="dm-eval-pred-ls" class="dm-select" onchange="dmEvalLsChanged()">
          <option value="">② 预测标注集</option>
        </select>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;">
        <select id="dm-eval-subset" class="dm-select" onchange="dmEvalLsChanged()">
          <option value="train">Train 子集</option>
          <option value="val">Val 子集</option>
        </select>
        <input type="number" id="dm-eval-max-images" class="dm-input" placeholder="最大图片数（留空=全部）" min="1">
      </div>
    </div>

    <details style="margin-top:2px;">
      <summary style="font-size:10px;color:var(--text-2);cursor:pointer;user-select:none;">手动输入路径（高级）</summary>
      <div style="margin-top:10px;display:flex;flex-direction:column;gap:10px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
          <div>
            <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">金标准标注目录</label>
            <div style="display:flex;gap:6px;">
              <input type="text" id="dm-eval-gold-path" class="dm-input" placeholder="金标准标注目录路径" style="flex:1;">
              <button type="button" class="dm-btn-primary" style="font-size:10px;padding:4px 10px;" onclick="dmToggleBrowser('dm-eval-gold-path')">📂</button>
            </div>
            <div id="dm-browser-dm-eval-gold-path" class="dm-dir-browser" style="display:none;"></div>
          </div>
          <div>
            <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">预测标注目录</label>
            <div style="display:flex;gap:6px;">
              <input type="text" id="dm-eval-pred-path" class="dm-input" placeholder="预测标注目录路径" style="flex:1;">
              <button type="button" class="dm-btn-primary" style="font-size:10px;padding:4px 10px;" onclick="dmToggleBrowser('dm-eval-pred-path')">📂</button>
            </div>
            <div id="dm-browser-dm-eval-pred-path" class="dm-dir-browser" style="display:none;"></div>
          </div>
        </div>
        <div>
          <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">图片目录</label>
          <div style="display:flex;gap:6px;">
            <input type="text" id="dm-eval-images-path" class="dm-input" placeholder="对应的图片目录路径" style="flex:1;">
            <button type="button" class="dm-btn-primary" style="font-size:10px;padding:4px 10px;" onclick="dmToggleBrowser('dm-eval-images-path')">📂</button>
          </div>
          <div id="dm-browser-dm-eval-images-path" class="dm-dir-browser" style="display:none;"></div>
        </div>
      </div>
    </details>

    <div style="display:flex;gap:8px;justify-content:flex-end;">
      <button type="button" class="dm-btn-primary" style="font-size:12px;padding:7px 20px;" onclick="dmRunEval('segment')">
        🎯 分割评估
      </button>
      <button type="button" class="dm-btn-primary" style="font-size:12px;padding:7px 20px;background:linear-gradient(180deg,#34d399,#059669) !important;border-color:rgba(52,211,153,0.4);" onclick="dmRunEval('classify')">
        🏷 分类评估
      </button>
    </div>

  </div>
</div>

<div id="dm-eval-status" style="margin-top:12px;"></div>
<div id="dm-eval-results" style="margin-top:16px;"></div>
`;
}

function dmEvalGroupChanged() {
    const gid = document.getElementById('dm-eval-group')?.value;
    const goldLs = document.getElementById('dm-eval-gold-ls');
    const predLs = document.getElementById('dm-eval-pred-ls');
    if (!goldLs || !predLs) return;
    if (!gid) {
        goldLs.innerHTML = '<option value="">① 金标准标注集</option>';
        predLs.innerHTML = '<option value="">② 预测标注集</option>';
        return;
    }
    const group = (_dmGroupsCache || []).find(g => g.group_id === gid);
    if (!group) return;
    const lsOpts = (group.label_sets || []).map(ls =>
        `<option value="${ls.set_id}">${escapeHtml(ls.set_name)} (${ls.label_format})</option>`
    ).join('');
    goldLs.innerHTML = '<option value="">选择金标准标注集</option>' + lsOpts;
    predLs.innerHTML = '<option value="">选择预测标注集</option>' + lsOpts;
}

async function dmEvalLsChanged() {
    const gid = document.getElementById('dm-eval-group')?.value;
    if (!gid) return;
    const goldLsId = document.getElementById('dm-eval-gold-ls')?.value;
    const predLsId = document.getElementById('dm-eval-pred-ls')?.value;
    const subset = document.getElementById('dm-eval-subset')?.value || 'train';
    const group = (_dmGroupsCache || []).find(g => g.group_id === gid);
    if (!group) return;

    try {
        const res = await fetch(`${DM_API}/datasets/resolve_paths`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ group_id: gid, gold_label_set_id: goldLsId, pred_label_set_id: predLsId, subset }),
        });
        if (res.ok) {
            const paths = await res.json();
            const goldInp = document.getElementById('dm-eval-gold-path');
            const predInp = document.getElementById('dm-eval-pred-path');
            const imgInp = document.getElementById('dm-eval-images-path');
            if (goldInp && paths.gold_path) goldInp.value = paths.gold_path;
            if (predInp && paths.pred_path) predInp.value = paths.pred_path;
            if (imgInp && paths.images_path) imgInp.value = paths.images_path;
        }
    } catch {}
}

async function dmRunEval(evalType) {
    let goldPath = document.getElementById('dm-eval-gold-path')?.value?.trim();
    let predPath = document.getElementById('dm-eval-pred-path')?.value?.trim();
    let imagesPath = document.getElementById('dm-eval-images-path')?.value?.trim();

    const gid = document.getElementById('dm-eval-group')?.value;
    if (gid && (!goldPath || !predPath || !imagesPath)) {
        const goldLsId = document.getElementById('dm-eval-gold-ls')?.value;
        const predLsId = document.getElementById('dm-eval-pred-ls')?.value;
        const subset = document.getElementById('dm-eval-subset')?.value || 'train';
        if (!goldLsId || !predLsId) { alert('请选择金标准标注集和预测标注集'); return; }
        try {
            const res = await fetch(`${DM_API}/datasets/resolve_paths`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ group_id: gid, gold_label_set_id: goldLsId, pred_label_set_id: predLsId, subset }),
            });
            if (res.ok) {
                const paths = await res.json();
                goldPath = paths.gold_path || goldPath;
                predPath = paths.pred_path || predPath;
                imagesPath = paths.images_path || imagesPath;
            }
        } catch {}
    }

    const maxImagesRaw = document.getElementById('dm-eval-max-images')?.value?.trim();
    const maxImages = maxImagesRaw ? parseInt(maxImagesRaw) : null;

    if (!goldPath) { alert('请选择金标准标注目录'); return; }
    if (!predPath) { alert('请选择预测标注目录'); return; }
    if (!imagesPath) { alert('请选择图片目录'); return; }

    const statusEl = document.getElementById('dm-eval-status');
    const resultsEl = document.getElementById('dm-eval-results');
    if (statusEl) statusEl.innerHTML = `<div class="dm-card" style="padding:14px;display:flex;align-items:center;gap:10px;">
        <div class="loader-ring" style="width:20px;height:20px;border-width:2px;"></div>
        <span style="font-size:12px;color:var(--text-1);">正在执行${evalType === 'segment' ? '分割' : '分类'}评估，请稍候...</span>
    </div>`;
    if (resultsEl) resultsEl.innerHTML = '';

    const endpoint = evalType === 'segment' ? '/api/evaluate/simple_segment' : '/api/evaluate/simple_classify';
    const body = {
        gold_labels_dir: goldPath,
        pred_labels_dir: predPath,
        images_dir: imagesPath,
    };
    if (maxImages && maxImages > 0) body.max_images = maxImages;

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `评估失败 (HTTP ${res.status})`);
        }
        const data = await res.json();
        if (statusEl) statusEl.innerHTML = '';
        if (resultsEl) _dmRenderEvalResults(resultsEl, data, evalType);
    } catch (err) {
        if (statusEl) statusEl.innerHTML = `<div class="dm-card" style="padding:14px;border-color:var(--danger);">
            <div style="color:var(--danger);font-size:12px;font-weight:600;">评估失败</div>
            <div style="color:var(--text-2);font-size:11px;margin-top:4px;">${escapeHtml(err.message)}</div>
        </div>`;
    }
}

function _dmRenderEvalResults(container, data, evalType) {
    const m = data.metrics || {};
    const fmtPct = v => (v * 100).toFixed(1) + '%';
    const fmtF = v => v.toFixed(3);
    const scoreColor = v => v >= 0.8 ? 'var(--success)' : v >= 0.5 ? 'var(--warn)' : 'var(--danger)';

    let metricsHtml = '';
    if (evalType === 'segment') {
        const cards = [
            { label: 'Precision', value: m.precision, fmt: fmtPct },
            { label: 'Recall', value: m.recall, fmt: fmtPct },
            { label: 'F1 Score', value: m.f1, fmt: fmtPct },
            { label: 'Mean IoU', value: m.mean_matched_iou, fmt: fmtF },
            { label: 'Mean Dice', value: m.mean_matched_dice, fmt: fmtF },
        ];
        metricsHtml = cards.map(c => `<div style="background:var(--bg-3);border-radius:var(--radius-md);padding:14px;text-align:center;">
            <div style="font-size:9px;color:var(--text-2);text-transform:uppercase;margin-bottom:4px;">${c.label}</div>
            <div style="font-size:24px;font-weight:800;color:${scoreColor(c.value || 0)};">${c.fmt(c.value || 0)}</div>
        </div>`).join('');
    } else {
        const cards = [
            { label: 'Accuracy', value: m.accuracy, fmt: fmtPct },
            { label: 'Precision', value: m.precision, fmt: fmtPct },
            { label: 'Recall', value: m.recall, fmt: fmtPct },
            { label: 'F1 Score', value: m.f1, fmt: fmtPct },
        ];
        metricsHtml = cards.map(c => `<div style="background:var(--bg-3);border-radius:var(--radius-md);padding:14px;text-align:center;">
            <div style="font-size:9px;color:var(--text-2);text-transform:uppercase;margin-bottom:4px;">${c.label}</div>
            <div style="font-size:24px;font-weight:800;color:${scoreColor(c.value || 0)};">${c.fmt(c.value || 0)}</div>
        </div>`).join('');
    }

    let summaryHtml = `
<div class="dm-card dm-card--glow">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
    <span style="font-size:14px;font-weight:700;color:var(--text-0);">
        ${evalType === 'segment' ? '🎯 分割评估结果' : '🏷 分类评估结果'}
    </span>
    <span class="tag">${data.image_count || 0} 张图片</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(${evalType === 'segment' ? 5 : 4},1fr);gap:10px;margin-bottom:14px;">
    ${metricsHtml}
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:11px;color:var(--text-2);">
    <div>预测总数: <strong style="color:var(--text-0);">${evalType === 'segment' ? m.pred_count : m.total_pred}</strong></div>
    <div>金标准总数: <strong style="color:var(--text-0);">${evalType === 'segment' ? m.gold_count : m.total_gold}</strong></div>
    <div>匹配数: <strong style="color:var(--success);">${evalType === 'segment' ? m.matched : m.total_correct}</strong></div>
  </div>
</div>`;

    // Per-class metrics (classify only)
    if (evalType === 'classify' && data.class_metrics && data.class_metrics.length > 0) {
        summaryHtml += `
<div class="dm-card" style="margin-top:12px;">
  <div style="font-size:13px;font-weight:700;color:var(--text-0);margin-bottom:10px;">各类别详情</div>
  <table class="dm-audit-table">
    <thead><tr>
        <th>类别</th><th>Gold</th><th>Pred</th><th>Correct</th><th>Precision</th><th>Recall</th><th>F1</th>
    </tr></thead>
    <tbody>
    ${data.class_metrics.map(c => `<tr>
        <td style="font-weight:600;color:var(--text-0);">${escapeHtml(c.class_name)}</td>
        <td>${c.gold}</td>
        <td>${c.pred}</td>
        <td style="color:var(--success);">${c.correct}</td>
        <td style="color:${scoreColor(c.precision)};">${fmtPct(c.precision)}</td>
        <td style="color:${scoreColor(c.recall)};">${fmtPct(c.recall)}</td>
        <td style="color:${scoreColor(c.f1)};font-weight:600;">${fmtPct(c.f1)}</td>
    </tr>`).join('')}
    </tbody>
  </table>
</div>`;
    }

    // Sample images table
    if (data.sample_images && data.sample_images.length > 0) {
        summaryHtml += `
<div class="dm-card" style="margin-top:12px;">
  <div style="font-size:13px;font-weight:700;color:var(--text-0);margin-bottom:10px;">逐图详情 (前${data.sample_images.length}张)</div>
  <div style="max-height:300px;overflow-y:auto;">
  <table class="dm-audit-table">
    <thead><tr>
        <th>文件名</th>
        <th>${evalType === 'segment' ? 'Pred' : 'Pred'}</th>
        <th>${evalType === 'segment' ? 'Gold' : 'Gold'}</th>
        <th>${evalType === 'segment' ? 'Matched' : 'Correct'}</th>
        ${evalType === 'segment' ? '<th>Precision</th><th>Recall</th><th>F1</th>' : ''}
    </tr></thead>
    <tbody>
    ${data.sample_images.map(s => `<tr>
        <td style="color:var(--text-0);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(s.filename)}">${escapeHtml(s.filename)}</td>
        <td>${s.pred_count}</td>
        <td>${s.gold_count}</td>
        <td style="color:var(--success);">${evalType === 'segment' ? s.matched : s.correct}</td>
        ${evalType === 'segment' ? `
            <td style="color:${scoreColor(s.precision || 0)};">${fmtPct(s.precision || 0)}</td>
            <td style="color:${scoreColor(s.recall || 0)};">${fmtPct(s.recall || 0)}</td>
            <td style="color:${scoreColor(s.f1 || 0)};font-weight:600;">${fmtPct(s.f1 || 0)}</td>
        ` : ''}
    </tr>`).join('')}
    </tbody>
  </table>
  </div>
</div>`;
    }

    container.innerHTML = summaryHtml;
}

// ══════════════════════════════════════════════════════════════
// Export Tab
// ══════════════════════════════════════════════════════════════

async function renderDMExport(container) {
    container.innerHTML = dmSkeletonSummaryRow();
    try {
        const [groupsRes, historyRes] = await Promise.all([
            fetch(`${DM_API}/groups`),
            fetch(`${DM_API}/export_history?limit=20`),
        ]);
        const groups = await groupsRes.json();
        const history = await historyRes.json();
        _dmGroupsCache = groups;

        const formats = [
            { value:'yolo', name:'YOLO', desc:'TXT 多边形/矩形', icon:'📦' },
            { value:'coco', name:'COCO', desc:'JSON 实例分割', icon:'🏷' },
            { value:'voc', name:'VOC', desc:'XML 目标检测', icon:'📄' },
            { value:'masks', name:'Mask', desc:'PNG 语义分割', icon:'🎨' },
            { value:'csv', name:'CSV', desc:'统计表格', icon:'📊' },
        ];

        let html = `
<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
  <div>
    <div class="dm-section-title">新建导出</div>
    <div class="dm-card" style="display:flex;flex-direction:column;gap:14px;">
      <div>
        <label style="font-size:10px;color:var(--text-2);text-transform:uppercase;display:block;margin-bottom:4px;">数据集</label>
        <select id="dm-export-group" class="dm-select" style="width:100%;">
            ${groups.map(g => `<option value="${escapeHtml(g.group_id)}">${escapeHtml(g.group_name)}</option>`).join('')}
        </select>
      </div>
      <div>
        <label style="font-size:10px;color:var(--text-2);text-transform:uppercase;display:block;margin-bottom:4px;">标注集</label>
        <select id="dm-export-labelset" class="dm-select" style="width:100%;"><option value="">加载中...</option></select>
      </div>
      <div>
        <label style="font-size:10px;color:var(--text-2);text-transform:uppercase;display:block;margin-bottom:4px;">子集</label>
        <select id="dm-export-subset" class="dm-select" style="width:100%;">
            <option value="train">Train</option>
            <option value="val">Val</option>
        </select>
      </div>
      <div>
        <label style="font-size:10px;color:var(--text-2);text-transform:uppercase;display:block;margin-bottom:6px;">导出格式</label>
        <div class="dm-radio-grid">
          ${formats.map(f => `<label class="dm-radio-item">
              <input type="radio" name="export-format" value="${f.value}" ${f.value==='yolo'?'checked':''}>
              <div>
                <div style="font-size:12px;font-weight:600;color:var(--text-0);">${f.icon} ${f.name}</div>
                <div style="font-size:9px;color:var(--text-2);">${f.desc}</div>
              </div>
          </label>`).join('')}
        </div>
      </div>
      <button id="dm-export-btn" class="dm-btn-primary" style="width:100%;padding:10px;font-size:12px;" onclick="runDMExport()">
        开始导出
      </button>
      <div id="dm-export-status" style="font-size:11px;color:var(--text-2);min-height:20px;"></div>
    </div>
  </div>
  <div>
    <div class="dm-section-title">导出历史</div>
    <div style="display:flex;flex-direction:column;gap:8px;">`;

        if (history.length === 0) {
            html += '<div class="dm-empty" style="padding:32px;">暂无导出记录</div>';
        } else {
            for (const h of history) {
                const fmtColor = {
                    'coco':'var(--accent)', 'voc':'var(--success)', 'mask':'#a78bfa',
                    'csv':'var(--warn)', 'yolo':'var(--danger)',
                }[h.export_format] || 'var(--text-2)';
                html += `<div class="dm-card" style="padding:12px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="background:${fmtColor}22;color:${fmtColor};padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;text-transform:uppercase;">${escapeHtml(h.export_format)}</span>
                            <span style="color:var(--text-1);font-size:11px;">${h.image_count} 图像</span>
                        </div>
                        <span style="color:var(--text-2);font-size:9px;">${(h.created_at||'').slice(0,16)}</span>
                    </div>
                    <div style="color:var(--text-2);font-size:9px;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(h.export_path||'')}">${escapeHtml(h.export_path||'')}</div>
                </div>`;
            }
        }
        html += '</div></div></div>';
        container.innerHTML = html;

        const groupSelect = document.getElementById('dm-export-group');
        if (groupSelect) {
            groupSelect.onchange = () => updateExportLabelSets(groups);
            updateExportLabelSets(groups);
        }
    } catch (e) {
        container.innerHTML = `<div class="dm-empty"><div style="color:var(--danger);">加载失败: ${escapeHtml(e.message)}</div></div>`;
    }
}

function updateExportLabelSets(groups) {
    const groupId = document.getElementById('dm-export-group')?.value;
    const lsSelect = document.getElementById('dm-export-labelset');
    if (!groupId || !lsSelect) return;
    const group = groups.find(g => g.group_id === groupId);
    if (!group) return;
    lsSelect.innerHTML = (group.label_sets || []).map(ls =>
        `<option value="${ls.set_id}">${ls.set_name} (${ls.label_format})</option>`
    ).join('');
}

async function runDMExport() {
    const groupId = document.getElementById('dm-export-group')?.value;
    const lsId = document.getElementById('dm-export-labelset')?.value;
    const subset = document.getElementById('dm-export-subset')?.value || 'train';
    const format = document.querySelector('input[name="export-format"]:checked')?.value || 'yolo';
    const status = document.getElementById('dm-export-status');
    const btn = document.getElementById('dm-export-btn');

    if (!groupId || !lsId) {
        if (status) status.innerHTML = '<span style="color:var(--danger);">请选择数据集和标注集</span>';
        return;
    }

    if (btn) { btn.disabled = true; btn.textContent = '导出中...'; }
    if (status) status.innerHTML = '<span style="color:var(--accent);">正在导出...</span>';

    try {
        const endpointMap = { coco:'/export/coco', voc:'/export/voc', masks:'/export/masks', csv:'/export/csv' };
        const endpoint = endpointMap[format] || '/export';
        const body = { group_id: groupId, label_set_id: lsId, subset };
        if (format === 'yolo') body.format = 'yolo_zip';

        const res = await fetch(`${DM_API}${endpoint}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (format === 'yolo' && res.ok && res.headers.get('content-type')?.includes('zip')) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${groupId.split(/[\\\/]/).pop()}_${lsId}_${subset}.zip`;
            a.click();
            URL.revokeObjectURL(url);
            if (status) status.innerHTML = `<span style="color:var(--success);">YOLO ZIP 已下载</span>`;
        } else {
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Export failed');
            if (status) status.innerHTML = `<span style="color:var(--success);">导出完成! ${data.images||data.image_count||0} 张图像 → ${escapeHtml(data.export_path||data.path||'完成')}</span>`;
        }
    } catch (e) {
        if (status) status.innerHTML = `<span style="color:var(--danger);">导出失败: ${escapeHtml(e.message)}</span>`;
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '开始导出'; }
    }
}

// ══════════════════════════════════════════════════════════════
// Audit Log Tab
// ══════════════════════════════════════════════════════════════

async function renderDMAudit(container) {
    container.innerHTML = dmSkeletonDatasetList(6);
    try {
        const res = await fetch(`${DM_API}/audit_log?limit=100`);
        const logs = await res.json();

        let html = `
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
  <div class="dm-section-title" style="margin:0;">操作审计日志</div>
  <select id="dm-audit-filter" class="dm-select" style="width:auto;" onchange="filterDMAudit()">
    <option value="">全部类别</option>
    <option value="annotation">标注操作</option>
    <option value="export">导出操作</option>
    <option value="project">项目管理</option>
  </select>
</div>
<div class="dm-card" style="padding:0;overflow:hidden;">
  <table class="dm-audit-table">
    <thead><tr>
      <th>时间</th><th>操作</th><th>类别</th><th>文件</th><th>详情</th>
    </tr></thead>
    <tbody>`;

        if (logs.length === 0) {
            html += '<tr><td colspan="5" style="text-align:center;padding:24px;color:var(--text-2);">暂无日志记录</td></tr>';
        } else {
            for (const log of logs) {
                const catColor = { annotation:'var(--accent)', export:'var(--success)', project:'var(--warn)' }[log.category] || 'var(--text-2)';
                let details = '';
                try {
                    const d = typeof log.details === 'string' ? JSON.parse(log.details) : log.details;
                    details = Object.entries(d||{}).map(([k,v]) => `${k}:${v}`).join(' ');
                } catch { details = log.details || ''; }

                html += `<tr class="dm-audit-row" data-category="${escapeHtml(log.category)}">
                    <td style="color:var(--text-2);white-space:nowrap;font-size:10px;">${(log.timestamp||'').slice(0,19).replace('T',' ')}</td>
                    <td style="color:var(--text-1);">${escapeHtml(log.action)}</td>
                    <td><span style="color:${catColor};font-size:9px;font-weight:700;">${escapeHtml(log.category)}</span></td>
                    <td style="color:var(--text-2);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(log.filename||'')}">${escapeHtml(log.filename||'-')}</td>
                    <td style="color:var(--text-2);font-size:9px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(details)}">${escapeHtml(details||'-')}</td>
                </tr>`;
            }
        }
        html += '</tbody></table></div>';
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<div class="dm-empty"><div style="color:var(--danger);">加载失败: ${escapeHtml(e.message)}</div></div>`;
    }
}

function filterDMAudit() {
    const cat = document.getElementById('dm-audit-filter')?.value || '';
    document.querySelectorAll('.dm-audit-row').forEach(row => {
        row.style.display = !cat || row.dataset.category === cat ? '' : 'none';
    });
}

// ══════════════════════════════════════════════════════════════
// Projects Tab
// ══════════════════════════════════════════════════════════════

async function renderDMProjects(container) {
    container.innerHTML = dmSkeletonDatasetList(3);
    try {
        const [projectsRes, groupsRes] = await Promise.all([
            fetch(`${DM_API}/projects`),
            fetch(`${DM_API}/groups`),
        ]);
        const projects = await projectsRes.json();
        _dmGroupsCache = await groupsRes.json();

        let html = `
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
  <div class="dm-section-title" style="margin:0;">项目管理</div>
  <button class="dm-btn-primary" onclick="showCreateProjectForm()">+ 新建项目</button>
</div>

<div id="dm-create-project-form" style="display:none;margin-bottom:16px;">
  <div class="dm-card" style="border-color:rgba(107,147,255,0.3);">
    <div style="font-size:13px;font-weight:700;color:var(--text-0);margin-bottom:14px;">新建项目</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div>
        <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">项目名称 <span style="color:var(--danger);">*</span></label>
        <input type="text" id="dm-project-name" class="dm-input" placeholder="如：BALF细胞标注第一批" style="width:100%;">
      </div>
      <div>
        <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">项目状态</label>
        <select id="dm-project-status" class="dm-select" style="width:100%;">
          <option value="active">活跃</option>
          <option value="pending">待处理</option>
        </select>
      </div>
      <div style="grid-column:1/-1;">
        <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">项目描述</label>
        <textarea id="dm-project-desc" class="dm-input" placeholder="描述项目目标、标注要求等" rows="2" style="width:100%;resize:vertical;"></textarea>
      </div>
      <div style="grid-column:1/-1;">
        <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">标注指南 <span style="color:var(--text-2);font-weight:400;">（可选，供标注人员参考）</span></label>
        <textarea id="dm-project-guide" class="dm-input" placeholder="如：分割精度要求、类别判断标准、边界处理规则等" rows="3" style="width:100%;resize:vertical;"></textarea>
      </div>
      <div style="grid-column:1/-1;">
        <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:4px;">关联数据集 <span style="color:var(--text-2);font-weight:400;">（可多选）</span></label>
        <div style="display:flex;flex-wrap:wrap;gap:6px;max-height:100px;overflow-y:auto;padding:8px;background:var(--bg-3);border-radius:var(--radius-md);border:1px solid var(--border);">
          ${_dmGroupsCache.map(g => {
              const tc = g.train_count || 0;
              const vc = g.val_count || 0;
              return `<label style="display:flex;align-items:center;gap:5px;padding:4px 10px;background:var(--bg-2);border:1px solid var(--border);border-radius:6px;cursor:pointer;font-size:10px;color:var(--text-1);transition:border-color .2s;" onmouseenter="this.style.borderColor='var(--accent)'" onmouseleave="this.style.borderColor='var(--border)'">
                <input type="checkbox" class="dm-project-ds-cb" value="${escapeHtml(g.group_id)}" style="accent-color:var(--accent);">
                <span>${escapeHtml(g.group_name)}</span>
                <span style="color:var(--text-2);font-size:9px;">${tc+vc}图</span>
              </label>`;
          }).join('')}
        </div>
      </div>
    </div>
    <div style="display:flex;gap:8px;margin-top:14px;justify-content:flex-end;">
      <button class="btn btn-ghost" style="font-size:11px;padding:5px 16px;" onclick="document.getElementById('dm-create-project-form').style.display='none'">取消</button>
      <button class="dm-btn-primary" style="padding:5px 20px;" onclick="createDMProject()">创建项目</button>
    </div>
  </div>
</div>

<div id="dm-projects-list" style="display:flex;flex-direction:column;gap:12px;">`;

        if (projects.length === 0) {
            html += `<div class="dm-empty">
                <div class="dm-empty__art">   ┌─────────────┐
   │  📁 无项目  │
   └─────────────┘</div>
                <div>暂无项目，点击上方按钮创建</div>
            </div>`;
        } else {
            for (const proj of projects) {
                const stMap = {
                    active: { bg:'rgba(45,212,168,0.1)', color:'var(--success)', text:'活跃' },
                    archived: { bg:'rgba(116,132,165,0.1)', color:'var(--text-2)', text:'已归档' },
                    pending: { bg:'rgba(245,185,66,0.1)', color:'var(--warn)', text:'待处理' },
                };
                const st = stMap[proj.status] || stMap.active;
                const dsIds = proj.dataset_ids || [];
                const dsNames = dsIds.map(id => {
                    const g = _dmGroupsCache.find(gr => gr.group_id === id);
                    return g ? g.group_name : id.split(/[\\\/]/).pop();
                });
                const safePid = escapeHtml(proj.id).replace(/'/g, "\\'");
                html += `
<div class="dm-project-card dm-card dm-card--glow" data-project-id="${escapeHtml(proj.id)}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div style="flex:1;min-width:0;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="font-size:14px;font-weight:700;color:var(--text-0);">${escapeHtml(proj.name)}</span>
        <span style="padding:2px 8px;border-radius:999px;font-size:9px;font-weight:600;background:${st.bg};color:${st.color};">${st.text}</span>
      </div>
      ${proj.description ? `<div style="font-size:11px;color:var(--text-2);margin-bottom:6px;">${escapeHtml(proj.description)}</div>` : ''}
      <div style="display:flex;gap:12px;font-size:10px;color:var(--text-2);flex-wrap:wrap;">
        <span>数据集: <strong style="color:var(--text-1);">${proj.dataset_count||0}</strong></span>
        <span>创建: ${(proj.created_at||'').slice(0,10)}</span>
        <span>更新: ${(proj.updated_at||'').slice(0,10)}</span>
      </div>
      ${dsNames.length > 0 ? `<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">
        ${dsIds.map((id, i) => {
            const encGid = encodeURIComponent(id).replace(/'/g, "\\'");
            return `<span class="tag" style="cursor:pointer;display:flex;align-items:center;gap:4px;" onclick="dmSelectDataset('${encGid}')">
              ${escapeHtml(dsNames[i])}
              <span style="color:var(--danger);font-size:10px;cursor:pointer;margin-left:2px;" onclick="event.stopPropagation();dmRemoveDatasetFromProject('${safePid}','${encGid}')" title="移除关联">×</span>
            </span>`;
        }).join('')}
        <button class="tag" style="cursor:pointer;border:1px dashed var(--border);background:transparent;color:var(--text-2);font-size:9px;" onclick="dmAddDatasetToProject('${safePid}')">+ 添加</button>
      </div>` : `<div style="margin-top:6px;">
        <button class="tag" style="cursor:pointer;border:1px dashed var(--border);background:transparent;color:var(--text-2);font-size:9px;" onclick="dmAddDatasetToProject('${safePid}')">+ 关联数据集</button>
      </div>`}
    </div>
    <div style="display:flex;gap:4px;flex-shrink:0;flex-wrap:wrap;">
      <button class="btn btn-ghost" style="font-size:10px;padding:4px 10px;" onclick="_toggleProjectDashboard('${safePid}')">仪表盘</button>
      <button class="btn btn-ghost" style="font-size:10px;padding:4px 10px;" onclick="editDMProject('${safePid}')">编辑</button>
      <button class="btn btn-ghost" style="font-size:10px;padding:4px 10px;" onclick="archiveDMProject('${safePid}','${escapeHtml(proj.status)}')">
        ${proj.status === 'archived' ? '恢复' : '归档'}
      </button>
      <button class="btn btn-ghost" style="font-size:10px;padding:4px 10px;color:var(--danger);" onclick="deleteDMProject('${safePid}','${escapeHtml(proj.name).replace(/'/g, "\\'")}')">删除</button>
    </div>
  </div>
  <div id="proj-dashboard-${escapeHtml(proj.id)}" style="display:none;margin-top:12px;padding-top:12px;border-top:1px solid var(--border);"></div>
</div>`;
            }
        }

        html += '</div>';
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<div class="dm-empty"><div style="color:var(--danger);">加载失败: ${escapeHtml(e.message)}</div></div>`;
    }
}

function showCreateProjectForm() {
    const form = document.getElementById('dm-create-project-form');
    if (form) {
        form.style.display = form.style.display === 'none' ? '' : 'none';
        if (form.style.display !== 'none') document.getElementById('dm-project-name')?.focus();
    }
}

async function createDMProject() {
    const name = document.getElementById('dm-project-name')?.value?.trim();
    if (!name) { alert('请输入项目名称'); return; }
    const desc = document.getElementById('dm-project-desc')?.value?.trim() || '';
    const status = document.getElementById('dm-project-status')?.value || 'active';
    const guide = document.getElementById('dm-project-guide')?.value?.trim() || '';
    const selectedDs = [...document.querySelectorAll('.dm-project-ds-cb:checked')].map(cb => cb.value);
    try {
        const res = await fetch(`${DM_API}/projects`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description: desc }),
        });
        const project = await res.json();
        if (!res.ok) throw new Error(project.detail || 'Failed');

        const updates = {};
        if (status !== 'active') updates.status = status;
        if (guide) updates.annotation_guide = guide;
        if (Object.keys(updates).length > 0) {
            const uRes = await fetch(`${DM_API}/projects/${project.id}`, {
                method: 'PUT', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates),
            });
            if (!uRes.ok) console.warn('Project update partial fail:', await uRes.text().catch(() => ''));
        }

        let linkFails = 0;
        for (const dsId of selectedDs) {
            const lRes = await fetch(`${DM_API}/projects/${project.id}/datasets`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ group_id: dsId }),
            });
            if (!lRes.ok) linkFails++;
        }
        if (linkFails > 0) console.warn(`${linkFails} dataset link(s) failed`);
        if (typeof showStatus === 'function') showStatus('项目创建成功');
        renderDMProjects(document.getElementById('dm-content'));
    } catch (e) { alert('创建项目失败: ' + e.message); }
}

async function editDMProject(projectId) {
    try {
        const res = await fetch(`${DM_API}/projects/${projectId}`);
        const project = await res.json();
        if (!res.ok) throw new Error('Failed');

        const cardEl = document.querySelector(`[data-project-id="${projectId}"]`);
        if (!cardEl) return;
        const dashDiv = document.getElementById(`proj-dashboard-${projectId}`);
        if (dashDiv) dashDiv.style.display = 'none';

        let existingForm = cardEl.querySelector('.dm-project-edit-form');
        if (existingForm) { existingForm.remove(); return; }

        const form = document.createElement('div');
        form.className = 'dm-project-edit-form';
        form.style.cssText = 'margin-top:12px;padding-top:12px;border-top:1px solid var(--border);';
        form.innerHTML = `
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div>
              <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:3px;">项目名称</label>
              <input type="text" class="dm-input dm-edit-name" value="${escapeHtml(project.name)}" style="width:100%;">
            </div>
            <div>
              <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:3px;">状态</label>
              <select class="dm-select dm-edit-status" style="width:100%;">
                <option value="active" ${project.status==='active'?'selected':''}>活跃</option>
                <option value="pending" ${project.status==='pending'?'selected':''}>待处理</option>
                <option value="archived" ${project.status==='archived'?'selected':''}>已归档</option>
              </select>
            </div>
            <div style="grid-column:1/-1;">
              <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:3px;">描述</label>
              <textarea class="dm-input dm-edit-desc" rows="2" style="width:100%;resize:vertical;">${escapeHtml(project.description||'')}</textarea>
            </div>
            <div style="grid-column:1/-1;">
              <label style="font-size:10px;color:var(--text-2);display:block;margin-bottom:3px;">标注指南</label>
              <textarea class="dm-input dm-edit-guide" rows="3" style="width:100%;resize:vertical;">${escapeHtml(project.annotation_guide||'')}</textarea>
            </div>
          </div>
          <div style="display:flex;gap:8px;margin-top:10px;justify-content:flex-end;">
            <button class="btn btn-ghost" style="font-size:11px;padding:4px 14px;" onclick="this.closest('.dm-project-edit-form').remove()">取消</button>
            <button class="dm-btn-primary" style="font-size:11px;padding:4px 16px;" onclick="_saveProjectEdit('${escapeHtml(projectId)}',this)">保存</button>
          </div>`;
        cardEl.appendChild(form);
        form.querySelector('.dm-edit-name')?.focus();
    } catch (e) { alert('加载失败: ' + e.message); }
}

async function _saveProjectEdit(projectId, btnEl) {
    const form = btnEl.closest('.dm-project-edit-form');
    if (!form) return;
    const name = form.querySelector('.dm-edit-name')?.value?.trim();
    const status = form.querySelector('.dm-edit-status')?.value;
    const desc = form.querySelector('.dm-edit-desc')?.value?.trim();
    const guide = form.querySelector('.dm-edit-guide')?.value?.trim();
    if (!name) { alert('名称不能为空'); return; }
    try {
        const res = await fetch(`${DM_API}/projects/${projectId}`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description: desc || '', status, annotation_guide: guide || '' }),
        });
        if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${res.status}`); }
        if (typeof showStatus === 'function') showStatus('项目已更新');
        renderDMProjects(document.getElementById('dm-content'));
    } catch (e) { alert('更新失败: ' + e.message); }
}

async function deleteDMProject(projectId, name) {
    _dmConfirm('删除项目', `确定要永久删除项目「${name}」吗？\n\n此操作不可恢复。关联的数据集不会被删除。`, async () => {
        try {
            const res = await fetch(`${DM_API}/projects/${projectId}`, { method: 'DELETE' });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `删除失败 (HTTP ${res.status})`);
            }
            if (typeof showStatus === 'function') showStatus('项目已删除');
            renderDMProjects(document.getElementById('dm-content'));
        } catch (e) { alert('删除失败: ' + e.message); }
    });
}

async function dmRemoveDatasetFromProject(projectId, encodedGroupId) {
    const gid = decodeURIComponent(encodedGroupId);
    const gName = (_dmGroupsCache || []).find(g => g.group_id === gid)?.group_name || gid.split(/[\\\/]/).pop();
    _dmConfirm('移除数据集', `从项目中移除数据集「${gName}」？\n\n数据集本身不会受影响。`, async () => {
        try {
            const res = await fetch(`${DM_API}/projects/${projectId}/datasets/${encodeURIComponent(gid)}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('移除失败');
            renderDMProjects(document.getElementById('dm-content'));
        } catch (e) { alert('操作失败: ' + e.message); }
    });
}

async function dmAddDatasetToProject(projectId) {
    if (!_dmGroupsCache || _dmGroupsCache.length === 0) { alert('没有可用的数据集'); return; }
    const opts = _dmGroupsCache.map(g => `${g.group_name} (${(g.train_count||0)+(g.val_count||0)}图)`);
    _dmSelect('选择要关联的数据集', opts, async (idx) => {
        const gid = _dmGroupsCache[idx].group_id;
        try {
            const res = await fetch(`${DM_API}/projects/${projectId}/datasets`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ group_id: gid }),
            });
            if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || `HTTP ${res.status}`); }
            if (typeof showStatus === 'function') showStatus('数据集已关联到项目');
            renderDMProjects(document.getElementById('dm-content'));
        } catch (e) { alert('关联失败: ' + e.message); }
    });
}

async function _toggleProjectDashboard(projectId) {
    const div = document.getElementById(`proj-dashboard-${projectId}`);
    if (!div) return;
    if (div.style.display !== 'none') { div.style.display = 'none'; return; }
    div.style.display = '';
    div.innerHTML = '<div style="text-align:center;padding:12px;color:var(--text-2);font-size:11px;">加载仪表盘...</div>';
    try {
        const projRes = await fetch(`${DM_API}/projects/${projectId}`);
        const proj = await projRes.json();
        const dsIds = (proj.datasets || []).map(d => d.group_id);
        if (dsIds.length === 0 && (proj.dataset_ids || []).length > 0) {
            dsIds.push(...(proj.dataset_ids || []));
        }
        if (dsIds.length === 0) { div.innerHTML = '<div style="text-align:center;padding:12px;color:var(--text-2);">该项目未关联数据集</div>'; return; }

        let totalImages = 0, totalLabeled = 0, totalAnns = 0;
        const allClassDist = {};
        for (const gid of dsIds) {
            try {
                const sRes = await fetch(`${DM_API}/datasets/stats?group_id=${encodeURIComponent(gid)}`);
                const stats = await sRes.json();
                totalImages += stats.total_images || 0;
                totalLabeled += stats.labeled_images || 0;
                totalAnns += stats.total_annotations || 0;
                for (const [cls, cnt] of Object.entries(stats.class_distribution || {})) {
                    allClassDist[cls] = (allClassDist[cls] || 0) + cnt;
                }
            } catch {}
        }
        const pct = totalImages > 0 ? (totalLabeled / totalImages * 100).toFixed(1) : 0;
        const pctColor = pct > 80 ? 'var(--success)' : pct > 50 ? 'var(--warn)' : 'var(--danger)';
        const classSorted = Object.entries(allClassDist).sort((a, b) => b[1] - a[1]);
        const classTotal = classSorted.reduce((s, [, v]) => s + v, 0);

        let html = '';
        if (proj.annotation_guide) {
            html += `<div style="margin-bottom:12px;padding:10px;background:rgba(107,147,255,0.06);border:1px solid rgba(107,147,255,0.15);border-radius:6px;">
                <div style="font-size:10px;font-weight:600;color:var(--accent);margin-bottom:4px;">📋 标注指南</div>
                <div style="font-size:11px;color:var(--text-1);white-space:pre-wrap;line-height:1.5;">${escapeHtml(proj.annotation_guide)}</div>
            </div>`;
        }

        html += `<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;">
            <div style="background:var(--bg-3);border-radius:6px;padding:10px;text-align:center;">
                <div style="font-size:9px;color:var(--text-2);">数据集</div>
                <div style="font-size:18px;font-weight:700;color:var(--text-0);">${dsIds.length}</div></div>
            <div style="background:var(--bg-3);border-radius:6px;padding:10px;text-align:center;">
                <div style="font-size:9px;color:var(--text-2);">总图像</div>
                <div style="font-size:18px;font-weight:700;color:var(--text-0);">${totalImages}</div></div>
            <div style="background:var(--bg-3);border-radius:6px;padding:10px;text-align:center;">
                <div style="font-size:9px;color:var(--text-2);">标注进度</div>
                <div style="font-size:18px;font-weight:700;color:${pctColor};">${pct}%</div></div>
            <div style="background:var(--bg-3);border-radius:6px;padding:10px;text-align:center;">
                <div style="font-size:9px;color:var(--text-2);">总标注</div>
                <div style="font-size:18px;font-weight:700;color:var(--accent);">${totalAnns}</div></div>
        </div>`;

        html += `<div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;">
            <span style="font-size:10px;color:var(--text-2);line-height:26px;">快捷操作:</span>`;
        for (const gid of dsIds) {
            const gName = (_dmGroupsCache||[]).find(g=>g.group_id===gid)?.group_name || gid.split(/[\\\/]/).pop();
            const encGid = encodeURIComponent(gid).replace(/'/g, "\\'");
            html += `<button class="btn btn-ghost" style="font-size:9px;padding:3px 8px;" onclick="dmSelectDataset('${encGid}')" title="打开标注">📝 ${escapeHtml(gName)}</button>`;
        }
        html += `<button class="btn btn-ghost" style="font-size:9px;padding:3px 8px;" onclick="dmProjectExport('${escapeHtml(projectId)}')">📦 导出项目</button>`;
        html += '</div>';

        if (classSorted.length > 0) {
            html += '<div style="font-size:11px;font-weight:600;color:var(--text-1);margin-bottom:6px;">类别分布</div><div style="display:flex;flex-direction:column;gap:4px;">';
            classSorted.slice(0, 10).forEach(([name, count], i) => {
                const p = (count / classTotal * 100).toFixed(1);
                html += `<div style="display:flex;align-items:center;gap:6px;font-size:10px;">
                    <span style="width:70px;text-align:right;color:var(--text-1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(name)}</span>
                    <div style="flex:1;height:14px;background:var(--bg-3);border-radius:3px;overflow:hidden;">
                        <div style="width:${p}%;height:100%;background:${DM_BAR_COLORS[i%DM_BAR_COLORS.length]};border-radius:3px;min-width:2px;"></div></div>
                    <span style="min-width:50px;text-align:right;color:var(--text-2);">${count} (${p}%)</span></div>`;
            });
            html += '</div>';
        }
        div.innerHTML = html;
    } catch (e) {
        div.innerHTML = `<div style="color:var(--danger);text-align:center;padding:12px;font-size:11px;">加载失败: ${escapeHtml(e.message)}</div>`;
    }
}

function dmProjectExport(projectId) {
    switchDMTab('export');
}

async function archiveDMProject(projectId, currentStatus) {
    const newStatus = currentStatus === 'archived' ? 'active' : 'archived';
    try {
        await fetch(`${DM_API}/projects/${projectId}`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus }),
        });
        renderDMProjects(document.getElementById('dm-content'));
    } catch (e) { alert('操作失败: ' + e.message); }
}

// ══════════════════════════════════════════════════════════════
// Directory Browser Component (reusable)
// ══════════════════════════════════════════════════════════════

async function dmToggleBrowser(targetInputId) {
    const browserId = `dm-browser-${targetInputId}`;
    const container = document.getElementById(browserId);
    if (!container) return;
    if (container.style.display !== 'none') {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'block';
    const inp = document.getElementById(targetInputId);
    const startPath = inp?.value?.trim() || '';
    _dmBrowseTo(browserId, targetInputId, startPath);
}

async function _dmBrowseTo(browserId, targetInputId, dirPath) {
    const container = document.getElementById(browserId);
    if (!container) return;
    container.innerHTML = '<div style="padding:10px;color:var(--text-2);font-size:11px;">加载目录...</div>';
    try {
        const res = await fetch(`${DM_API}/datasets/browse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: dirPath }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        _dmRenderBrowser(browserId, targetInputId, data);
    } catch (err) {
        container.innerHTML = `<div style="padding:10px;color:var(--danger);font-size:11px;">加载失败: ${escapeHtml(err.message)}</div>`;
    }
}

function _dmBrowseNav(browserId, targetInputId, encodedPath) {
    _dmBrowseTo(browserId, targetInputId, decodeURIComponent(encodedPath));
}

function _dmBrowseSelect(browserId, targetInputId, encodedPath) {
    const inp = document.getElementById(targetInputId);
    if (inp) inp.value = decodeURIComponent(encodedPath);
    const br = document.getElementById(browserId);
    if (br) br.style.display = 'none';
}

function _dmRenderBrowser(browserId, targetInputId, data) {
    const container = document.getElementById(browserId);
    if (!container) return;
    const current = data.current || '';
    const parent = data.parent || '';
    const items = data.items || [];
    const enc = s => encodeURIComponent(s);

    let breadcrumb = '';
    if (current) {
        const sep = current.includes('/') ? '/' : '\\';
        const parts = current.replace(/\\/g, '/').split('/').filter(Boolean);
        let built = '';
        breadcrumb = '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:2px;margin-bottom:8px;font-size:10px;">';
        breadcrumb += `<span class="dm-bc-item" onclick="_dmBrowseNav('${browserId}','${targetInputId}','')" style="cursor:pointer;color:var(--accent);">🖥 根</span>`;
        for (let i = 0; i < parts.length; i++) {
            const p = parts[i];
            built += (i === 0 && current.match(/^[A-Z]:/i)) ? p + sep : p + sep;
            const pathForNav = built.replace(/[\/\\]$/, '');
            breadcrumb += `<span style="color:var(--text-2);">›</span>`;
            if (i === parts.length - 1) {
                breadcrumb += `<span style="color:var(--text-0);font-weight:600;">${escapeHtml(p)}</span>`;
            } else {
                breadcrumb += `<span class="dm-bc-item" onclick="_dmBrowseNav('${browserId}','${targetInputId}','${enc(pathForNav)}')" style="cursor:pointer;color:var(--accent);">${escapeHtml(p)}</span>`;
            }
        }
        breadcrumb += '</div>';
    }

    let itemsHtml = '';
    if (parent && current) {
        itemsHtml += `<div class="dm-browser-item" onclick="_dmBrowseNav('${browserId}','${targetInputId}','${enc(parent)}')">
            <span style="margin-right:6px;">⬆</span><span style="color:var(--text-1);">..</span>
        </div>`;
    }
    for (const item of items) {
        const imgBadge = item.img_count > 0 ? `<span style="font-size:9px;color:var(--accent);margin-left:auto;flex-shrink:0;">${item.img_count} 图</span>` : '';
        const arrow = item.has_children ? '<span style="color:var(--text-2);margin-left:4px;font-size:9px;">›</span>' : '';
        itemsHtml += `<div class="dm-browser-item" onclick="_dmBrowseNav('${browserId}','${targetInputId}','${enc(item.path)}')">
            <span style="margin-right:6px;">${item.img_count > 0 ? '📁' : '📂'}</span>
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;color:var(--text-0);" title="${escapeHtml(item.path)}">${escapeHtml(item.name)}</span>
            ${imgBadge}${arrow}
        </div>`;
    }
    if (items.length === 0 && current) {
        itemsHtml += '<div style="padding:8px 12px;color:var(--text-2);font-size:11px;text-align:center;">此目录下没有子目录</div>';
    }

    const selectBtn = current ? `<div style="display:flex;gap:6px;margin-top:8px;">
        <button type="button" class="dm-btn-primary" style="font-size:11px;padding:5px 14px;flex:1;" onclick="_dmBrowseSelect('${browserId}','${targetInputId}','${enc(current)}')">✓ 选择此目录</button>
        <button type="button" class="btn btn-ghost" style="font-size:11px;padding:5px 10px;" onclick="document.getElementById('${browserId}').style.display='none';">取消</button>
    </div>` : '';

    container.innerHTML = `
    <div style="background:var(--bg-3);border:1px solid var(--border);border-radius:var(--radius-md);padding:10px;max-height:250px;display:flex;flex-direction:column;">
        ${breadcrumb}
        <div style="flex:1;overflow-y:auto;min-height:0;display:flex;flex-direction:column;gap:1px;">
            ${itemsHtml}
        </div>
        ${selectBtn}
    </div>`;
}

// ══════════════════════════════════════════════════════════════
// Add Dataset
// ══════════════════════════════════════════════════════════════

function dmShowAddDataset() {
    const panel = document.getElementById('dm-add-dataset-panel');
    if (!panel) return;
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    if (panel.style.display === 'block') {
        const inp = document.getElementById('dm-add-ds-path');
        if (inp) { inp.value = ''; inp.focus(); }
        const nameInp = document.getElementById('dm-add-ds-name');
        if (nameInp) nameInp.value = '';
        const prev = document.getElementById('dm-add-ds-preview');
        if (prev) prev.style.display = 'none';
        const browser = document.getElementById('dm-browser-dm-add-ds-path');
        if (browser) browser.style.display = 'none';
    }
}

async function dmPreviewDataset() {
    const pathInp = document.getElementById('dm-add-ds-path');
    const preview = document.getElementById('dm-add-ds-preview');
    if (!pathInp || !preview) return;
    const path = pathInp.value.trim();
    if (!path) { alert('请输入目录路径'); return; }

    preview.style.display = 'block';
    preview.innerHTML = '<div style="color:var(--text-2);font-size:11px;padding:8px;">正在检测目录结构...</div>';

    try {
        const res = await fetch(`${DM_API}/datasets/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `检测失败 (HTTP ${res.status})`);
        }
        const info = await res.json();
        const structLabels = { yolo: 'YOLO 标准结构', simple_split: '简单 Train/Val 分割', flat: '扁平目录', unknown: '未识别' };
        const structColor = info.structure === 'unknown' ? 'var(--danger)' : 'var(--success)';
        preview.innerHTML = `
        <div style="background:var(--bg-3);border-radius:var(--radius-md);padding:10px;border:1px solid var(--border);">
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;font-size:11px;">
                <div><span style="color:var(--text-2);">结构类型:</span> <strong style="color:${structColor};">${structLabels[info.structure] || info.structure}</strong></div>
                <div><span style="color:var(--text-2);">图片数量:</span> <strong style="color:var(--text-0);">${info.img_count || 0}</strong></div>
                <div><span style="color:var(--text-2);">已有标注:</span> <strong style="color:${info.has_labels ? 'var(--success)' : 'var(--text-2)'};">${info.has_labels ? '是' : '否'}</strong></div>
                <div><span style="color:var(--text-2);">验证集:</span> <strong style="color:${info.has_val ? 'var(--success)' : 'var(--text-2)'};">${info.has_val ? '是' : '否'}</strong></div>
            </div>
            ${info.structure === 'unknown' ? '<div style="margin-top:8px;font-size:10px;color:var(--danger);">⚠ 未检测到图片文件，请确认路径正确</div>' : ''}
        </div>`;
    } catch (err) {
        preview.innerHTML = `<div style="color:var(--danger);font-size:11px;padding:8px;background:var(--danger-dim);border-radius:var(--radius-md);">检测失败: ${escapeHtml(err.message)}</div>`;
    }
}

async function dmAddDataset() {
    const pathInp = document.getElementById('dm-add-ds-path');
    const nameInp = document.getElementById('dm-add-ds-name');
    if (!pathInp) return;
    const path = pathInp.value.trim();
    const name = nameInp ? nameInp.value.trim() : '';
    if (!path) { alert('请输入目录路径'); return; }

    const overlay = _dmShowLoadingOverlay('正在添加数据集...');
    try {
        _dmSetLoadingStep(overlay, 1, '正在检测目录结构...');
        const res = await fetch(`${DM_API}/datasets/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, name }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `添加失败 (HTTP ${res.status})`);
        }
        const result = await res.json();

        _dmSetLoadingStep(overlay, 2, '正在刷新数据集列表...');
        _dmGroupsCache = null;
        DatasetManager._initialized = false;

        _dmHideLoadingOverlay(overlay);

        const statusText = result.status === 'already_exists' ? '数据集已存在，已刷新' : '数据集添加成功';
        if (typeof showStatus === 'function') showStatus(statusText);

        const panel = document.getElementById('dm-add-dataset-panel');
        if (panel) panel.style.display = 'none';
        const content = document.getElementById('dm-content');
        if (content) renderDMDatasets(content);
    } catch (err) {
        _dmHideLoadingOverlay(overlay);
        alert('添加数据集失败: ' + err.message);
    }
}

// ══════════════════════════════════════════════════════════════
// Dataset CRUD — Rename / Delete / Metadata
// ══════════════════════════════════════════════════════════════

async function dmRenameDataset(userDsId, currentName) {
    if (!userDsId) { alert('此数据集来自YAML配置，不支持重命名'); return; }
    _dmPrompt('重命名数据集', currentName, async (newName) => {
        if (newName === currentName) return;
        try {
            const res = await fetch(`${DM_API}/datasets/${encodeURIComponent(userDsId)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `重命名失败 (HTTP ${res.status})`);
            }
            if (typeof showStatus === 'function') showStatus('数据集已重命名');
            const content = document.getElementById('dm-content');
            if (content) renderDMDatasets(content);
        } catch (e) { alert('重命名失败: ' + e.message); }
    });
}

async function dmHideDataset(encodedGroupId, name, userDsId) {
    const gid = decodeURIComponent(encodedGroupId);
    const isUserDs = !!userDsId;
    const action = isUserDs ? '移除' : '隐藏';
    const hint = isUserDs
        ? '这只会取消注册，不会删除磁盘上的图片和标注文件。'
        : '隐藏后可从「添加数据集」重新恢复。磁盘文件不受影响。';
    _dmConfirm(`${action}数据集`, `确定要${action}数据集「${name}」吗？\n\n${hint}`, async () => {
        try {
            if (isUserDs) {
                const res = await fetch(`${DM_API}/datasets/${encodeURIComponent(userDsId)}`, { method: 'DELETE' });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `${action}失败 (HTTP ${res.status})`);
                }
            } else {
                const res = await fetch(`${DM_API}/datasets/exclude`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ group_id: gid }),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `${action}失败 (HTTP ${res.status})`);
                }
            }
            if (typeof showStatus === 'function') showStatus(`数据集已${action}`);
            const content = document.getElementById('dm-content');
            if (content) renderDMDatasets(content);
        } catch (e) { alert(`${action}失败: ` + e.message); }
    });
}

// ══════════════════════════════════════════════════════════════
// Label Set Creation (creates real directories on disk)
// ══════════════════════════════════════════════════════════════

function _dmPrompt(title, placeholder, callback) {
    let overlay = document.getElementById('dm-prompt-overlay');
    if (overlay) overlay.remove();
    overlay = document.createElement('div');
    overlay.id = 'dm-prompt-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:99999;display:flex;align-items:center;justify-content:center;';
    overlay.innerHTML = `
      <div style="background:var(--bg-2,#1e1e2e);border:1px solid var(--border,#444);border-radius:10px;padding:20px 24px;min-width:340px;max-width:480px;box-shadow:0 8px 32px rgba(0,0,0,0.4);">
        <div style="font-size:13px;font-weight:700;color:var(--text-0,#eee);margin-bottom:12px;">${title}</div>
        <input id="dm-prompt-input" type="text" placeholder="${placeholder}"
          style="width:100%;box-sizing:border-box;padding:8px 10px;border-radius:6px;border:1px solid var(--border,#555);background:var(--bg-3,#2a2a3e);color:var(--text-0,#eee);font-size:12px;outline:none;" />
        <div style="font-size:10px;color:var(--text-2,#888);margin-top:6px;">只能包含字母、数字、下划线和横线</div>
        <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:14px;">
          <button id="dm-prompt-cancel" class="btn btn-ghost" style="font-size:11px;padding:5px 14px;">取消</button>
          <button id="dm-prompt-ok" class="dm-btn-primary" style="font-size:11px;padding:5px 14px;">确认创建</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const inp = document.getElementById('dm-prompt-input');
    const okBtn = document.getElementById('dm-prompt-ok');
    const cancelBtn = document.getElementById('dm-prompt-cancel');
    inp.focus();
    const close = () => overlay.remove();
    cancelBtn.onclick = close;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    const submit = () => { const v = inp.value.trim(); close(); if (v) callback(v); };
    okBtn.onclick = submit;
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); if (e.key === 'Escape') close(); });
}

function _dmConfirm(title, message, callback) {
    let overlay = document.getElementById('dm-prompt-overlay');
    if (overlay) overlay.remove();
    overlay = document.createElement('div');
    overlay.id = 'dm-prompt-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:99999;display:flex;align-items:center;justify-content:center;';
    overlay.innerHTML = `
      <div style="background:var(--bg-2,#1e1e2e);border:1px solid var(--border,#444);border-radius:10px;padding:20px 24px;min-width:320px;max-width:480px;box-shadow:0 8px 32px rgba(0,0,0,0.4);">
        <div style="font-size:13px;font-weight:700;color:var(--text-0,#eee);margin-bottom:10px;">${title}</div>
        <div style="font-size:11px;color:var(--text-1,#ccc);margin-bottom:14px;white-space:pre-wrap;">${message}</div>
        <div style="display:flex;gap:8px;justify-content:flex-end;">
          <button id="dm-prompt-cancel" class="btn btn-ghost" style="font-size:11px;padding:5px 14px;">取消</button>
          <button id="dm-prompt-ok" class="dm-btn-primary" style="font-size:11px;padding:5px 14px;">确认</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const close = () => overlay.remove();
    document.getElementById('dm-prompt-cancel').onclick = close;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    document.getElementById('dm-prompt-ok').onclick = () => { close(); callback(); };
}

function _dmSelect(title, options, callback) {
    let overlay = document.getElementById('dm-prompt-overlay');
    if (overlay) overlay.remove();
    overlay = document.createElement('div');
    overlay.id = 'dm-prompt-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:99999;display:flex;align-items:center;justify-content:center;';
    const optHtml = options.map((o, i) => `<div class="dm-select-opt" data-idx="${i}" style="padding:8px 12px;border-radius:6px;cursor:pointer;font-size:11px;color:var(--text-1,#ccc);border:1px solid var(--border,#444);transition:all 0.15s;"
      onmouseover="this.style.background='var(--bg-3,#2a2a3e)';this.style.borderColor='var(--accent,#7aa2ff)'"
      onmouseout="this.style.background='';this.style.borderColor='var(--border,#444)'">${i+1}. ${o}</div>`).join('');
    overlay.innerHTML = `
      <div style="background:var(--bg-2,#1e1e2e);border:1px solid var(--border,#444);border-radius:10px;padding:20px 24px;min-width:300px;max-width:480px;box-shadow:0 8px 32px rgba(0,0,0,0.4);">
        <div style="font-size:13px;font-weight:700;color:var(--text-0,#eee);margin-bottom:12px;">${title}</div>
        <div style="display:flex;flex-direction:column;gap:6px;max-height:300px;overflow-y:auto;">${optHtml}</div>
        <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:14px;">
          <button id="dm-prompt-cancel" class="btn btn-ghost" style="font-size:11px;padding:5px 14px;">取消</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    const close = () => overlay.remove();
    document.getElementById('dm-prompt-cancel').onclick = close;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    overlay.querySelectorAll('.dm-select-opt').forEach(el => {
        el.onclick = () => { const idx = parseInt(el.dataset.idx); close(); callback(idx); };
    });
}

async function dmCreateLabelSet(encodedGroupId) {
    const gid = decodeURIComponent(encodedGroupId);
    _dmPrompt('新建标注集', 'labels_polygon', async (safeName) => {
        if (!/^[a-zA-Z0-9_\-]+$/.test(safeName)) {
            alert('目录名只能包含字母、数字、下划线和横线');
            return;
        }
        try {
            const res = await fetch(`${DM_API}/add_label_set`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ group_id: gid, dir_name: safeName }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `创建失败 (HTTP ${res.status})`);
            }
            const result = await res.json();
            if (typeof showStatus === 'function') showStatus(`标注集 "${result.set_name}" 创建成功，目录已在磁盘上生成`);
            const content = document.getElementById('dm-content');
            if (content) renderDMDatasets(content);
        } catch (e) { alert('创建标注集失败: ' + e.message); }
    });
}

// ══════════════════════════════════════════════════════════════
// File Manager — view and manage annotation files for a dataset
// ══════════════════════════════════════════════════════════════

async function dmToggleFileManager(encodedGroupId) {
    const gid = decodeURIComponent(encodedGroupId);
    const safe = encodeURIComponent(gid).replace(/%/g, '_');
    const div = document.getElementById(`dm-files-${safe}`);
    if (!div) return;
    if (div.style.display !== 'none') { div.style.display = 'none'; return; }
    div.style.display = '';
    div.innerHTML = '<div style="text-align:center;padding:10px;color:var(--text-2);font-size:11px;">加载文件信息...</div>';

    const group = (_dmGroupsCache || []).find(g => g.group_id === gid);
    if (!group) { div.innerHTML = '<div style="color:var(--danger);font-size:11px;">未找到数据集</div>'; return; }

    try {
        const res = await fetch(`${DM_API}/datasets/file_info?group_id=${encodeURIComponent(gid)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const info = await res.json();
        _renderFileManager(div, info, gid, group);
    } catch (e) {
        div.innerHTML = `<div style="color:var(--danger);font-size:11px;">加载失败: ${escapeHtml(e.message)}</div>`;
    }
}

function _renderFileManager(div, info, gid, group) {
    const encGid = encodeURIComponent(gid).replace(/'/g, "\\'");
    let html = `<div style="font-size:11px;font-weight:600;color:var(--text-1);margin-bottom:8px;">📁 文件系统详情</div>`;

    html += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">
        <div style="background:var(--bg-3);border-radius:6px;padding:8px;text-align:center;">
            <div style="font-size:9px;color:var(--text-2);">训练图像</div>
            <div style="font-size:16px;font-weight:700;color:var(--text-0);">${info.train_images || 0}</div>
        </div>
        <div style="background:var(--bg-3);border-radius:6px;padding:8px;text-align:center;">
            <div style="font-size:9px;color:var(--text-2);">验证图像</div>
            <div style="font-size:16px;font-weight:700;color:var(--text-0);">${info.val_images || 0}</div>
        </div>
        <div style="background:var(--bg-3);border-radius:6px;padding:8px;text-align:center;">
            <div style="font-size:9px;color:var(--text-2);">磁盘路径</div>
            <div style="font-size:9px;color:var(--text-1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(info.base_path||'')}">${escapeHtml((info.base_path||'').split(/[\\\/]/).slice(-2).join('/'))}</div>
        </div>
    </div>`;

    if (info.label_sets && info.label_sets.length > 0) {
        html += '<div style="font-size:10px;font-weight:600;color:var(--text-1);margin-bottom:6px;">标注集目录:</div>';
        html += '<div style="display:flex;flex-direction:column;gap:4px;margin-bottom:10px;">';
        for (const ls of info.label_sets) {
            const trainPct = info.train_images > 0 ? Math.round((ls.train_labels / info.train_images) * 100) : 0;
            html += `<div style="display:flex;align-items:center;gap:8px;font-size:10px;padding:4px 8px;background:var(--bg-3);border-radius:4px;">
                <span style="font-weight:600;color:var(--text-0);min-width:80px;">${escapeHtml(ls.set_name)}</span>
                <span style="color:var(--text-2);">格式: ${escapeHtml(ls.format)}</span>
                <span style="color:var(--text-2);">Train标注: <strong style="color:var(--text-1);">${ls.train_labels}</strong>/${info.train_images}</span>
                <div style="flex:1;height:6px;background:var(--bg-1);border-radius:3px;overflow:hidden;min-width:40px;">
                    <div style="height:100%;background:${trainPct>80?'var(--success)':trainPct>40?'var(--warn)':'var(--danger)'};width:${trainPct}%;border-radius:3px;"></div>
                </div>
                <span style="color:var(--text-2);min-width:30px;text-align:right;">${trainPct}%</span>
            </div>`;
        }
        html += '</div>';
    }

    html += `<div style="display:flex;gap:6px;flex-wrap:wrap;">
        <button class="btn btn-ghost" style="font-size:9px;padding:3px 10px;" onclick="dmCreateLabelSet('${encGid}')">+ 新建标注集</button>
        <button class="btn btn-ghost" style="font-size:9px;padding:3px 10px;" onclick="dmInitAnnotations('${encGid}')">初始化空标注</button>
        <button class="btn btn-ghost" style="font-size:9px;padding:3px 10px;" onclick="dmGenerateBboxSet('${encGid}')">生成 BBox 标注集</button>
    </div>`;

    div.innerHTML = html;
}

async function dmInitAnnotations(encodedGroupId) {
    const gid = decodeURIComponent(encodedGroupId);
    const group = (_dmGroupsCache || []).find(g => g.group_id === gid);
    if (!group || !group.label_sets || group.label_sets.length === 0) {
        alert('请先创建标注集');
        return;
    }
    const doInit = async (lsId) => {
        _dmConfirm('初始化空标注', '将为所有未标注的图像创建空的 .txt 标注文件。\n这样可以确保标注目录结构完整。', async () => {
            try {
                const res = await fetch(`${DM_API}/datasets/init_annotations`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ group_id: gid, label_set_id: lsId }),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `初始化失败 (HTTP ${res.status})`);
                }
                const result = await res.json();
                if (typeof showStatus === 'function') showStatus(`已创建 ${result.created} 个空标注文件`);
                dmToggleFileManager(encodedGroupId);
                dmToggleFileManager(encodedGroupId);
            } catch (e) { alert('初始化失败: ' + e.message); }
        });
    };
    if (group.label_sets.length === 1) {
        doInit(group.label_sets[0].set_id);
    } else {
        const opts = group.label_sets.map(ls => `${ls.set_name} (${ls.label_format})`);
        _dmSelect('选择要初始化的标注集', opts, idx => doInit(group.label_sets[idx].set_id));
    }
}

async function dmGenerateBboxSet(encodedGroupId) {
    const gid = decodeURIComponent(encodedGroupId);
    const group = (_dmGroupsCache || []).find(g => g.group_id === gid);
    if (!group) return;
    const polygonSets = (group.label_sets || []).filter(ls => ls.label_format === 'polygon');
    if (polygonSets.length === 0) { alert('没有可用的多边形标注集，无法生成 BBox'); return; }
    const doGenerate = (srcId) => {
        _dmPrompt('BBox 标注集名称', 'auto_bbox', async (targetName) => {
            try {
                const res = await fetch(`${DM_API}/generate_bbox_set`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ group_id: gid, source_set_id: srcId, target_name: targetName }),
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `生成失败 (HTTP ${res.status})`);
                }
                const result = await res.json();
                if (typeof showStatus === 'function') showStatus(`BBox标注集已生成: ${result.files}文件, ${result.boxes}框`);
                const content = document.getElementById('dm-content');
                if (content) renderDMDatasets(content);
            } catch (e) { alert('生成BBox失败: ' + e.message); }
        });
    };
    if (polygonSets.length === 1) {
        doGenerate(polygonSets[0].set_id);
    } else {
        const opts = polygonSets.map(ls => ls.set_name);
        _dmSelect('选择源标注集', opts, idx => doGenerate(polygonSets[idx].set_id));
    }
}
