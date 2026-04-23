// ══════════════════════════════════════════════════════════════
// Enhanced Dataset Manager
// ══════════════════════════════════════════════════════════════

const DatasetManager = {
    // State
    datasets: [],
    selectedDatasets: new Set(),
    currentFilter: {
        status: null,
        priority: null,
        starred: false,
        tag: null,
        search: '',
        sortBy: 'name',
        sortOrder: 'asc'
    },
    summary: null,
    allTags: [],
    
    // DOM Elements cache
    els: {},
    
    // Initialize
    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadSummary();
        this.loadDatasets();
    },
    
    cacheElements() {
        this.els.modal = document.getElementById('dataset-manager-modal');
        this.els.list = document.getElementById('ds-list');
        this.els.empty = document.getElementById('ds-list-empty');
        this.els.addPath = document.getElementById('ds-add-path');
        this.els.addName = document.getElementById('ds-add-name');
        this.els.browseUp = document.getElementById('ds-browse-up');
        this.els.browseSelect = document.getElementById('ds-browse-select');
        this.els.browseClose = document.getElementById('ds-browse-close');
        this.els.addBtn = document.getElementById('ds-add-btn');
        this.els.browseBtn = document.getElementById('ds-browse-btn');
        this.els.detectResult = document.getElementById('ds-detect-result');
        this.els.browsePanel = document.getElementById('ds-browse-panel');
        this.els.browseList = document.getElementById('ds-browse-list');
        this.els.browseBreadcrumb = document.getElementById('ds-browse-breadcrumb');
        this.els.toast = document.getElementById('ds-modal-toast');
        this.els.filterBar = document.getElementById('ds-filter-bar');
        this.els.selectAll = document.getElementById('ds-select-all');
        this.els.bulkActions = document.getElementById('ds-bulk-actions');
        this.els.selectedCount = document.getElementById('ds-selected-count');
        this.els.summary = document.getElementById('ds-summary');
    },
    
    bindEvents() {
        // Add button - adds the dataset
        if (this.els.addBtn) {
            this.els.addBtn.addEventListener('click', () => this.addDataset());
        }
        
        // Browse button
        if (this.els.browseBtn) {
            this.els.browseBtn.addEventListener('click', () => this.openBrowse());
        }
        
        // Browse panel buttons
        if (this.els.browseUp) {
            this.els.browseUp.addEventListener('click', () => this.browseUp());
        }
        if (this.els.browseSelect) {
            this.els.browseSelect.addEventListener('click', () => this.selectBrowsePath());
        }
        if (this.els.browseClose) {
            this.els.browseClose.addEventListener('click', () => this.closeBrowse());
        }
        
        // Path input change - detect dataset
        if (this.els.addPath) {
            this.els.addPath.addEventListener('change', () => this.detectPath());
            this.els.addPath.addEventListener('blur', () => this.detectPath());
        }

        // Select all checkbox
        if (this.els.selectAll) {
            this.els.selectAll.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.datasets.forEach(ds => this.selectedDatasets.add(ds.group_id));
                } else {
                    this.selectedDatasets.clear();
                }
                this.updateSelectedCount();
                this.render();
            });
        }
        
        // Bulk actions
        if (this.els.bulkActions) {
            this.els.bulkActions.addEventListener('change', (e) => {
                const action = e.target.value;
                if (action) {
                    this.executeBulkAction(action);
                    e.target.value = '';
                }
            });
        }

        // Filter bar
        if (this.els.filterBar) {
            this.els.filterBar.addEventListener('click', (e) => {
                const chip = e.target.closest('.ds-filter-chip');
                if (!chip) return;
                const filterType = chip.dataset.filter;
                const value = chip.dataset.value;

                if (filterType === 'starred') {
                    chip.classList.toggle('active');
                    this.currentFilter.starred = chip.classList.contains('active');
                } else {
                    chip.parentElement.querySelectorAll('.ds-filter-chip').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    this.currentFilter[filterType] = value || null;
                }
                this.loadDatasets();
            });
        }
    },
    
    // API Methods
    async loadDatasets() {
        // Show loading state
        if (this.els.list) {
            this.els.list.innerHTML = '<div style="text-align:center; padding:20px; color:var(--text-2);"><div class="loading-spinner" style="display:inline-block; width:20px; height:20px; border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite;"></div><div style="margin-top:8px; font-size:12px;">加载数据集...</div></div>';
        }
        try {
            const params = new URLSearchParams();
            if (this.currentFilter.status) params.append('status', this.currentFilter.status);
            if (this.currentFilter.priority) params.append('priority', this.currentFilter.priority);
            if (this.currentFilter.tag) params.append('tag', this.currentFilter.tag);
            params.append('sort_by', this.currentFilter.sortBy);
            params.append('sort_order', this.currentFilter.sortOrder);

            const res = await fetch(`${API}/datasets/enhanced?${params}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            this.datasets = await res.json();

            // Apply search filter locally for responsiveness
            if (this.currentFilter.search) {
                const searchLower = this.currentFilter.search.toLowerCase();
                this.datasets = this.datasets.filter(ds =>
                    ds.group_name.toLowerCase().includes(searchLower) ||
                    (ds.description && ds.description.toLowerCase().includes(searchLower))
                );
            }
            // Apply starred filter locally
            if (this.currentFilter.starred) {
                this.datasets = this.datasets.filter(ds => ds.starred);
            }

            this.render();
        } catch (e) {
            console.error('Failed to load datasets:', e);
            if (this.els.list) {
                this.els.list.innerHTML = '<div style="text-align:center; padding:20px; color:#f87171; cursor:pointer;" onclick="DatasetManager.loadDatasets()"><div style="font-size:13px;">加载失败</div><div style="font-size:11px; margin-top:4px; color:var(--text-2);">点击重试</div></div>';
            }
            showStatus('加载数据集失败', true);
        }
    },
    
    async loadSummary() {
        try {
            const res = await fetch(`${API}/datasets/summary`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            this.summary = await res.json();
            this.allTags = this.summary.all_tags || [];
            this.renderSummary();
        } catch (e) {
            console.error('Failed to load summary:', e);
        }
    },
    
    async updateMetadata(groupId, updates) {
        try {
            const res = await fetch(`${API}/datasets/${groupId}/metadata`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates)
            });
            if (!res.ok) throw new Error('Update failed');
            showStatus('元数据已更新');
            this.loadDatasets();
        } catch (e) {
            console.error('Failed to update metadata:', e);
            showStatus('更新失败', true);
        }
    },
    
    async executeBulkAction(action) {
        if (this.selectedDatasets.size === 0) {
            showStatus('请先选择数据集', true);
            return;
        }
        
        const groupIds = Array.from(this.selectedDatasets);
        let body = { action, group_ids: groupIds };
        
        if (action === 'update_tags') {
            const tags = prompt('输入标签（用逗号分隔）:');
            if (!tags) return;
            body.tags = tags.split(',').map(t => t.trim()).filter(t => t);
        }
        if (action === 'update_status') {
            const st = prompt('选择状态: active / archived / pending');
            if (!st || !['active', 'archived', 'pending'].includes(st)) return;
            body.status = st;
        }
        if (action === 'update_priority') {
            const pr = prompt('选择优先级: high / normal / low');
            if (!pr || !['high', 'normal', 'low'].includes(pr)) return;
            body.priority = pr;
        }
        
        try {
            showLoading(true, '执行中...');
            const res = await fetch(`${API}/datasets/bulk_action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const result = await res.json();
            
            if (result.success.length > 0) {
                showStatus(`成功处理 ${result.success.length} 个数据集`);
            }
            if (result.failed.length > 0) {
                console.error('Failed:', result.failed);
                showStatus(`${result.failed.length} 个数据集处理失败`, true);
            }
            
            this.selectedDatasets.clear();
            this.loadDatasets();
            this.loadSummary();
        } catch (e) {
            console.error('Bulk action failed:', e);
            showStatus('操作失败', true);
        } finally {
            showLoading(false);
        }
    },
    
    // Rendering
    render() {
        if (!this.els.list) return;

        if (this.datasets.length === 0) {
            this.els.list.innerHTML = '<div class="empty-state">没有找到匹配的数据集</div>';
            return;
        }

        this.els.list.innerHTML = this.datasets.map(ds => this.renderDatasetCard(ds)).join('');

        // Bind individual checkbox events
        this.els.list.querySelectorAll('.dataset-select').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const groupId = e.target.dataset.groupId;
                if (e.target.checked) {
                    this.selectedDatasets.add(groupId);
                } else {
                    this.selectedDatasets.delete(groupId);
                }
                this.updateSelectAllState();
                this.updateSelectedCount();
            });
        });
    },
    
    renderDatasetCard(ds) {
        const isSelected = this.selectedDatasets.has(ds.group_id);
        const progressPercent = Math.round(ds.label_progress * 100);
        const progressColor = progressPercent >= 90 ? 'var(--success)' :
                              progressPercent >= 50 ? 'var(--warn)' : 'var(--accent)';
        const gid = ds.group_id.replace(/'/g, "\\'");
        const statusLabels = { active: '活跃', archived: '归档', pending: '待处理' };
        const priorityLabels = { high: '高', normal: '正常', low: '低' };

        return `
            <div class="dataset-card ${isSelected ? 'selected' : ''}" data-group-id="${ds.group_id}">
                <div class="dataset-card-header">
                    <button class="ds-star-btn ${ds.starred ? 'starred' : ''}" onclick="DatasetManager.toggleStar('${gid}')" title="收藏">&#9733;</button>
                    <input type="checkbox" class="dataset-select" data-group-id="${ds.group_id}"
                           ${isSelected ? 'checked' : ''}>
                    <span class="dataset-name" title="${ds.group_id}">${ds.group_name}</span>
                    <span class="ds-priority-dot priority-${ds.priority}" onclick="DatasetManager.cyclePriority('${gid}')" title="优先级: ${priorityLabels[ds.priority] || ds.priority}"></span>
                    <span class="dataset-status status-${ds.status}" onclick="DatasetManager.cycleStatus('${gid}')" style="cursor:pointer" title="点击切换状态">${statusLabels[ds.status] || ds.status}</span>
                </div>

                <div class="dataset-stats">
                    <span class="stat-item">训练: ${ds.train_count}</span>
                    ${ds.has_val ? `<span class="stat-item">验证: ${ds.val_count}</span>` : ''}
                    <span class="stat-item">类别: ${ds.nc}</span>
                    ${ds.assigned_to ? `<span class="stat-item ds-assigned">&#128100; ${ds.assigned_to}</span>` : ''}
                    <span class="stat-item ds-updated">${this.formatTimeAgo(ds.updated_at)}</span>
                </div>

                <div class="dataset-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressPercent}%; background: ${progressColor};"></div>
                    </div>
                    <span class="progress-text">${progressPercent}%</span>
                </div>

                ${ds.names && Object.keys(ds.names).length > 0 && !(Object.keys(ds.names).length === 1 && Object.values(ds.names)[0] === 'object') ? `
                    <div class="dataset-classes" style="display:flex; flex-wrap:wrap; gap:3px; margin-bottom:6px;">
                        ${Object.entries(ds.names).map(([i, n]) => {
                            const hue = Math.round(parseInt(i) * 360 / Object.keys(ds.names).length);
                            return `<span style="font-size:9px; padding:1px 5px; border-radius:999px; background:hsl(${hue},50%,20%); color:hsl(${hue},80%,75%); border:1px solid hsl(${hue},40%,30%);">${n}</span>`;
                        }).join('')}
                    </div>
                ` : ''}

                ${ds.description ? `<div class="dataset-description">${ds.description}</div>` : ''}
                ${ds.notes ? `<div class="dataset-notes-preview" title="${ds.notes.replace(/"/g, '&quot;')}">&#128221; ${ds.notes.length > 60 ? ds.notes.substring(0, 60) + '...' : ds.notes}</div>` : ''}

                <div class="dataset-tags">
                    ${ds.tags.map(tag => `<span class="tag">${tag}<button class="tag-remove" onclick="event.stopPropagation(); DatasetManager.removeTag('${gid}', '${tag.replace(/'/g, "\\'")}')">&times;</button></span>`).join('')}
                    <button class="tag-add-btn" onclick="DatasetManager.showAddTag('${gid}')" title="添加标签">+</button>
                </div>

                <div class="dataset-actions">
                    <button class="btn btn-sm btn-ghost" onclick="DatasetManager.editMetadata('${gid}')">编辑</button>
                    <button class="btn btn-sm btn-ghost" onclick="DatasetManager.runQualityCheck('${gid}')">质检</button>
                    <button class="btn btn-sm btn-ghost" onclick="DatasetManager.showVisualization('${gid}')">可视化</button>
                    <button class="btn btn-sm btn-ghost" onclick="DatasetManager.previewSplit('${gid}')">分割</button>
                    <button class="btn btn-sm btn-primary" onclick="selectGroup('${gid}')">打开</button>
                </div>
            </div>
        `;
    },
    
    renderSummary() {
        if (!this.els.summary || !this.summary) return;
        
        const s = this.summary;
        this.els.summary.innerHTML = `
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-value">${s.total_datasets}</span>
                    <span class="summary-label">数据集</span>
                </div>
                <div class="summary-item">
                    <span class="summary-value">${s.total_images.toLocaleString()}</span>
                    <span class="summary-label">图片</span>
                </div>
                <div class="summary-item">
                    <span class="summary-value">${Math.round(s.avg_label_progress * 100)}%</span>
                    <span class="summary-label">平均进度</span>
                </div>
            </div>
        `;
    },
    
    updateSelectAllState() {
        if (this.els.selectAll) {
            const allSelected = this.datasets.length > 0 &&
                               this.datasets.every(ds => this.selectedDatasets.has(ds.group_id));
            this.els.selectAll.checked = allSelected;
            this.els.selectAll.indeterminate = !allSelected && this.selectedDatasets.size > 0;
        }
    },

    updateSelectedCount() {
        if (this.els.selectedCount) {
            const n = this.selectedDatasets.size;
            this.els.selectedCount.textContent = n > 0 ? `已选 ${n} 项` : '';
        }
    },

    formatTimeAgo(isoString) {
        if (!isoString) return '';
        const diff = Date.now() - new Date(isoString).getTime();
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return '刚刚';
        if (mins < 60) return `${mins}分钟前`;
        const hrs = Math.floor(mins / 60);
        if (hrs < 24) return `${hrs}小时前`;
        const days = Math.floor(hrs / 24);
        if (days < 30) return `${days}天前`;
        return new Date(isoString).toLocaleDateString('zh-CN');
    },

    // ── Quick inline actions (optimistic UI) ──

    toggleStar(groupId) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        const newVal = !ds.starred;
        ds.starred = newVal;
        this.render();
        this.updateMetadata(groupId, { starred: newVal });
    },

    cycleStatus(groupId) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        const order = ['active', 'archived', 'pending'];
        const next = order[(order.indexOf(ds.status) + 1) % order.length];
        ds.status = next;
        this.render();
        this.updateMetadata(groupId, { status: next });
    },

    cyclePriority(groupId) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        const order = ['normal', 'high', 'low'];
        const next = order[(order.indexOf(ds.priority) + 1) % order.length];
        ds.priority = next;
        this.render();
        this.updateMetadata(groupId, { priority: next });
    },

    removeTag(groupId, tag) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        ds.tags = ds.tags.filter(t => t !== tag);
        this.render();
        this.updateMetadata(groupId, { tags: ds.tags });
    },

    showAddTag(groupId) {
        document.querySelectorAll('.ds-tag-input-wrap').forEach(el => el.remove());
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;

        const card = this.els.list.querySelector(`[data-group-id="${CSS.escape(groupId)}"]`);
        if (!card) return;
        const tagsDiv = card.querySelector('.dataset-tags');
        const addBtn = tagsDiv.querySelector('.tag-add-btn');

        const wrap = document.createElement('span');
        wrap.className = 'ds-tag-input-wrap';
        wrap.innerHTML = `<input type="text" list="ds-tag-suggestions" class="ds-tag-input"
            placeholder="标签..." style="font-size:10px; width:72px; padding:1px 6px; border:1px solid var(--accent); border-radius:999px; background:var(--bg-0); color:var(--text-0); outline:none;">
            <datalist id="ds-tag-suggestions">${(this.allTags || []).filter(t => !ds.tags.includes(t)).map(t => '<option value="' + t + '">').join('')}</datalist>`;
        tagsDiv.insertBefore(wrap, addBtn);

        const input = wrap.querySelector('input');
        input.focus();
        const submit = () => {
            const val = input.value.trim();
            if (val && !ds.tags.includes(val)) {
                ds.tags.push(val);
                this.render();
                this.updateMetadata(groupId, { tags: ds.tags });
            } else {
                wrap.remove();
            }
        };
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') submit();
            if (e.key === 'Escape') wrap.remove();
        });
        input.addEventListener('blur', submit);
    },

    // Actions
    editMetadata(groupId) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        
        // Create modal for editing metadata
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 500px;">
                <h3>编辑数据集元数据</h3>
                <div class="form-group">
                    <label>描述</label>
                    <textarea id="edit-description" rows="3">${ds.description || ''}</textarea>
                </div>
                <div class="form-group">
                    <label>标签（逗号分隔）</label>
                    <input type="text" id="edit-tags" value="${ds.tags.join(', ')}">
                </div>
                <div class="form-group">
                    <label>状态</label>
                    <select id="edit-status">
                        <option value="active" ${ds.status === 'active' ? 'selected' : ''}>活跃</option>
                        <option value="archived" ${ds.status === 'archived' ? 'selected' : ''}>归档</option>
                        <option value="pending" ${ds.status === 'pending' ? 'selected' : ''}>待处理</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>优先级</label>
                    <select id="edit-priority">
                        <option value="high" ${ds.priority === 'high' ? 'selected' : ''}>高</option>
                        <option value="normal" ${ds.priority === 'normal' ? 'selected' : ''}>正常</option>
                        <option value="low" ${ds.priority === 'low' ? 'selected' : ''}>低</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>备注</label>
                    <textarea id="edit-notes" rows="2">${ds.notes || ''}</textarea>
                </div>
                <div class="modal-actions">
                    <button class="btn btn-primary" id="save-metadata">保存</button>
                    <button class="btn btn-ghost" id="cancel-metadata">取消</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
        
        modal.querySelector('#cancel-metadata').addEventListener('click', () => modal.remove());
        
        modal.querySelector('#save-metadata').addEventListener('click', () => {
            const updates = {
                description: modal.querySelector('#edit-description').value,
                tags: modal.querySelector('#edit-tags').value.split(',').map(t => t.trim()).filter(t => t),
                status: modal.querySelector('#edit-status').value,
                priority: modal.querySelector('#edit-priority').value,
                notes: modal.querySelector('#edit-notes').value,
            };
            this.updateMetadata(groupId, updates);
            modal.remove();
        });
    },
    
    // Data Quality Check
    async runQualityCheck(groupId) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        
        showLoading(true, '正在检查数据质量...');
        
        try {
            const res = await fetch(`${API}/datasets/quality_check`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    group_id: groupId,
                    label_set_id: ds.label_sets[0]?.set_id || 'default',
                    subset: 'train',
                    checks: ['all']
                })
            });
            
            const result = await res.json();
            this.showQualityReport(result);
        } catch (e) {
            console.error('Quality check failed:', e);
            showStatus('质量检查失败', true);
        } finally {
            showLoading(false);
        }
    },
    
    showQualityReport(result) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        
        const issuesHtml = result.issues.map(issue => `
            <div class="quality-issue severity-${issue.severity}">
                <h4>${issue.title} (${issue.count})</h4>
                <p>${issue.description}</p>
                ${issue.items.length > 0 ? `
                    <details>
                        <summary>查看详情 (${issue.items.length} 项)</summary>
                        <ul class="issue-items">
                            ${issue.items.slice(0, 20).map(item => `
                                <li>${item.filename}${item.reason ? ` - ${item.reason}` : ''}</li>
                            `).join('')}
                        </ul>
                    </details>
                ` : ''}
            </div>
        `).join('');
        
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 700px; max-height: 80vh; overflow-y: auto;">
                <h3>📊 数据质量报告 - ${result.group_name}</h3>
                
                <div class="quality-summary">
                    <div class="summary-grid">
                        <div class="summary-item">
                            <span class="summary-value">${result.summary.total_checked}</span>
                            <span class="summary-label">检查文件</span>
                        </div>
                        <div class="summary-item critical">
                            <span class="summary-value">${result.summary.severity_counts.critical}</span>
                            <span class="summary-label">严重问题</span>
                        </div>
                        <div class="summary-item warning">
                            <span class="summary-value">${result.summary.severity_counts.warning}</span>
                            <span class="summary-label">警告</span>
                        </div>
                        <div class="summary-item info">
                            <span class="summary-value">${result.summary.severity_counts.info}</span>
                            <span class="summary-label">提示</span>
                        </div>
                    </div>
                </div>
                
                ${issuesHtml || '<p class="empty-state">✅ 未发现质量问题</p>'}
                
                <div class="modal-actions">
                    <button class="btn btn-primary" id="close-quality">关闭</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        modal.querySelector('#close-quality').addEventListener('click', () => modal.remove());
    },
    
    // Data Split Tool
    async previewSplit(groupId) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 500px;">
                <h3>📂 数据分割工具</h3>
                
                <div class="form-group">
                    <label>分割策略</label>
                    <select id="split-strategy">
                        <option value="random">随机分割</option>
                        <option value="stratified">分层分割（保持类别分布）</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>训练集比例</label>
                    <input type="range" id="train-ratio" min="0.5" max="0.9" step="0.05" value="0.8">
                    <span id="train-ratio-value">80%</span>
                </div>
                
                <div class="form-group">
                    <label>验证集比例</label>
                    <input type="range" id="val-ratio" min="0.05" max="0.3" step="0.05" value="0.1">
                    <span id="val-ratio-value">10%</span>
                </div>
                
                <div class="form-group">
                    <label>测试集比例</label>
                    <input type="range" id="test-ratio" min="0" max="0.3" step="0.05" value="0.1">
                    <span id="test-ratio-value">10%</span>
                </div>
                
                <div id="split-preview" class="split-preview">
                    <p>点击"预览"查看分割结果</p>
                </div>
                
                <div class="modal-actions">
                    <button class="btn btn-ghost" id="preview-split">预览</button>
                    <button class="btn btn-primary" id="apply-split" disabled>应用分割</button>
                    <button class="btn btn-ghost" id="cancel-split">取消</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Range slider updates
        const updateRatios = () => {
            const train = parseFloat(modal.querySelector('#train-ratio').value);
            const val = parseFloat(modal.querySelector('#val-ratio').value);
            const test = parseFloat(modal.querySelector('#test-ratio').value);
            
            modal.querySelector('#train-ratio-value').textContent = Math.round(train * 100) + '%';
            modal.querySelector('#val-ratio-value').textContent = Math.round(val * 100) + '%';
            modal.querySelector('#test-ratio-value').textContent = Math.round(test * 100) + '%';
            
            const total = train + val + test;
            const applyBtn = modal.querySelector('#apply-split');
            if (Math.abs(total - 1.0) > 0.001) {
                modal.querySelector('#split-preview').innerHTML = `<p class="error">比例总和必须等于 100% (当前: ${Math.round(total * 100)}%)</p>`;
                applyBtn.disabled = true;
            } else {
                modal.querySelector('#split-preview').innerHTML = '<p>点击"预览"查看分割结果</p>';
            }
        };
        
        modal.querySelectorAll('input[type="range"]').forEach(input => {
            input.addEventListener('input', updateRatios);
        });
        
        modal.querySelector('#cancel-split').addEventListener('click', () => modal.remove());
        
        modal.querySelector('#preview-split').addEventListener('click', async () => {
            const strategy = modal.querySelector('#split-strategy').value;
            const train = parseFloat(modal.querySelector('#train-ratio').value);
            const val = parseFloat(modal.querySelector('#val-ratio').value);
            const test = parseFloat(modal.querySelector('#test-ratio').value);
            
            showLoading(true, '计算分割方案...');
            
            try {
                const res = await fetch(`${API}/datasets/split_preview`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        group_id: groupId,
                        label_set_id: ds.label_sets[0]?.set_id || 'default',
                        strategy,
                        train_ratio: train,
                        val_ratio: val,
                        test_ratio: test,
                        dry_run: true
                    })
                });
                
                const result = await res.json();
                
                modal.querySelector('#split-preview').innerHTML = `
                    <div class="split-stats">
                        <div class="split-item">
                            <span class="split-label">训练集</span>
                            <span class="split-count">${result.splits.train.count} 张</span>
                            <span class="split-percent">${result.splits.train.percentage}%</span>
                        </div>
                        <div class="split-item">
                            <span class="split-label">验证集</span>
                            <span class="split-count">${result.splits.val.count} 张</span>
                            <span class="split-percent">${result.splits.val.percentage}%</span>
                        </div>
                        <div class="split-item">
                            <span class="split-label">测试集</span>
                            <span class="split-count">${result.splits.test.count} 张</span>
                            <span class="split-percent">${result.splits.test.percentage}%</span>
                        </div>
                    </div>
                    <p class="split-total">总计: ${result.total_images} 张图片</p>
                `;
                
                modal.querySelector('#apply-split').disabled = false;
            } catch (e) {
                console.error('Split preview failed:', e);
                showStatus('预览失败', true);
            } finally {
                showLoading(false);
            }
        });
        
        modal.querySelector('#apply-split').addEventListener('click', () => {
            showStatus('分割配置已生成，请在YOLO配置文件中应用');
            modal.remove();
        });
    },
    
    // Visualization
    async showVisualization(groupId) {
        const ds = this.datasets.find(d => d.group_id === groupId);
        if (!ds) return;
        
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 800px; max-height: 90vh; overflow-y: auto;">
                <h3>📈 数据可视化 - ${ds.group_name}</h3>
                
                <div class="viz-tabs">
                    <button class="viz-tab active" data-viz="class_distribution">类别分布</button>
                    <button class="viz-tab" data-viz="bbox_size_distribution">尺寸分布</button>
                    <button class="viz-tab" data-viz="annotation_count_per_image">标注数量</button>
                </div>
                
                <div id="viz-container" class="viz-container">
                    <p>加载中...</p>
                </div>
                
                <div class="modal-actions">
                    <button class="btn btn-ghost" id="close-viz">关闭</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        const loadViz = async (vizType) => {
            const container = modal.querySelector('#viz-container');
            container.innerHTML = '<p>加载中...</p>';
            
            try {
                const res = await fetch(`${API}/datasets/visualization?group_id=${groupId}&label_set_id=${ds.label_sets[0]?.set_id || 'default'}&viz_type=${vizType}`);
                const result = await res.json();
                
                if (result.chart_type === 'bar') {
                    this.renderBarChart(container, result);
                } else if (result.chart_type === 'pie') {
                    this.renderPieChart(container, result);
                } else {
                    container.innerHTML = `<p>图表类型: ${result.chart_type}</p><pre>${JSON.stringify(result.data, null, 2)}</pre>`;
                }
            } catch (e) {
                container.innerHTML = '<p class="error">加载失败</p>';
            }
        };
        
        // Tab switching
        modal.querySelectorAll('.viz-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                modal.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                loadViz(tab.dataset.viz);
            });
        });
        
        // Initial load
        loadViz('class_distribution');
        
        modal.querySelector('#close-viz').addEventListener('click', () => modal.remove());
    },
    
    renderBarChart(container, data) {
        const maxValue = Math.max(...data.data.values);
        const html = `
            <div class="chart-title">${data.title}</div>
            <div class="bar-chart">
                ${data.data.labels.map((label, i) => {
                    const value = data.data.values[i];
                    const percent = (value / maxValue * 100).toFixed(1);
                    const color = Array.isArray(data.data.colors) ? data.data.colors[i] : data.data.colors;
                    return `
                        <div class="bar-item">
                            <div class="bar-label" title="${label}">${label}</div>
                            <div class="bar-wrapper">
                                <div class="bar" style="width: ${percent}%; background: ${color};"></div>
                            </div>
                            <div class="bar-value">${value.toLocaleString()}</div>
                        </div>
                    `;
                }).join('')}
            </div>
            ${data.total_annotations ? `<p class="chart-total">总计: ${data.total_annotations.toLocaleString()} 个标注</p>` : ''}
        `;
        container.innerHTML = html;
    },
    
    renderPieChart(container, data) {
        const total = data.data.values.reduce((a, b) => a + b, 0);
        let currentAngle = 0;
        
        const segments = data.data.labels.map((label, i) => {
            const value = data.data.values[i];
            const angle = (value / total) * 360;
            const startAngle = currentAngle;
            currentAngle += angle;
            return { label, value, angle, startAngle, color: data.data.colors[i] };
        });
        
        // Create conic gradient
        let gradientStr = segments.map((s, i) => 
            `${s.color} ${s.startAngle}deg ${s.startAngle + s.angle}deg`
        ).join(', ');
        
        const html = `
            <div class="chart-title">${data.title}</div>
            <div class="pie-chart-container">
                <div class="pie-chart" style="background: conic-gradient(${gradientStr});"></div>
                <div class="pie-legend">
                    ${segments.map(s => `
                        <div class="legend-item">
                            <span class="legend-color" style="background: ${s.color};"></span>
                            <span class="legend-label">${s.label}</span>
                            <span class="legend-value">${s.value} (${(s.value/total*100).toFixed(1)}%)</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            <p class="chart-total">总计: ${total.toLocaleString()} 个标注</p>
        `;
        container.innerHTML = html;
    },
    
    // Modal control
    open() {
        if (this.els.modal) {
            this.els.modal.style.display = 'flex';
            this.loadDatasets();
            this.loadSummary();
        }
    },
    
    close() {
        if (this.els.modal) {
            this.els.modal.style.display = 'none';
        }
    },
    
    // Utility
    debounce(fn, ms) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => fn.apply(this, args), ms);
        };
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if the dataset manager modal exists
    if (document.getElementById('dataset-manager-modal')) {
        DatasetManager.init();
    }
});
