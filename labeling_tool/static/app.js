// ══════════════════════════════════════════════════════════════
// 标注编辑系统 - app.js (Optimized)
// ══════════════════════════════════════════════════════════════

const API = '/api';

// ── State ─────────────────────────────────────────────────────
const state = {
    // Image groups & label sets
    groups: [],
    currentGroup: null,
    currentLabelSet: null,
    subset: 'train',

    // Images
    images: [],          // [{filename, has_label}]
    currentImageIndex: -1,
    imageWidth: 0,
    imageHeight: 0,

    // Classes
    classes: [],
    colors: {},

    // Mode
    mode: 'select',      // select | adjust | box | sam | polygon
    isDrawing: false,
    startPos: null,

    // Pan mode state
    _spaceHeld: false,
    _prePanMode: null,

    // Clipboard for copy/paste
    _clipboard: [],

    // Annotation list filter state
    _annFilterText: '',
    _annFilterClass: null,

    // Flagged images cache
    _flaggedImages: new Set(),

    // Polygon drawing state
    polygonPoints: [],       // [{x, y}] in image coords during polygon drawing
    polygonKonvaLine: null,  // temp Konva.Line while drawing
    polygonKonvaDots: [],    // temp Konva.Circles for vertices
    polygonPreviewLine: null, // cursor-following preview edge

    // Visualization
    showBoxes: true,
    showBoundary: true,
    showFill: false,
    showStroke: true,
    vizModeValue: 'both',
    showSelectedOnly: false,
    prevViz: null,

    // Annotations
    annotationCounter: 0,
    dirty: false,         // Unsaved changes?

    // Undo/Redo stacks
    undoStack: [],
    redoStack: [],
    maxUndo: 100,

    // Image preload cache
    imageCache: new Map(),         // cacheKey -> Image object
    imagePendingCache: new Map(),  // cacheKey -> Promise<Image>
    annotationCache: new Map(),    // cacheKey -> annotations[]
    annotationPendingCache: new Map(), // cacheKey -> Promise<annotations[]>
    prefetchRange: 3,              // preload ±3 images
    annotationPrefetchRange: 6,    // preload ±6 annotation files
    fewshot: {
        classifyMethod: 'hybrid',
        supports: [],
        status: '',
        statusIsError: false,
        busy: false,
        lastMode: '',
        currentResult: null,
        evalResult: null,
        supportFilterClass: 'all',
        evalScope: 'all',
        rangeStart: '0',
        rangeEnd: '0',
        expandedSupportKey: null,
        usePrompts: false,
        promptMode: 'auto',
        promptEnsembles: {},
        imageProtoWeight: 0.5,
        textProtoWeight: 0.5,
        primaryPromptWeight: 0.75,
        // Hybrid text prompt settings (任意类数)
        hybridPromptTemplate: '',           // 模板, 留空用默认
        hybridTextPromptNames: {},          // { classId: descPhrase }
    },
};

// Expose to window for cross-script access (data-manager.js)
window.state = state;

// ── Performance utilities ────────────────────────────────────
function throttle(fn, delay) {
    let last = 0, timer = null;
    return function (...args) {
        const now = Date.now();
        if (now - last >= delay) {
            last = now;
            fn.apply(this, args);
        } else {
            clearTimeout(timer);
            timer = setTimeout(() => { last = Date.now(); fn.apply(this, args); }, delay - (now - last));
        }
    };
}

function debounce(fn, delay) {
    let timer = null;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

function cloneAnnotations(anns) {
    return (anns || []).map(ann => ({
        ...ann,
        points: Array.isArray(ann.points) ? [...ann.points] : ann.points,
    }));
}

// ── Konva objects ─────────────────────────────────────────────
let stage, imageLayer, boxLayer, annotLayer, labelLayer, tempLayer, cellposeOverlayLayer, samPreviewLayer;
let currentImageNode;
let tempRect, tempLabel;
let selectedGroup = null;
let selectedGroups = new Set();

// SAM3 batch preview data
let samPreviewData = null;  // [{class_id, points}, ...]
let samPreviewSourceBboxGroups = [];  // Konva groups of bbox annotations used as SAM3 input

// CellposeSAM preview data
let cellposePreviewData = null;  // [{class_id, points}, ...]
let cellposeBatchPollTimer = null;
let activeCellposeBatchJobId = null;
let cellposeBatchPhase = 'idle';
let imageSelectToken = 0;
let loadingOverlayTimer = null;
let saveRequestCounter = 0;

// Global abort controller for long-running operations
let _longRunAbort = null;
function requestStopLongRun() { if (_longRunAbort) _longRunAbort.abort(); }
function startLongRun() { _longRunAbort = new AbortController(); return _longRunAbort.signal; }

// Adjust mode handles
let adjustRect = null;       // The editable rect
let adjustHandles = [];      // 4 corner circles [TL, TR, BR, BL]
let adjustOrigPts = null;    // Backup of original polygon points before adjust
let adjustOrigBBox = null;   // Original bbox {x1,y1,x2,y2} for reference

// ── DOM refs ──────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const els = {
    groupSelect:     $('group-select'),
    groupInfo:       $('group-info'),
    labelsetSelect:  $('labelset-select'),
    subsetSelect:    $('subset-select'),
    imageList:       $('image-list'),
    imageCounter:    $('image-counter'),
    searchInput:     $('search-input'),
    imageStatusFilter: $('image-status-filter'),
    canvasContainer: $('canvas-container'),
    classSelect:     $('class-select'),
    classList:       $('class-list'),
    modeSelect:      $('mode-select'),
    modeAdjust:      $('mode-adjust'),
    modeBox:         $('mode-box'),
    modePolygon:     $('mode-polygon'),
    modeSam:         $('mode-sam'),
    saveBtn:         $('save-btn'),
    prevBtn:         $('prev-btn'),
    nextBtn:         $('next-btn'),
    deleteBtn:       $('delete-btn'),
    batchAddSupportBtn: $('batch-add-support-btn'),
    loading:         $('loading-overlay'),
    status:          $('status-msg'),
    imageSize:       $('image-size'),
    zoomLevel:       $('zoom-level'),
    modeHint:        $('mode-hint'),
    quickGroupName:  $('quick-group-name'),
    quickLabelsetName:$('quick-labelset-name'),
    quickImageName:  $('quick-image-name'),
    quickSupportCount:$('quick-support-count'),
    annotationList:  $('annotation-list'),
    annCount:        $('ann-count'),
    vizMode:         $('viz-mode'),
    vizSelectedOnly: $('viz-selected-only'),
    unsavedDot:      $('unsaved-dot'),
    saveNextBtn:     $('save-next-btn'),
    adjustActions:   $('adjust-actions'),
    adjustConfirm:   $('adjust-confirm'),
    adjustCancel:    $('adjust-cancel'),
    scanGroupsBtn:      $('scan-groups-btn'),
    groupSearchPanel:   $('group-search-panel'),
    groupSearchInput:   $('group-search-input'),
    groupSearchResults: $('group-search-results'),
    genBboxBtn:            $('gen-bbox-btn'),
    scanLabelsetsBtn:      $('scan-labelsets-btn'),
    labelsetSearchPanel:   $('labelset-search-panel'),
    labelsetSearchInput:   $('labelset-search-input'),
    labelsetSearchResults: $('labelset-search-results'),
    addClassBtn:     $('add-class-btn'),
    addClassForm:    $('add-class-form'),
    newClassName:    $('new-class-name'),
    addClassOk:      $('add-class-ok'),
    addClassNo:      $('add-class-no'),
    // SAM3 prompt controls
    samTextRow:        $('sam-text-row'),
    samTextPrompt:     $('sam-text-prompt'),
    samTextRun:        $('sam-text-run'),
    samBatchBtn:       $('sam-batch-btn'),
    samPreviewActions: $('sam-preview-actions'),
    samAcceptBtn:      $('sam-accept-btn'),
    samRejectBtn:      $('sam-reject-btn'),
    samStatus:         $('sam-status'),
    // CellposeSAM visualization
    cellposeRunBtn:      $('cellpose-run-btn'),
    cellposeClearBtn:    $('cellpose-clear-btn'),
    cellposeDiameters:   $('cellpose-diameters'),
    cellposeGpu:         $('cellpose-gpu'),
    cellposeStatus:      $('cellpose-status'),
    cellposeColor:       $('cellpose-color'),
    cellposeVisible:     $('cellpose-visible'),
    cellposeAcceptActions: $('cellpose-accept-actions'),
    cellposeAcceptBtn:   $('cellpose-accept-btn'),
    cellposeRejectBtn:   $('cellpose-reject-btn'),
    cellposeBatchStart:  $('cellpose-batch-start'),
    cellposeBatchEnd:    $('cellpose-batch-end'),
    cellposeBatchAllBtn: $('cellpose-batch-all-btn'),
    cellposeBatchTailBtn: $('cellpose-batch-tail-btn'),
    cellposeBatchRunBtn: $('cellpose-batch-run-btn'),
    cellposeBatchCancelBtn: $('cellpose-batch-cancel-btn'),
    cellposeBatchSaveBtn: $('cellpose-batch-save-btn'),
    cellposeBatchDiscardBtn: $('cellpose-batch-discard-btn'),
    cellposeBatchSkip:   $('cellpose-batch-skip-existing'),
    cellposeBatchOverwrite: $('cellpose-batch-overwrite-existing'),
    cellposeBatchStatus: $('cellpose-batch-status'),
    cellposeBatchProgress: $('cellpose-batch-progress'),
    cellposeBatchCurrent: $('cellpose-batch-current'),
    cellposeBatchPercent: $('cellpose-batch-percent'),
    cellposeBatchBar: $('cellpose-batch-bar'),
    cellposeBatchSavedBar: $('cellpose-batch-saved-bar'),
    cellposeBatchSkippedBar: $('cellpose-batch-skipped-bar'),
    cellposeBatchFailedBar: $('cellpose-batch-failed-bar'),
    cellposeBatchSavedText: $('cellpose-batch-saved-text'),
    cellposeBatchSkippedText: $('cellpose-batch-skipped-text'),
    cellposeBatchFailedText: $('cellpose-batch-failed-text'),
    // Smart classification panel
    classifyMethod: $('classify-method'),
    fewshotSupportCounts: $('fewshot-support-counts'),
    fewshotSupportList: $('fewshot-support-list'),
    fewshotSupportFilter: $('fewshot-support-filter'),
    fewshotSupportHeaderCount: $('fewshot-support-header-count'),
    fewshotManageBtn: $('fewshot-manage-btn'),
    fewshotSupportPopover: $('fewshot-support-popover'),
    fewshotPopoverClose: $('fewshot-popover-close'),
    fewshotStatus: $('fewshot-status'),
    fewshotResultPanel: $('fewshot-result-panel'),
    fewshotAddSelectedBtn: $('fewshot-add-selected-btn'),
    fewshotRemoveSelectedBtn: $('fewshot-remove-selected-btn'),
    fewshotClearBtn: $('fewshot-clear-btn'),
    fewshotTemperature: $('fewshot-temperature'),
    fewshotMaxImages: $('fewshot-max-images'),
    fewshotEvalScope: $('fewshot-eval-scope'),
    fewshotRangeStart: $('fewshot-range-start'),
    fewshotRangeEnd: $('fewshot-range-end'),
    fewshotUsePrompts: $('fewshot-use-prompts'),
    fewshotPromptOptions: $('fewshot-prompt-options'),
    fewshotPromptMode: $('fewshot-prompt-mode'),
    fewshotImageProtoWeight: $('fewshot-image-proto-weight'),
    fewshotTextProtoWeight: $('fewshot-text-proto-weight'),
    fewshotCustomPrompts: $('fewshot-custom-prompts'),
    fewshotRangeHint: $('fewshot-range-hint'),
    fewshotRunCurrentBtn: $('fewshot-run-current-btn'),
    fewshotEvalSubsetBtn: $('fewshot-eval-subset-btn'),
    fewshotBatchClassifyBtn: $('fewshot-batch-classify-btn'),
    fewshotMetrics: $('fewshot-metrics'),
    fewshotApplyBtn: $('fewshot-apply-btn'),
    fewshotRevertBtn: $('fewshot-revert-btn'),
    fewshotPreviewActions: $('fewshot-preview-actions'),
    fewshotResultList: $('fewshot-result-list'),
    fewshotSaveSupportsBtn: $('fewshot-save-supports-btn'),
    fewshotPromptResetBtn: $('fewshot-prompt-reset-btn'),
    fewshotPromptEditor: $('fewshot-prompt-editor'),
    // Hybrid-specific elements
    hybridSettingsCollapsible: $('hybrid-settings-collapsible'),
    hybridPromptCollapsible: $('hybrid-prompt-collapsible'),
    hybridEnableSizeRefiner: $('hybrid-enable-size-refiner'),
    hybridSrMargin: $('hybrid-sr-margin'),
    hybridSrSeparation: $('hybrid-sr-separation'),
    hybridSrScale: $('hybrid-sr-scale'),
    hybridSrMaxAdjust: $('hybrid-sr-max-adjust'),
    hybridPromptTemplate: $('hybrid-prompt-template'),
    hybridTextPrompts: $('hybrid-text-prompts'),
    hybridPromptClearBtn: $('hybrid-prompt-clear-btn'),
    hybridMaxImages: $('hybrid-max-images'),
    hybridEvalRow: $('hybrid-eval-row'),
    basicTempRow: $('basic-temp-row'),
    fewshotPromptCollapsible: $('fewshot-prompt-collapsible'),
    evalGoldLabelset: $('eval-gold-labelset'),
    evalSegmentMethod: $('eval-segment-method'),
    evalPromptLabelset: $('eval-prompt-labelset'),
    evalMaxImages: $('eval-max-images'),
    evalClassifyGoldBtn: $('eval-classify-gold-btn'),
    evalSegmentGoldBtn: $('eval-segment-gold-btn'),
    evalStatus: $('eval-status'),
    // Footer status bar
    footerMode:      $('footer-mode'),
    footerImage:     $('footer-image'),
    footerAnns:      $('footer-anns'),
    footerDirty:     $('footer-dirty'),
    footerSaved:     $('footer-saved'),
    footerAutosave:  $('footer-autosave'),
};

// ══════════════════════════════════════════════════════════════
// Init
// ══════════════════════════════════════════════════════════════

async function init() {
    initKonva();
    setupEventListeners();
    updateClassifyMethodUI();
    await loadGroups();
    updateModeHint();
    updateStatusBar();
    renderFewshotPanel();
    startAutoSaveTimer();
    _updateEmptyState();
}

function initKonva() {
    const w = els.canvasContainer.clientWidth;
    const h = els.canvasContainer.clientHeight;

    stage = new Konva.Stage({
        container: 'konva-holder',
        width: w,
        height: h,
    });
    stage.draggable(false);

    imageLayer = new Konva.Layer();
    boxLayer   = new Konva.Layer();
    annotLayer = new Konva.Layer();
    labelLayer = new Konva.Layer({ listening: false });
    cellposeOverlayLayer = new Konva.Layer({ listening: false });
    samPreviewLayer = new Konva.Layer({ listening: false });
    tempLayer  = new Konva.Layer();

    stage.add(imageLayer);
    stage.add(boxLayer);
    stage.add(annotLayer);
    stage.add(labelLayer);
    stage.add(cellposeOverlayLayer);
    stage.add(samPreviewLayer);
    stage.add(tempLayer);

    // ── Zoom with scroll wheel (throttled for smoothness) ──
    const handleZoom = throttle((e) => {
        const oldScale = stage.scaleX();
        const pointer = stage.getPointerPosition();
        if (!pointer) return;
        const mousePointTo = {
            x: (pointer.x - stage.x()) / oldScale,
            y: (pointer.y - stage.y()) / oldScale,
        };
        const direction = e.evt.deltaY < 0 ? 1 : -1;
        const factor = 1.12;
        let newScale = direction > 0 ? oldScale * factor : oldScale / factor;
        newScale = Math.max(0.05, Math.min(20, newScale));

        stage.scale({ x: newScale, y: newScale });
        stage.position({
            x: pointer.x - mousePointTo.x * newScale,
            y: pointer.y - mousePointTo.y * newScale,
        });
        _clampStagePosition();
        updateZoomUI(newScale);
        updateScaledSizes();
    }, 16); // ~60fps

    stage.on('wheel', (e) => {
        e.evt.preventDefault();
        handleZoom(e);
    });

    stage.on('mousedown', onMouseDown);
    stage.on('mousemove', onMouseMove);
    stage.on('mouseup', onMouseUp);
    stage.on('dblclick', () => {
        if (state.mode === 'polygon' && state.polygonPoints.length >= 3) {
            _finishPolygonDrawing();
        }
    });

    // Resize observer — keep stage size synced and image visible
    let _resizeTimer = null;
    const ro = new ResizeObserver(() => {
        stage.width(els.canvasContainer.clientWidth);
        stage.height(els.canvasContainer.clientHeight);
        if (state.imageWidth && state.imageHeight) {
            _clampStagePosition();
            clearTimeout(_resizeTimer);
            _resizeTimer = setTimeout(() => fitImageToView(), 200);
        }
    });
    ro.observe(els.canvasContainer);
}


// ══════════════════════════════════════════════════════════════
// Data Loading
// ══════════════════════════════════════════════════════════════

async function loadGroups() {
    const res = await fetch(`${API}/groups`);
    state.groups = await res.json();

    if (state.groups.length > 0) {
        els.groupSelect.innerHTML = state.groups.map(g =>
            `<option value="${g.group_id}">${g.group_name}</option>`
        ).join('');
        await selectGroup(state.groups[0].group_id);
    } else {
        els.groupSelect.innerHTML = '<option value="">暂无数据集</option>';
        els.groupInfo.textContent = '请通过数据管理中心添加数据集';
        els.imageList.innerHTML = `<div class="empty-state" style="padding:24px 16px;">
            <strong>暂无数据集</strong>
            <span>按 <kbd style="background:var(--bg-3);padding:1px 5px;border-radius:3px;font-size:10px;font-family:monospace;border:1px solid var(--border);">M</kbd> 打开数据管理中心</span>
        </div>`;
    }
    _updateEmptyState();
}

function warmupPreviewCache() {
    if (!state.currentGroup) return;
    fetch(`${API}/images/warmup_jpeg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            group_id: state.currentGroup.group_id,
            subset: state.subset,
        }),
    }).catch(() => {});
}

async function _ensureContextSaved(reasonLabel = '切换上下文') {
    if (!state.dirty) return true;
    if (!state.currentGroup || !state.currentLabelSet || state.currentImageIndex < 0) {
        console.warn('_ensureContextSaved: clearing dirty (no valid context)');
        state.dirty = false;
        return true;
    }
    const ok = await saveAnnotations();
    if (!ok) {
        showStatus(`${reasonLabel}前保存失败，请先处理后再继续`, true);
        return false;
    }
    return true;
}

async function selectGroup(groupId) {
    if (!(await _ensureContextSaved('切换图片集'))) return false;
    state.currentGroup = state.groups.find(g => g.group_id === groupId);
    if (!state.currentGroup) return false;
    els.groupSelect.value = groupId;

    // Tear down any active batch job from previous group
    _stopCellposeBatchPolling();
    if (activeCellposeBatchJobId) {
        activeCellposeBatchJobId = null;
        cellposeBatchPhase = 'idle';
    }
    _clearCellposePreview();
    _clearSamPreview();

    resetFewshotState();

    const g = state.currentGroup;
    els.groupInfo.textContent = `${g.nc}类 · Train:${g.train_count} Val:${g.val_count}`;

    // Populate label set dropdown
    els.labelsetSelect.innerHTML = g.label_sets.map(ls =>
        `<option value="${ls.set_id}">${ls.set_name} (${ls.label_format})</option>`
    ).join('');

    // Select first label set
    if (g.label_sets.length > 0) {
        state.currentLabelSet = g.label_sets[0];
    }

    await loadClasses();
    await loadImages();
    await loadSavedSupports();
    loadHybridPromptNames();
    warmupPreviewCache();
    _updateEmptyState();
    updateStatusBar();
    return true;
}

async function selectLabelSet(setId) {
    if (!state.currentGroup) return false;
    if (!(await _ensureContextSaved('切换标注集'))) return false;
    state.currentLabelSet = state.currentGroup.label_sets.find(ls => ls.set_id === setId);
    if (!state.currentLabelSet) return false;
    els.labelsetSelect.value = setId;
    state.fewshot.currentResult = null;
    state.fewshot.evalResult = null;
    state.fewshot.lastMode = '';
    state.fewshot.busy = false;
    await loadSavedSupports();
    loadHybridPromptNames();
    return true;
}

async function loadClasses() {
    if (!state.currentGroup) return;
    const res = await fetch(`${API}/classes?group_id=${encodeURIComponent(state.currentGroup.group_id)}`);
    state.classes = await res.json();

    els.classSelect.innerHTML = state.classes.map(c =>
        `<option value="${c.ID}">${c.Name} (${c.ID})</option>`
    ).join('');

    state.colors = {};
    state.classes.forEach(c => { state.colors[c.ID] = c.Color; });
    renderClassList();
    renderFewshotPanel();
}

async function loadImages() {
    if (!state.currentGroup || !state.currentLabelSet) return;
    const gid = encodeURIComponent(state.currentGroup.group_id);
    const sid = encodeURIComponent(state.currentLabelSet.set_id);
    const res = await fetch(`${API}/images?group_id=${gid}&label_set_id=${sid}&subset=${state.subset}`);
    state.images = await res.json();
    state.currentImageIndex = -1;
    state.annotationCounter = 0;
    state.fewshot.currentResult = null;
    state.fewshot.evalResult = null;
    state.fewshot.lastMode = '';
    state.fewshot.rangeStart = '0';
    state.fewshot.rangeEnd = String(Math.max(0, state.images.length - 1));
    // Load flagged images for this dataset
    _loadFlaggedImages().catch(() => {});
    renderImageList();
    _thumbStripState.rendered = false;
    _thumbStripState.currentCenter = -1;
    refreshCellposeBatchRange();
    refreshFewshotRangeOptions();
    warmupPreviewCache();

    if (state.images.length > 0) {
        await selectImage(0);
    } else {
        imageLayer.destroyChildren();
        annotLayer.destroyChildren();
        boxLayer.destroyChildren();
        if (labelLayer) labelLayer.destroyChildren();
        stage.batchDraw();
        els.imageCounter.textContent = '0 / 0';
    }
    _updateEmptyState();
    updateCellposeBatchRunState();
    renderFewshotPanel();
}

async function reloadImagesKeepSelection() {
    const currentFilename = state.images[state.currentImageIndex]?.filename || null;
    await loadImages();
    if (!currentFilename || state.images.length === 0) return;
    const idx = state.images.findIndex(img => img.filename === currentFilename);
    if (idx >= 0) {
        await selectImage(idx);
    }
}

function refreshCellposeBatchRange() {
    if (!els.cellposeBatchStart || !els.cellposeBatchEnd) return;
    if (!state.images.length) {
        els.cellposeBatchStart.innerHTML = '<option value="">无图片</option>';
        els.cellposeBatchEnd.innerHTML = '<option value="">无图片</option>';
        if (els.cellposeBatchProgress) els.cellposeBatchProgress.textContent = '';
        if (els.cellposeBatchStatus) els.cellposeBatchStatus.textContent = '当前子集没有可批量处理的图片';
        return;
    }

    const prevStart = els.cellposeBatchStart.value;
    const prevEnd = els.cellposeBatchEnd.value;
    const options = state.images.map((img, idx) =>
        `<option value="${idx}">${String(idx + 1).padStart(3, '0')} · ${img.filename}</option>`
    ).join('');
    els.cellposeBatchStart.innerHTML = options;
    els.cellposeBatchEnd.innerHTML = options;

    const validStart = prevStart !== '' && Number(prevStart) >= 0 && Number(prevStart) < state.images.length;
    const validEnd = prevEnd !== '' && Number(prevEnd) >= 0 && Number(prevEnd) < state.images.length;
    els.cellposeBatchStart.value = validStart ? prevStart : '0';
    els.cellposeBatchEnd.value = validEnd ? prevEnd : String(state.images.length - 1);
}

function _getCellposeDiameters() {
    const diamText = els.cellposeDiameters?.value || '30';
    return diamText.split(/[,，\s]+/).map(Number).filter(v => v > 0 && !isNaN(v));
}

function _formatBatchRangeLabel(startIndex, endIndex) {
    if (!state.images.length) return '未选择图片';
    const start = state.images[startIndex];
    const end = state.images[endIndex];
    if (!start || !end) return '范围无效';
    return `${startIndex + 1}-${endIndex + 1} · ${start.filename}${startIndex === endIndex ? '' : ` → ${end.filename}`}`;
}

function _stopCellposeBatchPolling() {
    if (cellposeBatchPollTimer) {
        clearTimeout(cellposeBatchPollTimer);
        cellposeBatchPollTimer = null;
    }
}

function _updateCellposeBatchActionButtons() {
    const phase = cellposeBatchPhase;
    const canResolveFailedSave = phase === 'failed' && !!activeCellposeBatchJobId;
    if (els.cellposeBatchCancelBtn) {
        const showCancel = ['queued', 'running', 'cancel_requested'].includes(phase) && !!activeCellposeBatchJobId;
        els.cellposeBatchCancelBtn.style.display = showCancel ? '' : 'none';
        els.cellposeBatchCancelBtn.disabled = phase === 'cancel_requested';
    }
    if (els.cellposeBatchSaveBtn) {
        const showSave = (phase === 'awaiting_save' || canResolveFailedSave) && !!activeCellposeBatchJobId;
        els.cellposeBatchSaveBtn.style.display = showSave ? '' : 'none';
        els.cellposeBatchSaveBtn.disabled = !showSave;
    }
    if (els.cellposeBatchDiscardBtn) {
        const showDiscard = (phase === 'awaiting_save' || canResolveFailedSave) && !!activeCellposeBatchJobId;
        els.cellposeBatchDiscardBtn.style.display = showDiscard ? '' : 'none';
        els.cellposeBatchDiscardBtn.disabled = !showDiscard;
    }
}

function _setCellposeBatchProgress({
    total = 0,
    processed = 0,
    current = '',
    saved = 0,
    skipped = 0,
    failed = 0,
    status = 'idle',
} = {}) {
    const safeTotal = Math.max(0, total);
    const safeProcessed = Math.max(0, Math.min(processed, safeTotal || processed));
    const percent = safeTotal > 0 ? Math.round((safeProcessed / safeTotal) * 100) : 0;

    if (els.cellposeBatchPercent) {
        els.cellposeBatchPercent.textContent = `${percent}%`;
    }
    if (els.cellposeBatchProgress) {
        els.cellposeBatchProgress.textContent = `${safeProcessed} / ${safeTotal}`;
    }
    if (els.cellposeBatchCurrent) {
        els.cellposeBatchCurrent.textContent = current || (status === 'running' ? '处理中...' : '等待启动');
    }

    if (els.cellposeBatchBar) {
        els.cellposeBatchBar.style.width = `${percent}%`;
        els.cellposeBatchBar.classList.remove('running', 'done', 'failed');
        if (status === 'running' || status === 'queued') {
            els.cellposeBatchBar.classList.add('running');
        } else if (status === 'completed') {
            els.cellposeBatchBar.classList.add('done');
        } else if (status === 'failed') {
            els.cellposeBatchBar.classList.add('failed');
        }
    }

    const denom = safeTotal || 1;
    if (els.cellposeBatchSavedBar) els.cellposeBatchSavedBar.style.width = `${(saved / denom) * 100}%`;
    if (els.cellposeBatchSkippedBar) els.cellposeBatchSkippedBar.style.width = `${(skipped / denom) * 100}%`;
    if (els.cellposeBatchFailedBar) els.cellposeBatchFailedBar.style.width = `${(failed / denom) * 100}%`;

    const primaryLabel = status === 'saved' ? '保存' : (status === 'awaiting_save' ? '待保存' : '分割');
    if (els.cellposeBatchSavedText) els.cellposeBatchSavedText.textContent = `${primaryLabel} ${saved}`;
    if (els.cellposeBatchSkippedText) els.cellposeBatchSkippedText.textContent = `跳过 ${skipped}`;
    if (els.cellposeBatchFailedText) els.cellposeBatchFailedText.textContent = `失败 ${failed}`;
}

function updateCellposeBatchRunState() {
    if (!els.cellposeBatchRunBtn) return;

    const hasImages = state.images.length > 0;
    const isPolygonSet = !state.currentLabelSet || state.currentLabelSet.label_format !== 'bbox';
    const startValue = Number(els.cellposeBatchStart?.value ?? -1);
    const endValue = Number(els.cellposeBatchEnd?.value ?? -1);
    const validRange = hasImages && Number.isInteger(startValue) && Number.isInteger(endValue)
        && startValue >= 0 && endValue >= startValue && endValue < state.images.length;
    const busy = !!activeCellposeBatchJobId;

    els.cellposeBatchRunBtn.disabled = !hasImages || !isPolygonSet || !validRange || busy;
    _updateCellposeBatchActionButtons();

    if (!els.cellposeBatchStatus) return;
    if (!hasImages) {
        _setCellposeBatchProgress({ total: 0, processed: 0, current: '无图片', status: 'idle' });
        els.cellposeBatchStatus.textContent = '当前子集没有图片，无法批量分割';
        return;
    }
    if (!isPolygonSet) {
        _setCellposeBatchProgress({ total: state.images.length, processed: 0, current: '仅支持 polygon', status: 'failed' });
        els.cellposeBatchStatus.textContent = '当前标注集是 bbox，批量 CellposeSAM 仅支持 polygon 标注集';
        return;
    }
    if (!validRange) {
        _setCellposeBatchProgress({ total: state.images.length, processed: 0, current: '等待范围选择', status: 'idle' });
        els.cellposeBatchStatus.textContent = '请选择有效的起止范围';
        return;
    }
    if (busy) return;

    const total = endValue - startValue + 1;
    _setCellposeBatchProgress({
        total,
        processed: 0,
        current: '准备开始',
        saved: 0,
        skipped: 0,
        failed: 0,
        status: 'idle',
    });
    els.cellposeBatchStatus.textContent = '';
}

function getCurrentAnnotationsPayload() {
    const anns = [];
    annotLayer.getChildren().forEach(group => {
        if (group.name() !== 'annotation') return;
        anns.push({
            class_id: group.getAttr('classId'),
            ann_type: group.getAttr('annType'),
            points: group.getAttr('normPoints').slice(),
            annotation_uid: group.getAttr('uid'),
        });
    });
    return anns;
}

function buildFewshotSupportKey(item) {
    const uid = item.annotation_uid || '';
    if (uid) {
        return [
            item.subset || '',
            item.filename || '',
            item.class_id,
            item.ann_type || 'polygon',
            `uid:${uid}`,
        ].join('::');
    }
    const pts = (item.points || []).map(v => Number(v).toFixed(6)).join(',');
    return [
        item.subset || '',
        item.filename || '',
        item.class_id,
        item.ann_type || 'polygon',
        pts,
    ].join('::');
}

function getSelectedAnnotationPayload() {
    if (!selectedGroup || state.currentImageIndex < 0) return null;
    const currentImage = state.images[state.currentImageIndex];
    if (!currentImage) return null;
    return {
        filename: currentImage.filename,
        subset: state.subset,
        class_id: selectedGroup.getAttr('classId'),
        ann_type: selectedGroup.getAttr('annType'),
        points: selectedGroup.getAttr('normPoints').slice(),
        annotation_uid: selectedGroup.getAttr('uid'),
    };
}

function annotationMatchesSupport(item, support) {
    if (!item || !support) return false;
    const itemUid = item.annotation_uid || '';
    const supportUid = support.annotation_uid || '';
    if (itemUid && supportUid) return itemUid === supportUid;
    return buildFewshotSupportKey(item) === (support.support_key || buildFewshotSupportKey(support));
}

function findAnnotationGroupForPrediction(pred) {
    const groups = annotLayer.getChildren().filter(g => g.name() === 'annotation');
    const annUid = pred?.annotation_uid || '';
    if (annUid) {
        const group = groups.find(g => g.getAttr('uid') === annUid);
        if (group) return group;
    }
    const idx = (pred?.instance_id >= 1 ? pred.instance_id - 1 : pred?.instance_id);
    if (Number.isInteger(idx) && idx >= 0 && idx < groups.length) return groups[idx];
    return null;
}

function fewshotSupportCounts() {
    const counts = {};
    state.fewshot.supports.forEach(item => {
        counts[item.class_id] = (counts[item.class_id] || 0) + 1;
    });
    return counts;
}

function hasCompleteFewshotSupport() {
    if (!state.classes.length) return false;
    const counts = fewshotSupportCounts();
    // 至少 2 个不同类别有 support（允许部分类别缺失，后端自动收窄）
    const coveredClasses = state.classes.filter(cls => (counts[cls.ID] || 0) > 0);
    return coveredClasses.length >= 2;
}

function setFewshotStatus(msg, isError = false) {
    state.fewshot.status = msg || '';
    state.fewshot.statusIsError = !!isError;
    if (els.fewshotStatus) {
        els.fewshotStatus.textContent = state.fewshot.status || '';
        els.fewshotStatus.style.color = isError ? 'var(--danger)' : 'var(--text-2)';
    }
}

function resetFewshotState(message = '') {
    state.fewshot.supports = [];
    state.fewshot.currentResult = null;
    state.fewshot.evalResult = null;
    state.fewshot.lastMode = '';
    state.fewshot.busy = false;
    state.fewshot.supportFilterClass = 'all';
    state.fewshot.evalScope = 'all';
    state.fewshot.rangeStart = '0';
    state.fewshot.rangeEnd = String(Math.max(0, state.images.length - 1));
    setFewshotStatus(message, false);
    renderFewshotPanel();
}

function formatPct(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function refreshFewshotRangeOptions() {
    if (!els.fewshotRangeStart || !els.fewshotRangeEnd) return;
    if (!state.images.length) {
        els.fewshotRangeStart.innerHTML = '<option value="0">无图片</option>';
        els.fewshotRangeEnd.innerHTML = '<option value="0">无图片</option>';
        state.fewshot.rangeStart = '0';
        state.fewshot.rangeEnd = '0';
        return;
    }

    const options = state.images.map((img, idx) =>
        `<option value="${idx}">${String(idx + 1).padStart(3, '0')} · ${img.filename}</option>`
    ).join('');
    els.fewshotRangeStart.innerHTML = options;
    els.fewshotRangeEnd.innerHTML = options;

    const maxIdx = state.images.length - 1;
    const validStart = Number(state.fewshot.rangeStart);
    const validEnd = Number(state.fewshot.rangeEnd);
    state.fewshot.rangeStart = String(Number.isInteger(validStart) && validStart >= 0 && validStart <= maxIdx ? validStart : 0);
    state.fewshot.rangeEnd = String(Number.isInteger(validEnd) && validEnd >= 0 && validEnd <= maxIdx ? validEnd : maxIdx);
    if (Number(state.fewshot.rangeEnd) < Number(state.fewshot.rangeStart)) {
        state.fewshot.rangeEnd = state.fewshot.rangeStart;
    }
    els.fewshotRangeStart.value = state.fewshot.rangeStart;
    els.fewshotRangeEnd.value = state.fewshot.rangeEnd;
}

function getFewshotPromptParams() {
    const usePrompts = state.fewshot.usePrompts;
    const imgW = Math.max(0, Math.min(1, Number(els.fewshotImageProtoWeight?.value) || 0.5));
    const txtW = Math.max(0, Math.min(1, Number(els.fewshotTextProtoWeight?.value) || 0.5));
    let promptEnsembles = state.fewshot.promptEnsembles || null;
    if (state.fewshot.promptMode === 'custom' && els.fewshotCustomPrompts) {
        const collected = {};
        els.fewshotCustomPrompts.querySelectorAll('.fewshot-class-prompt-input').forEach(inp => {
            const cid = inp.dataset.classId;
            if (!cid) return;
            const parts = inp.value.split(/[,，]/).map(s => s.trim()).filter(Boolean);
            if (parts.length) collected[cid] = parts;
        });
        if (Object.keys(collected).length) promptEnsembles = collected;
    }
    return {
        use_prompts: usePrompts,
        prompt_mode: state.fewshot.promptMode,
        prompt_ensembles: promptEnsembles,
        image_proto_weight: usePrompts ? imgW : 0.5,
        text_proto_weight: usePrompts ? txtW : 0.5,
        primary_prompt_weight: 0.75,
    };
}

function getFewshotEvalRange() {
    const total = state.images.length;
    if (!total) return { start: 0, end: -1, label: '无图片' };

    if (state.fewshot.evalScope === 'current_to_end') {
        const start = Math.max(0, state.currentImageIndex);
        const end = total - 1;
        return { start, end, label: `${start + 1} - ${end + 1}` };
    }
    if (state.fewshot.evalScope === 'range') {
        const start = Math.max(0, Math.min(total - 1, Number(state.fewshot.rangeStart) || 0));
        const end = Math.max(start, Math.min(total - 1, Number(state.fewshot.rangeEnd) || start));
        return { start, end, label: `${start + 1} - ${end + 1}` };
    }
    return { start: 0, end: total - 1, label: `1 - ${total}` };
}

function updateClassifyMethodUI() {
    const isHybrid = state.fewshot.classifyMethod === 'hybrid';
    if (els.hybridSettingsCollapsible) els.hybridSettingsCollapsible.style.display = isHybrid ? '' : 'none';
    if (els.hybridPromptCollapsible) els.hybridPromptCollapsible.style.display = isHybrid ? '' : 'none';
    if (els.fewshotPromptCollapsible) els.fewshotPromptCollapsible.style.display = isHybrid ? 'none' : '';
    if (els.basicTempRow) els.basicTempRow.style.display = isHybrid ? 'none' : '';
    if (els.fewshotBatchClassifyBtn) {
        els.fewshotBatchClassifyBtn.style.display = isHybrid ? '' : 'none';
    }
}

function getHybridParams() {
    return {
        enable_size_refiner: els.hybridEnableSizeRefiner?.checked ?? true,
        size_refiner_trigger_margin: Number(els.hybridSrMargin?.value) || 0.12,
        size_refiner_min_separation: Number(els.hybridSrSeparation?.value) || 1.0,
        size_refiner_score_scale: Number(els.hybridSrScale?.value) || 0.06,
        size_refiner_max_adjust: Number(els.hybridSrMaxAdjust?.value) || 0.08,
    };
}

function getHybridPromptParams() {
    const template = (els.hybridPromptTemplate?.value || '').trim() || null;
    // 从 state 里取，过滤空值
    const raw = state.fewshot.hybridTextPromptNames || {};
    const names = {};
    for (const [cid, phrase] of Object.entries(raw)) {
        const p = (phrase || '').trim();
        if (p) names[cid] = p;
    }
    return {
        prompt_template: template,
        text_prompt_names: Object.keys(names).length ? names : null,
    };
}

function renderFewshotPanel() {
    const counts = fewshotSupportCounts();
    const selectedPayload = getSelectedAnnotationPayload();
    const selectedExists = !!selectedPayload && state.fewshot.supports.some(item => annotationMatchesSupport(selectedPayload, item));
    const canRun = state.fewshot.supports.length > 0 && hasCompleteFewshotSupport() && !state.fewshot.busy;
    const supportFilter = state.fewshot.supportFilterClass;
    const filteredSupports = supportFilter === 'all'
        ? state.fewshot.supports
        : state.fewshot.supports.filter(item => String(item.class_id) === String(supportFilter));
    const evalRange = getFewshotEvalRange();
    const result = state.fewshot.lastMode === 'eval' ? state.fewshot.evalResult : state.fewshot.currentResult;

    if (els.fewshotSupportCounts) {
        const classesWithSupport = state.classes.filter(cls => (counts[cls.ID] || 0) > 0);
        if (classesWithSupport.length === 0) {
            els.fewshotSupportCounts.innerHTML = '';
        } else {
            els.fewshotSupportCounts.innerHTML = classesWithSupport.map(cls => {
                const count = counts[cls.ID] || 0;
                const color = state.colors[cls.ID] || '#fff';
                return `<span class="fewshot-chip">
                    <span style="width:6px; height:6px; border-radius:50%; background:${color}; flex-shrink:0;"></span>
                    <strong>${cls.Name}</strong>
                    <span class="fewshot-chip-count">${count}</span>
                </span>`;
            }).join('');
        }
    }

    if (els.fewshotSupportHeaderCount) {
        const n = state.fewshot.supports.length;
        els.fewshotSupportHeaderCount.textContent = n ? `(${n})` : '(0)';
    }
    if (els.fewshotSupportFilter) {
        els.fewshotSupportFilter.innerHTML = [
            '<option value="all">全部类别</option>',
            ...state.classes.map(cls => `<option value="${cls.ID}">${cls.Name}</option>`),
        ].join('');
        els.fewshotSupportFilter.value = state.fewshot.supportFilterClass;
    }

    if (els.fewshotAddSelectedBtn) {
        els.fewshotAddSelectedBtn.disabled = !selectedPayload || state.fewshot.busy || selectedExists;
        els.fewshotAddSelectedBtn.textContent = selectedExists ? '已加入' : '+ 当前';
    }
    if (els.fewshotRemoveSelectedBtn) {
        els.fewshotRemoveSelectedBtn.disabled = !selectedExists || state.fewshot.busy;
        els.fewshotRemoveSelectedBtn.textContent = '- 当前';
    }
    if (els.fewshotClearBtn) {
        els.fewshotClearBtn.disabled = state.fewshot.supports.length === 0 || state.fewshot.busy;
    }
    if (els.fewshotRunCurrentBtn) {
        els.fewshotRunCurrentBtn.disabled = !canRun || state.currentImageIndex < 0;
    }
    if (els.fewshotEvalSubsetBtn) {
        els.fewshotEvalSubsetBtn.disabled = !canRun;
    }
    if (els.fewshotBatchClassifyBtn) {
        els.fewshotBatchClassifyBtn.disabled = !canRun || state.fewshot.busy;
    }
    if (els.fewshotEvalScope) {
        els.fewshotEvalScope.value = state.fewshot.evalScope;
    }
    if (els.fewshotRangeStart) {
        els.fewshotRangeStart.value = state.fewshot.rangeStart;
        els.fewshotRangeStart.disabled = state.fewshot.evalScope !== 'range';
    }
    if (els.fewshotRangeEnd) {
        els.fewshotRangeEnd.value = state.fewshot.rangeEnd;
        els.fewshotRangeEnd.disabled = state.fewshot.evalScope !== 'range';
    }
    if (els.fewshotRangeHint) {
        const scopeLabel = state.fewshot.evalScope === 'all'
            ? '全部图片'
            : state.fewshot.evalScope === 'current_to_end'
                ? '从当前图到末尾'
                : '自定义范围';
        els.fewshotRangeHint.textContent = `本次评估范围: ${scopeLabel} · ${evalRange.label}`;
    }

    if (els.fewshotUsePrompts) {
        els.fewshotUsePrompts.checked = state.fewshot.usePrompts;
    }
    if (els.fewshotPromptOptions) {
        els.fewshotPromptOptions.style.display = state.fewshot.usePrompts ? '' : 'none';
    }
    if (els.fewshotPromptMode) {
        els.fewshotPromptMode.value = state.fewshot.promptMode;
    }
    if (els.fewshotImageProtoWeight) {
        els.fewshotImageProtoWeight.value = String(state.fewshot.imageProtoWeight);
    }
    if (els.fewshotTextProtoWeight) {
        els.fewshotTextProtoWeight.value = String(state.fewshot.textProtoWeight);
    }
    // ── Hybrid text prompt names (per-class, any number of classes) ──
    if (els.hybridPromptTemplate) {
        if (document.activeElement !== els.hybridPromptTemplate) {
            els.hybridPromptTemplate.value = state.fewshot.hybridPromptTemplate || '';
        }
    }
    if (els.hybridTextPrompts && state.classes.length) {
        const safe = (s) => String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
        const counts = fewshotSupportCounts();
        // Only re-render if class list changed (avoid clobbering live input)
        const currentCids = [...els.hybridTextPrompts.querySelectorAll('.hybrid-tp-input')].map(i => i.dataset.classId).join(',');
        const newCids = state.classes.map(c => String(c.ID)).join(',');
        if (currentCids !== newCids) {
            els.hybridTextPrompts.innerHTML = state.classes.map(cls => {
                const cid = String(cls.ID);
                const phrase = (state.fewshot.hybridTextPromptNames[cid] || '').replace(/"/g, '&quot;');
                const hasSupport = (counts[cls.ID] || 0) > 0;
                const dot = hasSupport ? '●' : '○';
                const dotColor = hasSupport ? 'var(--success)' : 'var(--text-3)';
                return `<div style="display:flex; align-items:center; gap:4px; font-size:9px;">
                    <span style="color:${dotColor}; flex-shrink:0;" title="${hasSupport?'有 support':'无 support'}">${dot}</span>
                    <span style="color:var(--text-2); min-width:38px; flex-shrink:0;">${safe(cls.Name)}</span>
                    <input type="text" class="select-input hybrid-tp-input" data-class-id="${cid}"
                        placeholder="${safe(cls.Name)} 的描述词（留空用类别名）"
                        value="${phrase}"
                        style="flex:1; font-size:9px; padding:2px 4px;">
                </div>`;
            }).join('');
            els.hybridTextPrompts.querySelectorAll('.hybrid-tp-input').forEach(inp => {
                inp.oninput = () => {
                    state.fewshot.hybridTextPromptNames[inp.dataset.classId] = inp.value;
                    persistHybridPromptNames();
                };
            });
        } else {
            // Just refresh the dot colors if support changed
            els.hybridTextPrompts.querySelectorAll('.hybrid-tp-input').forEach(inp => {
                const cid = parseInt(inp.dataset.classId);
                const hasSupport = (counts[cid] || 0) > 0;
                const dot = inp.closest('div')?.querySelector('span');
                if (dot) {
                    dot.textContent = hasSupport ? '●' : '○';
                    dot.style.color = hasSupport ? 'var(--success)' : 'var(--text-3)';
                }
            });
        }
    }

    if (els.fewshotCustomPrompts) {
        els.fewshotCustomPrompts.style.display = state.fewshot.promptMode === 'custom' ? '' : 'none';
        if (state.fewshot.promptMode === 'custom') {
            els.fewshotCustomPrompts.innerHTML = state.classes.map(cls => {
                const prompts = state.fewshot.promptEnsembles[String(cls.ID)] || [];
                const val = prompts.length ? prompts.join(', ') : '';
                const safe = (s) => String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
                return `<div class="fewshot-class-prompt">
                    <div class="fewshot-class-prompt-label">${safe(cls.Name)}</div>
                    <input type="text" class="fewshot-class-prompt-input" data-class-id="${cls.ID}" placeholder="如: a photomicrograph of a ${safe(cls.Name)} cell" value="${safe(val)}">
                </div>`;
            }).join('');
            els.fewshotCustomPrompts.querySelectorAll('.fewshot-class-prompt-input').forEach(inp => {
                inp.oninput = () => {
                    const cid = String(inp.dataset.classId);
                    const parts = inp.value.split(/[,，]/).map(s => s.trim()).filter(Boolean);
                    state.fewshot.promptEnsembles[cid] = parts.length ? parts : [`a photomicrograph of a cell`];
                };
            });
        }
    }

    if (els.fewshotSupportList) {
        if (!filteredSupports.length) {
            els.fewshotSupportList.innerHTML = '<div class="fewshot-empty">暂无 support</div>';
        } else {
            els.fewshotSupportList.innerHTML = filteredSupports.map(item => {
                const cls = state.classes.find(c => parseInt(c.ID) === parseInt(item.class_id));
                const name = cls ? cls.Name : `Class ${item.class_id}`;
                const color = state.colors[item.class_id] || '#fff';
                const expanded = state.fewshot.expandedSupportKey === item.support_key ? 'expanded' : '';
                return `<div class="fewshot-support-item ${expanded}" data-key="${item.support_key}">
                            <div class="fewshot-support-head" data-key="${item.support_key}">
                                <div class="fewshot-support-left">
                                    <span class="fewshot-support-arrow">▸</span>
                                    <div class="fewshot-support-main">
                                        <div class="fewshot-support-title">
                                            <span style="color:${color}; font-weight:700;">●</span> ${name}
                                        </div>
                                    </div>
                                </div>
                                <button class="fewshot-support-remove" data-key="${item.support_key}" title="移除">×</button>
                            </div>
                            <div class="fewshot-support-sub">${item.subset} · ${item.filename}</div>
                        </div>`;
            }).join('');
            els.fewshotSupportList.querySelectorAll('.fewshot-support-head').forEach(head => {
                head.onclick = (e) => {
                    if (e.target.closest('.fewshot-support-remove')) return;
                    const key = head.dataset.key;
                    state.fewshot.expandedSupportKey = state.fewshot.expandedSupportKey === key ? null : key;
                    renderFewshotPanel();
                };
            });
            els.fewshotSupportList.querySelectorAll('.fewshot-support-remove').forEach(btn => {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    state.fewshot.supports = state.fewshot.supports.filter(item => item.support_key !== btn.dataset.key);
                    if (state.fewshot.expandedSupportKey === btn.dataset.key) {
                        state.fewshot.expandedSupportKey = null;
                    }
                    state.fewshot.currentResult = null;
                    state.fewshot.evalResult = null;
                    setFewshotStatus('已移除一个 support 样本');
                    persistSupports();
                    renderFewshotPanel();
                };
            });
        }
    }

    if (els.fewshotResultPanel) {
        els.fewshotResultPanel.classList.toggle('fewshot-hidden', !result);
    }
    if (els.fewshotMetrics) {
        if (!result) {
            els.fewshotMetrics.innerHTML = '';
        } else if (state.fewshot.lastMode === 'eval') {
            const m = result.metrics || {};
            const total = m.total ?? (result.sample_predictions?.length ?? 0);
            const acc = m.accuracy, f1 = m.macro_f1;
            let html = `<div class="sc-metric-cards" style="margin-top:0;">`;
            html += `<div class="sc-metric-card"><div class="sc-metric-val">${total}</div><div class="sc-metric-label">Cells</div></div>`;
            if (acc != null) html += `<div class="sc-metric-card"><div class="sc-metric-val">${(acc*100).toFixed(1)}%</div><div class="sc-metric-label">Acc</div></div>`;
            if (f1 != null) html += `<div class="sc-metric-card"><div class="sc-metric-val">${(f1*100).toFixed(1)}%</div><div class="sc-metric-label">F1</div></div>`;
            html += `</div>`;
            els.fewshotMetrics.innerHTML = html;
        } else {
            const total = result.predictions?.length ?? 0;
            els.fewshotMetrics.innerHTML = `<div class="fewshot-metric">
                <span class="fewshot-metric-label">当前图细胞数</span>
                <span class="fewshot-metric-value">${total}</span>
            </div>`;
        }
    }

    if (els.fewshotResultList) {
        if (!result) {
            els.fewshotResultList.innerHTML = '';
        } else if (state.fewshot.lastMode === 'eval') {
            const rows = (result.sample_predictions || []);
            if (!rows.length) {
                els.fewshotResultList.innerHTML = '<div class="fewshot-empty">无预测结果</div>';
            } else {
                els.fewshotResultList.innerHTML = rows.map(item =>
                    `<div class="fewshot-result-item">
                        <div>${item.filename} · Cell ${item.instance_id}</div>
                        <div>预测: ${item.pred_class_name}</div>
                        <div style="color:var(--text-2)">置信度 ${formatPct(item.confidence)}</div>
                    </div>`
                ).join('');
            }
        } else {
            const rows = (result.predictions || []);
            els.fewshotResultList.innerHTML = rows.map(item =>
                `<div class="fewshot-result-item">
                    <div>Cell ${item.instance_id} · 预测: ${item.pred_class_name}</div>
                    <div style="color:var(--text-2)">置信度 ${formatPct(item.confidence)}</div>
                </div>`
            ).join('');
        }
    }

    const hasCurrentPreview = result && state.fewshot.lastMode === 'current';
    if (els.fewshotPreviewActions) {
        els.fewshotPreviewActions.style.display = hasCurrentPreview ? '' : 'none';
    }
    if (els.fewshotApplyBtn) {
        els.fewshotApplyBtn.disabled = state.fewshot.busy || !hasCurrentPreview;
    }
    if (els.fewshotRevertBtn) {
        els.fewshotRevertBtn.disabled = state.fewshot.busy || !state.fewshot._previewBackup;
    }

    if (els.fewshotStatus) {
        els.fewshotStatus.textContent = state.fewshot.status;
        els.fewshotStatus.style.color = state.fewshot.statusIsError ? 'var(--danger)' : 'var(--text-2)';
    }
    renderResearchEvalPanel();
    updateStatusBar();
}

function renderResearchEvalPanel() {
    if (!els.evalGoldLabelset || !els.evalPromptLabelset) return;
    const labelSets = state.currentGroup?.label_sets || [];
    const prevGold = els.evalGoldLabelset.value;
    const prevPrompt = els.evalPromptLabelset.value;
    const options = labelSets.length
        ? labelSets.map(ls => `<option value="${ls.set_id}">${ls.set_name} (${ls.label_format})</option>`).join('')
        : '<option value="">无可用标注集</option>';
    els.evalGoldLabelset.innerHTML = options;
    els.evalPromptLabelset.innerHTML = options;
    if (prevGold && labelSets.some(ls => ls.set_id === prevGold)) {
        els.evalGoldLabelset.value = prevGold;
    } else {
        const currentId = state.currentLabelSet?.set_id || '';
        if (currentId && labelSets.some(ls => ls.set_id === currentId)) {
            els.evalGoldLabelset.value = currentId;
        }
    }
    if (prevPrompt && labelSets.some(ls => ls.set_id === prevPrompt)) {
        els.evalPromptLabelset.value = prevPrompt;
    } else {
        const currentId = state.currentLabelSet?.set_id || '';
        const bboxSet = labelSets.find(ls => ls.label_format === 'bbox');
        els.evalPromptLabelset.value = bboxSet?.set_id || currentId;
    }
    const usesSam3 = (els.evalSegmentMethod?.value || 'cellpose') === 'sam3';
    els.evalPromptLabelset.disabled = !usesSam3;
    if (els.evalClassifyGoldBtn) {
        els.evalClassifyGoldBtn.disabled = !state.currentGroup || !state.currentLabelSet || state.fewshot.busy;
    }
    if (els.evalSegmentGoldBtn) {
        els.evalSegmentGoldBtn.disabled = !state.currentGroup || state.fewshot.busy;
    }
}

function setResearchEvalStatus(msg, isError = false) {
    if (!els.evalStatus) return;
    els.evalStatus.textContent = msg || '';
    els.evalStatus.style.color = isError ? 'var(--danger)' : 'var(--text-2)';
    els.evalStatus.classList.toggle('has-result', !!msg && !isError);
}

function setFewshotSupportPopoverVisible(visible) {
    if (!els.fewshotSupportPopover) return;
    els.fewshotSupportPopover.classList.toggle('collapsed-panel', !visible);
}

function addSelectedAnnotationToFewshot() {
    const payload = getSelectedAnnotationPayload();
    if (!payload) {
        setFewshotStatus('请先选中一个标注框或轮廓。', true);
        renderFewshotPanel();
        return;
    }
    const support = { ...payload, support_key: buildFewshotSupportKey(payload) };
    if (state.fewshot.supports.some(item => item.support_key === support.support_key)) {
        setFewshotStatus('这个 support 已经加入过了。');
        renderFewshotPanel();
        return;
    }
    state.fewshot.supports.push(support);
    state.fewshot.currentResult = null;
    state.fewshot.evalResult = null;
    setFewshotStatus(`已加入 support: ${support.filename}`);
    persistSupports();
    renderFewshotPanel();
}

function addSelectedAnnotationsAsSupports() {
    if (selectedGroups.size === 0 || state.currentImageIndex < 0) {
        setFewshotStatus('请先勾选要添加的标注（多选）。', true);
        renderFewshotPanel();
        return;
    }
    const currentImage = state.images[state.currentImageIndex];
    if (!currentImage) return;
    const filename = currentImage.filename;
    const subset = state.subset;
    let added = 0;
    selectedGroups.forEach(group => {
        const pts = group.getAttr('normPoints');
        const payload = {
            filename,
            subset,
            class_id: group.getAttr('classId'),
            ann_type: group.getAttr('annType') || 'polygon',
            points: Array.isArray(pts) ? pts.slice() : [],
            annotation_uid: group.getAttr('uid'),
        };
        const support_key = buildFewshotSupportKey(payload);
        if (state.fewshot.supports.some(item => annotationMatchesSupport(payload, item))) return;
        state.fewshot.supports.push({ ...payload, support_key });
        added++;
    });
    state.fewshot.currentResult = null;
    state.fewshot.evalResult = null;
    if (added > 0) {
        persistSupports();
        setFewshotStatus(`已添加 ${added} 个 support。`);
    } else {
        setFewshotStatus('选中的标注都已存在于 support 列表中。');
    }
    renderFewshotPanel();
}

function removeCurrentSelectedFromFewshot() {
    const payload = getSelectedAnnotationPayload();
    if (!payload) {
        setFewshotStatus('请先选中一个 support 标注。', true);
        renderFewshotPanel();
        return;
    }
    const prevCount = state.fewshot.supports.length;
    state.fewshot.supports = state.fewshot.supports.filter(item => !annotationMatchesSupport(payload, item));
    if (state.fewshot.supports.length === prevCount) {
        setFewshotStatus('当前选中标注不在 support 列表中。');
    } else {
        state.fewshot.currentResult = null;
        state.fewshot.evalResult = null;
        setFewshotStatus('已移除当前选中的 support。');
        persistSupports();
    }
    renderFewshotPanel();
}

function clearFewshotSupports(showMsg = true) {
    state.fewshot.supports = [];
    state.fewshot.currentResult = null;
    state.fewshot.evalResult = null;
    state.fewshot.lastMode = '';
    setFewshotStatus(showMsg ? '已清空全部 support。' : '');
    persistSupports();
    renderFewshotPanel();
}

async function persistSupports() {
    if (!state.currentGroup) return;
    const labelSetId = state.currentLabelSet?.set_id || '';
    try {
        await fetch(`${API}/supports/save`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                group_id: state.currentGroup.group_id,
                label_set_id: labelSetId,
                supports: state.fewshot.supports,
            }),
        });
    } catch (e) {
        console.warn('persistSupports failed:', e);
    }
}

function persistHybridPromptNames() {
    try {
        const key = `hybridTextPrompts_${state.currentGroup?.group_id || '_global'}`;
        localStorage.setItem(key, JSON.stringify({
            template: state.fewshot.hybridPromptTemplate || '',
            names: state.fewshot.hybridTextPromptNames || {},
        }));
    } catch (e) { /* quota / private */ }
}

function loadHybridPromptNames() {
    try {
        const key = `hybridTextPrompts_${state.currentGroup?.group_id || '_global'}`;
        const raw = localStorage.getItem(key);
        if (!raw) return;
        const data = JSON.parse(raw);
        state.fewshot.hybridPromptTemplate = data.template || '';
        state.fewshot.hybridTextPromptNames = data.names || {};
    } catch (e) { /* ignore */ }
}

async function loadSavedSupports() {
    if (!state.currentGroup) return;
    const labelSetId = state.currentLabelSet?.set_id || '';
    try {
        const url = `${API}/supports?group_id=${encodeURIComponent(state.currentGroup.group_id)}`
            + `&label_set_id=${encodeURIComponent(labelSetId)}&validate=true`;
        const res = await fetch(url);
        if (!res.ok) return;
        const data = await res.json();
        const items = data.supports || [];
        if (!items.length) {
            state.fewshot.supports = [];
            renderFewshotPanel();
            return;
        }
        state.fewshot.supports = items.map(item => ({
            ...item,
            support_key: item.support_key || buildFewshotSupportKey(item),
        }));
        const source = data.source === 'default' ? '默认' : '已保存';
        let msg = `已加载${source} support (${items.length} 个)`;
        if (data.validation?.removed_count > 0) {
            msg += `，移除了 ${data.validation.removed_count} 个失效项`;
        }
        setFewshotStatus(msg);
        renderFewshotPanel();
    } catch (e) {
        console.warn('loadSavedSupports failed:', e);
    }
}

function previewFewshotClassification(result) {
    if (!result?.predictions?.length) return;
    const backup = [];

    result.predictions.forEach(pred => {
        const group = findAnnotationGroupForPrediction(pred);
        if (!group) return;
        backup.push({
            uid: group.getAttr('uid'),
            classId: group.getAttr('classId'),
            stroke: group.findOne('Line')?.stroke(),
        });
        const color = state.colors[pred.pred_class_id] || '#fff';
        const poly = group.findOne('Line');
        if (poly) {
            poly.stroke(color);
            poly.opacity(pred.correct === false ? 0.5 : 1.0);
        }
    });
    if (!backup.length) return;
    state.fewshot._previewBackup = backup;
    annotLayer.batchDraw();
    if (els.fewshotPreviewActions) els.fewshotPreviewActions.style.display = '';
}

function revertFewshotPreview() {
    const backup = state.fewshot._previewBackup;
    if (!backup) return;
    backup.forEach((b) => {
        const group = annotLayer.getChildren().find(g => g.name() === 'annotation' && g.getAttr('uid') === b.uid);
        if (!group) return;
        group.setAttr('classId', b.classId);
        const poly = group.findOne('Line');
        if (poly) {
            poly.stroke(b.stroke || '#fff');
            poly.opacity(1.0);
        }
    });
    annotLayer.batchDraw();
    state.fewshot._previewBackup = null;
    state.fewshot.currentResult = null;
    state.fewshot.lastMode = '';
    if (els.fewshotPreviewActions) els.fewshotPreviewActions.style.display = 'none';
    setFewshotStatus('已撤销预览');
    renderFewshotPanel();
}

function applyFewshotResult() {
    const result = state.fewshot.currentResult;
    if (!result || state.fewshot.lastMode !== 'current' || !result.predictions?.length) return;
    const matchedGroups = result.predictions.map(pred => findAnnotationGroupForPrediction(pred)).filter(Boolean);
    if (matchedGroups.length !== result.predictions.length) {
        setFewshotStatus('部分预测结果已无法定位到当前标注，请重新预测。', true);
        renderFewshotPanel();
        return;
    }
    pushUndoState();
    result.predictions.forEach(pred => {
        const group = findAnnotationGroupForPrediction(pred);
        if (!group) return;
        const newClassId = pred.pred_class_id;
        group.setAttr('classId', newClassId);
        const color = state.colors[newClassId] || '#fff';
        const poly = group.findOne('Line');
        if (poly) {
            poly.stroke(color);
            poly.opacity(1.0);
        }
    });
    state.fewshot._previewBackup = null;
    setDirty(true);
    renderClassList();
    renderAnnotationList();
    renderVisualization();
    setFewshotStatus('已应用预测结果到标注，请保存。');
    showStatus('已应用 Few-shot 预测结果');
    renderFewshotPanel();
}

async function runFewshotCurrentPrediction() {
    if (!state.currentGroup || !state.currentLabelSet || state.currentImageIndex < 0) {
        showStatus('请先选择图片和标注集', true);
        return;
    }
    if (!hasCompleteFewshotSupport()) {
        setFewshotStatus('每个类别至少要有 1 个 support 才能运行。', true);
        renderFewshotPanel();
        return;
    }

    const isHybrid = state.fewshot.classifyMethod === 'hybrid';
    const methodLabel = isHybrid ? 'Hybrid Adaptive' : 'Few-shot';

    if (!isHybrid) {
        const temperature = Number(els.fewshotTemperature?.value || 1);
        if (!(temperature > 0)) {
            setFewshotStatus('Temperature 必须大于 0。', true);
            renderFewshotPanel();
            return;
        }
    }

    state.fewshot.busy = true;
    state.fewshot.evalResult = null;
    renderFewshotPanel();
    showLoading(true, `${methodLabel} 当前图预测中`);
    try {
        let endpoint, body;
        if (isHybrid) {
            endpoint = `${API}/hybrid/predict_current`;
            body = {
                group_id: state.currentGroup.group_id,
                label_set_id: state.currentLabelSet.set_id,
                subset: state.subset,
                filename: state.images[state.currentImageIndex].filename,
                support_items: state.fewshot.supports,
                query_annotations: getCurrentAnnotationsPayload(),
                ...getHybridParams(),
                ...getHybridPromptParams(),
            };
        } else {
            endpoint = `${API}/fewshot/predict_current`;
            body = {
                group_id: state.currentGroup.group_id,
                label_set_id: state.currentLabelSet.set_id,
                subset: state.subset,
                filename: state.images[state.currentImageIndex].filename,
                support_items: state.fewshot.supports,
                query_annotations: getCurrentAnnotationsPayload(),
                temperature: Number(els.fewshotTemperature?.value || 1),
                ...getFewshotPromptParams(),
            };
        }
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        state.fewshot.currentResult = data;
        state.fewshot.lastMode = 'current';
        previewFewshotClassification(data);
        setFewshotStatus(`当前图预测完成，共 ${data.predictions?.length ?? 0} 个细胞。点击"应用"保存结果。`);
        showStatus(`${methodLabel} 当前图预测完成`);
    } catch (e) {
        setFewshotStatus(e.message || `${methodLabel} 当前图预测失败`, true);
        showStatus(`${methodLabel} 预测失败: ${e.message}`, true);
    } finally {
        state.fewshot.busy = false;
        showLoading(false);
        renderFewshotPanel();
    }
}

async function runFewshotBatchClassifyAndSave() {
    if (!state.currentGroup || !state.currentLabelSet) {
        setFewshotStatus('请先选择图片集和标注集。', true);
        renderFewshotPanel();
        return;
    }
    if (!hasCompleteFewshotSupport()) {
        setFewshotStatus('每个类别至少要有 1 个 support 才能批量分类。', true);
        renderFewshotPanel();
        return;
    }
    if (state.fewshot.classifyMethod !== 'hybrid') {
        setFewshotStatus('批量分类并保存仅支持 Hybrid Adaptive 模式。', true);
        renderFewshotPanel();
        return;
    }
    const evalRange = getFewshotEvalRange();
    if (evalRange.end < evalRange.start) {
        setFewshotStatus('当前评估范围无效。', true);
        renderFewshotPanel();
        return;
    }
    state.fewshot.busy = true;
    renderFewshotPanel();
    const total = evalRange.end - evalRange.start + 1;
    const abortSignal = startLongRun();
    showLoading(true, `批量分类 0/${total}…`, { stoppable: true });
    let done = 0;
    let failed = 0;
    let stopped = false;
    try {
        for (let idx = evalRange.start; idx <= evalRange.end; idx++) {
            if (abortSignal.aborted) { stopped = true; break; }
            showLoading(true, `批量分类 ${idx - evalRange.start + 1}/${total}…`, { stoppable: true });
            await selectImage(idx);
            const filename = state.images[state.currentImageIndex].filename;
            const queryAnns = getCurrentAnnotationsPayload().filter(
                ann => !state.fewshot.supports.some(support => annotationMatchesSupport(ann, support))
            );
            if (!queryAnns || queryAnns.length === 0) continue;
            try {
                const res = await fetch(`${API}/hybrid/predict_current`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        group_id: state.currentGroup.group_id,
                        label_set_id: state.currentLabelSet.set_id,
                        subset: state.subset,
                        filename,
                        support_items: state.fewshot.supports,
                        query_annotations: queryAnns,
                        ...getHybridParams(),
                        ...getHybridPromptParams(),
                    }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
                state.fewshot.currentResult = data;
                state.fewshot.lastMode = 'current';
                applyFewshotResult();
                const ok = await saveAnnotations();
                if (ok) done++; else failed++;
            } catch (e) {
                failed++;
                console.warn(`Batch classify failed for ${filename}:`, e);
            }
        }
        const stopLabel = stopped ? ' (已停止)' : '';
        setFewshotStatus(`批量分类完成${stopLabel}: ${done} 张已保存${failed > 0 ? `, ${failed} 张失败` : ''}`, failed > 0);
        showStatus(stopped ? `已停止，${done} 张已保存` : (failed > 0 ? `批量完成，${failed} 张失败` : '批量分类并保存完成'));
    } finally {
        _longRunAbort = null;
        state.fewshot.busy = false;
        showLoading(false);
        renderFewshotPanel();
    }
}

async function runFewshotSubsetEvaluation() {
    if (!state.currentGroup || !state.currentLabelSet) {
        showStatus('请先选择图片集和标注集', true);
        return;
    }
    if (!hasCompleteFewshotSupport()) {
        setFewshotStatus('每个类别至少要有 1 个 support 才能评估。', true);
        renderFewshotPanel();
        return;
    }
    const isHybrid = state.fewshot.classifyMethod === 'hybrid';
    const maxImagesRaw = (els.hybridMaxImages?.value?.trim()) || '';
    const maxImages = maxImagesRaw ? Number(maxImagesRaw) : null;
    const evalRange = getFewshotEvalRange();
    if (maxImagesRaw && !(maxImages > 0)) {
        setFewshotStatus('评估图片数需要是正整数。', true);
        renderFewshotPanel();
        return;
    }
    if (evalRange.end < evalRange.start) {
        setFewshotStatus('当前评估图片范围无效。', true);
        renderFewshotPanel();
        return;
    }

    state.fewshot.busy = true;
    state.fewshot.currentResult = null;
    renderFewshotPanel();
    const endpoint = isHybrid ? `${API}/hybrid/evaluate_subset` : `${API}/fewshot/evaluate_subset`;
    const methodLabel = isHybrid ? 'Hybrid Adaptive' : 'Few-shot';
    if (!isHybrid) {
        const temperature = Number(els.fewshotTemperature?.value || 1);
        if (!(temperature > 0)) {
            setFewshotStatus('Temperature 必须大于 0。', true);
            state.fewshot.busy = false;
            renderFewshotPanel();
            return;
        }
    }
    const abortSignal = startLongRun();
    showLoading(true, `${methodLabel} 子集评估中`, { stoppable: true });
    try {
        const body = {
            group_id: state.currentGroup.group_id,
            label_set_id: state.currentLabelSet.set_id,
            subset: state.subset,
            support_items: state.fewshot.supports,
            max_images: maxImages,
            start_index: evalRange.start,
            end_index: evalRange.end,
        };
        if (isHybrid) {
            Object.assign(body, getHybridParams(), getHybridPromptParams());
        } else {
            Object.assign(body, {
                temperature: Number(els.fewshotTemperature?.value || 1),
                ...getFewshotPromptParams(),
            });
        }
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: abortSignal,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        state.fewshot.evalResult = data;
        state.fewshot.lastMode = 'eval';
        setFewshotStatus(`子集评估完成，共 ${data.sample_predictions?.length ?? 0} 条预测记录`);
        showStatus(`${methodLabel} 子集评估完成`);
    } catch (e) {
        if (e.name === 'AbortError') {
            setFewshotStatus(`${methodLabel} 子集评估已停止`, true);
            showStatus('已停止子集评估');
        } else {
            setFewshotStatus(e.message || `${methodLabel} 子集评估失败`, true);
            showStatus(`${methodLabel} 评估失败: ${e.message}`, true);
        }
    } finally {
        _longRunAbort = null;
        state.fewshot.busy = false;
        showLoading(false);
        renderFewshotPanel();
    }
}

async function runGoldClassificationEvaluation() {
    if (!state.currentGroup || !state.currentLabelSet) {
        setResearchEvalStatus('请先选择图片集和标注集。', true);
        return;
    }
    if (!hasCompleteFewshotSupport()) {
        setResearchEvalStatus('请先为每个类别准备完整的 support。', true);
        return;
    }
    const goldLabelSetId = els.evalGoldLabelset?.value || '';
    if (!goldLabelSetId) {
        setResearchEvalStatus('请选择金标准标注集。', true);
        return;
    }
    const maxImagesRaw = els.evalMaxImages?.value?.trim() || '';
    const maxImages = maxImagesRaw ? Number(maxImagesRaw) : null;
    if (maxImagesRaw && !(maxImages > 0)) {
        setResearchEvalStatus('评估图片数需要是正整数。', true);
        return;
    }

    state.fewshot.busy = true;
    renderFewshotPanel();
    const abortSignal = startLongRun();
    showLoading(true, '分类金标准评估中', { stoppable: true });
    setResearchEvalStatus('正在运行分类金标准评估…');
    clearEvalResultDetail();
    try {
        const res = await fetch(`${API}/hybrid/evaluate_gold`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                group_id: state.currentGroup.group_id,
                label_set_id: state.currentLabelSet.set_id,
                gold_label_set_id: goldLabelSetId,
                subset: state.subset,
                support_items: state.fewshot.supports,
                max_images: maxImages,
                ...getHybridParams(),
                ...getHybridPromptParams(),
            }),
            signal: abortSignal,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        renderClassifyGoldResult(data);
        showStatus('分类金标准评估完成');
    } catch (e) {
        if (e.name === 'AbortError') {
            setResearchEvalStatus('分类评估已被用户停止', true);
            showStatus('已停止分类评估');
        } else {
            setResearchEvalStatus(`分类评估失败：${e.message}`, true);
            showStatus(`分类金标准评估失败: ${e.message}`, true);
        }
        clearEvalResultDetail();
    } finally {
        _longRunAbort = null;
        state.fewshot.busy = false;
        showLoading(false);
        renderFewshotPanel();
    }
}

function renderClassifyGoldResult(data) {
    const m = data.metrics || {};
    setResearchEvalStatus('分类金标准评估完成');
    const detail = document.getElementById('eval-result-detail');
    if (!detail) return;
    const acc = (m.accuracy ?? 0), f1 = (m.macro_f1 ?? 0);
    const cells = data.cell_count ?? m.total ?? 0;
    let html = `<div class="sc-metric-cards">
        <div class="sc-metric-card"><div class="sc-metric-val">${(acc*100).toFixed(1)}%</div><div class="sc-metric-label">Accuracy</div></div>
        <div class="sc-metric-card"><div class="sc-metric-val">${(f1*100).toFixed(1)}%</div><div class="sc-metric-label">Macro-F1</div></div>
        <div class="sc-metric-card"><div class="sc-metric-val">${cells}</div><div class="sc-metric-label">Cells</div></div>
    </div>`;
    if (m.per_class_f1 || m.per_class) {
        const pc = m.per_class_f1 || m.per_class || {};
        html += `<table class="sc-per-class-table"><tr><th>类别</th><th>F1</th><th>P</th><th>R</th></tr>`;
        for (const [cls, val] of Object.entries(pc)) {
            if (typeof val === 'object') {
                html += `<tr><td>${cls}</td><td>${(val.f1??0).toFixed(3)}</td><td>${(val.precision??0).toFixed(3)}</td><td>${(val.recall??0).toFixed(3)}</td></tr>`;
            } else {
                html += `<tr><td>${cls}</td><td colspan="3">${Number(val).toFixed(3)}</td></tr>`;
            }
        }
        html += `</table>`;
    }
    detail.innerHTML = html;
    detail.style.display = '';
}

function renderSegmentGoldResult(data) {
    const m = data.metrics || {};
    setResearchEvalStatus('分割金标准评估完成');
    const detail = document.getElementById('eval-result-detail');
    if (!detail) return;
    let html = `<div class="sc-metric-cards">
        <div class="sc-metric-card"><div class="sc-metric-val">${((m.precision??0)*100).toFixed(1)}%</div><div class="sc-metric-label">Precision</div></div>
        <div class="sc-metric-card"><div class="sc-metric-val">${((m.recall??0)*100).toFixed(1)}%</div><div class="sc-metric-label">Recall</div></div>
        <div class="sc-metric-card"><div class="sc-metric-val">${((m.mean_matched_dice??0)*100).toFixed(1)}%</div><div class="sc-metric-label">Dice</div></div>
    </div>`;
    if (m.matched_count != null || m.pred_count != null) {
        html += `<div style="font-size:9px; color:var(--text-2); margin-top:4px;">
            预测 ${m.pred_count??'?'} 个 · 匹配 ${m.matched_count??'?'} 个 · 金标准 ${m.gold_count??'?'} 个
        </div>`;
    }
    detail.innerHTML = html;
    detail.style.display = '';
}

function clearEvalResultDetail() {
    const detail = document.getElementById('eval-result-detail');
    if (detail) { detail.innerHTML = ''; detail.style.display = 'none'; }
}

async function runGoldSegmentationEvaluation() {
    if (!state.currentGroup) {
        setResearchEvalStatus('请先选择图片集。', true);
        return;
    }
    const goldLabelSetId = els.evalGoldLabelset?.value || '';
    if (!goldLabelSetId) {
        setResearchEvalStatus('请选择金标准标注集。', true);
        return;
    }
    const method = els.evalSegmentMethod?.value || 'cellpose';
    const maxImagesRaw = els.evalMaxImages?.value?.trim() || '';
    const maxImages = maxImagesRaw ? Number(maxImagesRaw) : null;
    if (maxImagesRaw && !(maxImages > 0)) {
        setResearchEvalStatus('评估图片数需要是正整数。', true);
        return;
    }
    const body = {
        group_id: state.currentGroup.group_id,
        subset: state.subset,
        gold_label_set_id: goldLabelSetId,
        method,
        max_images: maxImages,
    };
    if (method === 'cellpose') {
        body.diameters = _getCellposeDiameters();
        body.gpu = els.cellposeGpu?.checked ?? true;
        body.class_id = parseInt(els.classSelect.value, 10) || 0;
        body.min_area = 100;
    } else {
        body.prompt_label_set_id = els.evalPromptLabelset?.value || '';
        body.prompt_types = _getSelectedStrategies();
        body.prompt_type = _getSamPromptType();
        body.text_prompt = _getSamTextPrompt();
    }

    state.fewshot.busy = true;
    renderFewshotPanel();
    const abortSignal = startLongRun();
    showLoading(true, '分割金标准评估中', { stoppable: true });
    setResearchEvalStatus(`正在运行 ${method === 'cellpose' ? 'CellposeSAM' : 'SAM3'} 金标准评估…`);
    renderEvalResultDetail(null);
    try {
        const res = await fetch(`${API}/segment/evaluate_gold`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: abortSignal,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        renderSegmentGoldResult(data);
        showStatus('分割金标准评估完成');
    } catch (e) {
        if (e.name === 'AbortError') {
            setResearchEvalStatus('分割评估已被用户停止', true);
            showStatus('已停止分割评估');
        } else {
            setResearchEvalStatus(`分割评估失败：${e.message}`, true);
            showStatus(`分割金标准评估失败: ${e.message}`, true);
        }
        clearEvalResultDetail();
    } finally {
        _longRunAbort = null;
        state.fewshot.busy = false;
        showLoading(false);
        renderFewshotPanel();
    }
}

async function addClass(name) {
    if (!state.currentGroup || !name.trim()) return;
    try {
        const res = await fetch(`${API}/add_class`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                group_id: state.currentGroup.group_id,
                class_name: name.trim(),
            }),
        });
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        // Update local state
        state.currentGroup.nc = data.nc;
        state.classes.push({ ID: data.ID, Name: data.Name, Color: data.Color });
        state.colors[data.ID] = data.Color;
        els.classSelect.innerHTML += `<option value="${data.ID}">${data.Name} (${data.ID})</option>`;
        renderClassList();
        showStatus(`已添加类别: ${data.Name}`);
    } catch (e) {
        showStatus('添加类别失败', true);
    }
}

// ── Image list: virtual scroll for large lists ───────────────
const IMG_ITEM_HEIGHT = 30;  // px per image item
let _imageListFilter = '';
let _filteredIndices = [];   // indices into state.images that match filter
let _imageListScrollRAF = null;

function renderImageList(fullRebuild = true) {
    if (fullRebuild) {
        _imageListFilter = els.searchInput.value.toLowerCase();
        const statusFilter = els.imageStatusFilter ? els.imageStatusFilter.value : 'all';
        _filteredIndices = [];
        for (let i = 0; i < state.images.length; i++) {
            const img = state.images[i];
            if (_imageListFilter && !img.filename.toLowerCase().includes(_imageListFilter)) continue;
            if (statusFilter === 'labeled' && !img.has_label) continue;
            if (statusFilter === 'unlabeled' && img.has_label) continue;
            if (statusFilter === 'flagged' && !state._flaggedImages?.has(img.filename)) continue;
            _filteredIndices.push(i);
        }
    }

    // For small lists (< 500), use simple rendering for simplicity
    if (_filteredIndices.length < 500) {
        _renderImageListSimple();
        return;
    }

    // Virtual scroll for large lists
    _renderImageListVirtual();
}

function _renderImageListSimple() {
    if (_filteredIndices.length === 0) {
        const hasImages = state.images.length > 0;
        els.imageList.innerHTML = `<div class="empty-state">
            <strong>${hasImages ? '没有匹配的图片' : '当前子集暂无图片'}</strong>
            <span>${hasImages ? '试试修改搜索词，或切换图片集/子集。' : '先切换图片集、子集，或扫描新的目录。'}</span>
        </div>`;
        return;
    }
    const labeledCount = state.images.filter(i => i.has_label).length;
    const totalCount = state.images.length;
    const pct = totalCount > 0 ? Math.round(labeledCount / totalCount * 100) : 0;
    const progressColor = pct > 80 ? 'var(--success)' : pct > 40 ? 'var(--warn)' : 'var(--danger)';

    let html = `<div style="padding:3px 8px;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;gap:6px;">
        <div style="flex:1;height:3px;background:var(--bg-3);border-radius:2px;overflow:hidden;">
            <div style="width:${pct}%;height:100%;background:${progressColor};border-radius:2px;transition:width 0.3s;"></div>
        </div>
        <span style="font-size:9px;color:var(--text-2);white-space:nowrap;font-variant-numeric:tabular-nums;">${labeledCount}/${totalCount} (${pct}%)</span>
    </div>`;

    html += _filteredIndices.map(idx => {
        const img = state.images[idx];
        const isActive = idx === state.currentImageIndex ? 'active-image' : '';
        const hasLabel = img.has_label ? 'has-label' : '';
        const isFlagged = state._flaggedImages.has(img.filename);
        const flagIcon = isFlagged ? '<span style="color:var(--success);font-size:9px;margin-left:auto;flex-shrink:0;" title="已标记">&#x2713;</span>' : '';
        return `<div class="img-item ${isActive} ${hasLabel}" data-idx="${idx}" style="display:flex;align-items:center;gap:2px;"><span class="truncate" style="flex:1;min-width:0;">${img.filename}</span>${flagIcon}</div>`;
    }).join('');
    els.imageList.innerHTML = html;

    // Use event delegation instead of per-item handlers
    // (handled in setupEventListeners)
}

function _renderImageListVirtual() {
    const container = els.imageList;
    const scrollTop = container.scrollTop;
    const viewH = container.clientHeight;
    const total = _filteredIndices.length;
    const totalH = total * IMG_ITEM_HEIGHT;

    const startIdx = Math.max(0, Math.floor(scrollTop / IMG_ITEM_HEIGHT) - 5);
    const endIdx = Math.min(total, Math.ceil((scrollTop + viewH) / IMG_ITEM_HEIGHT) + 5);

    let html = `<div style="height:${totalH}px;position:relative;">`;
    for (let i = startIdx; i < endIdx; i++) {
        const idx = _filteredIndices[i];
        const img = state.images[idx];
        const isActive = idx === state.currentImageIndex ? 'active-image' : '';
        const hasLabel = img.has_label ? 'has-label' : '';
        const isFlagged = state._flaggedImages.has(img.filename);
        const flagIcon = isFlagged ? '<span style="color:var(--success);font-size:9px;margin-left:auto;flex-shrink:0;">&#x2713;</span>' : '';
        html += `<div class="img-item ${isActive} ${hasLabel}" data-idx="${idx}" style="position:absolute;top:${i * IMG_ITEM_HEIGHT}px;left:0;right:0;height:${IMG_ITEM_HEIGHT}px;display:flex;align-items:center;gap:2px;"><span class="truncate" style="flex:1;min-width:0;">${img.filename}</span>${flagIcon}</div>`;
    }
    html += '</div>';
    container.innerHTML = html;
    container.scrollTop = scrollTop; // preserve scroll position after re-render
}

/** Lightweight active-state update: just toggle CSS classes without full rebuild */
function updateImageListActive(prevIdx, newIdx) {
    if (_filteredIndices.length >= 500) {
        _renderImageListVirtual();
        return;
    }
    const prev = els.imageList.querySelector(`[data-idx="${prevIdx}"]`);
    const next = els.imageList.querySelector(`[data-idx="${newIdx}"]`);
    if (prev) prev.classList.remove('active-image');
    if (next) next.classList.add('active-image');
}

/** Update single image item has_label class (after save) to avoid full list re-render */
function updateImageListHasLabel(imageIndex, hasLabel) {
    const el = els.imageList.querySelector(`[data-idx="${imageIndex}"]`);
    if (!el) return;
    el.classList.toggle('has-label', !!hasLabel);
}

async function selectImage(index) {
    if (index < 0 || index >= state.images.length) return;
    if (index === state.currentImageIndex) return;
    const loadToken = ++imageSelectToken;

    // Auto-save if dirty (instead of blocking confirm dialog)
    if (state.dirty) {
        const ok = await saveAnnotations();
        if (!ok) return;
    }

    // Reset any in-progress drawing state
    if (state.mode === 'polygon' && state.polygonPoints.length > 0) {
        _cancelPolygonDrawing();
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        if (tempRect) { tempRect.destroy(); tempRect = null; }
        if (tempLabel) { tempLabel.destroy(); tempLabel = null; }
        tempLayer.batchDraw();
    }

    const prevIndex = state.currentImageIndex;
    state.currentImageIndex = index;

    // Lightweight active-state update (avoid full rebuild)
    updateImageListActive(prevIndex, index);
    els.imageCounter.textContent = `${index + 1} / ${state.images.length}`;

    const img = state.images[index];
    const annCacheKey = _annotationCacheKey(img.filename);
    const imgCacheKey = _imageCacheKey(img.filename);
    const imageCached = state.imageCache.has(imgCacheKey);
    const annsCached = state.annotationCache.has(annCacheKey);

    if (!imageCached || !annsCached) {
        _scheduleLoadingOverlay('加载图片中');
    }

    const imagePromise = _loadImageObject(img.filename);
    const annotationPromise = _fetchAnnotationsForImage(img.filename);

    let imgObj;
    try {
        imgObj = await imagePromise;
    } catch (e) {
        console.error('Failed to load image:', e);
        if (loadToken === imageSelectToken) {
            state.currentImageIndex = prevIndex;
            updateImageListActive(index, prevIndex);
            els.imageCounter.textContent = prevIndex >= 0
                ? `${prevIndex + 1} / ${state.images.length}` : `- / ${state.images.length}`;
            showStatus('加载图片失败', true);
            _hideLoadingOverlay();
        }
        return;
    }
    if (loadToken !== imageSelectToken) return;

    state.imageWidth = imgObj.width;
    state.imageHeight = imgObj.height;

    // Reuse or replace image node
    if (currentImageNode) {
        currentImageNode.image(imgObj);
    } else {
        imageLayer.destroyChildren();
        currentImageNode = new Konva.Image({ image: imgObj, x: 0, y: 0 });
        imageLayer.add(currentImageNode);
    }

    // Fit to container
    fitImageToView();

    els.imageSize.textContent = `${imgObj.width}×${imgObj.height}`;

    if (!annsCached) {
        renderAnnotations([]);
    }

    try {
        const anns = await annotationPromise;
        if (loadToken !== imageSelectToken) return;
        state._annotLoadFailed = false;
        renderAnnotations(anns);
    } catch (e) {
        console.error('Failed to load annotations:', e);
        if (loadToken === imageSelectToken) {
            state._annotLoadFailed = true;
            annotLayer.destroyChildren();
            annotLayer.add(new Konva.Text({
                text: '⚠ 标注加载失败\n保存已禁用，请刷新重试',
                x: 20, y: 20, fontSize: 14, fill: '#ef4444',
                fontFamily: 'system-ui', listening: false,
            }));
            annotLayer.batchDraw();
            showStatus('加载标注失败，保存已禁用', true);
        }
    } finally {
        if (loadToken === imageSelectToken) {
            _hideLoadingOverlay();
        }
    }

    // Scroll active image into view
    const activeEl = els.imageList.querySelector('.active-image');
    if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });

    _clearCellposePreview();
    _clearSamPreview();
    state.fewshot.currentResult = null;
    state.fewshot.lastMode = state.fewshot.evalResult ? 'eval' : '';
    prefetchImages(index);
    prefetchAnnotations(index);
    _updateEmptyState();
    updateStatusBar();
    renderFewshotPanel();
    renderThumbnailStrip();

    if (cellposeBatchPhase === 'awaiting_save') {
        _loadBatchPreviewForCurrentImage();
    }
}

/** Build image URL */
function _imageUrl(filename) {
    return `${API}/image_jpeg?group_id=${encodeURIComponent(state.currentGroup.group_id)}&subset=${state.subset}&filename=${encodeURIComponent(filename)}`;
}

function _imageCacheKey(filename) {
    return _imageUrl(filename);
}

function _annotationCacheKey(filename) {
    return _annotationCacheKeyForContext({
        group_id: state.currentGroup?.group_id || '',
        label_set_id: state.currentLabelSet?.set_id || '',
        subset: state.subset,
    }, filename);
}

function _annotationCacheKeyForContext(context, filename) {
    return [
        context?.group_id || '',
        context?.label_set_id || '',
        context?.subset || '',
        filename,
    ].join('::');
}

function _scheduleLoadingOverlay(message) {
    clearTimeout(loadingOverlayTimer);
    loadingOverlayTimer = setTimeout(() => showLoading(true, message), 120);
}

function _hideLoadingOverlay() {
    clearTimeout(loadingOverlayTimer);
    loadingOverlayTimer = null;
    showLoading(false);
}

function _loadImageObject(filename) {
    const cacheKey = _imageCacheKey(filename);
    const cached = state.imageCache.get(cacheKey);
    if (cached) return Promise.resolve(cached);

    const pending = state.imagePendingCache.get(cacheKey);
    if (pending) return pending;

    const img = new Image();
    const promise = new Promise((resolve, reject) => {
        img.onload = () => {
            state.imagePendingCache.delete(cacheKey);
            state.imageCache.set(cacheKey, img);
            resolve(img);
        };
        img.onerror = (err) => {
            state.imagePendingCache.delete(cacheKey);
            reject(err);
        };
    });
    state.imagePendingCache.set(cacheKey, promise);
    img.src = _imageUrl(filename);
    return promise;
}

async function _fetchAnnotationsForImage(filename) {
    const cacheKey = _annotationCacheKey(filename);
    const cached = state.annotationCache.get(cacheKey);
    if (cached) return cached;

    const pending = state.annotationPendingCache.get(cacheKey);
    if (pending) return pending;

    const gid = encodeURIComponent(state.currentGroup.group_id);
    const sid = encodeURIComponent(state.currentLabelSet.set_id);
    const promise = fetch(`${API}/annotations?group_id=${gid}&label_set_id=${sid}&subset=${state.subset}&filename=${encodeURIComponent(filename)}`)
        .then(async (res) => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const anns = await res.json();
            state.annotationCache.set(cacheKey, cloneAnnotations(anns));
            state.annotationPendingCache.delete(cacheKey);
            return anns;
        })
        .catch((err) => {
            state.annotationPendingCache.delete(cacheKey);
            throw err;
        });

    state.annotationPendingCache.set(cacheKey, promise);
    return promise;
}

/** Fit image to canvas container */
function fitImageToView() {
    const cw = els.canvasContainer.clientWidth;
    const ch = els.canvasContainer.clientHeight;
    const scale = Math.min(cw / state.imageWidth, ch / state.imageHeight) * 0.9;
    stage.scale({ x: scale, y: scale });
    stage.position({
        x: (cw - state.imageWidth * scale) / 2,
        y: (ch - state.imageHeight * scale) / 2,
    });
    updateZoomUI(scale);
    updateScaledSizes();
}

function _clampStagePosition() {
    if (!state.imageWidth || !state.imageHeight) return;
    const s = stage.scaleX();
    const cw = els.canvasContainer.clientWidth;
    const ch = els.canvasContainer.clientHeight;
    const imgW = state.imageWidth * s;
    const imgH = state.imageHeight * s;
    const margin = 60;
    let x = stage.x(), y = stage.y();
    x = Math.max(margin - imgW, Math.min(cw - margin, x));
    y = Math.max(margin - imgH, Math.min(ch - margin, y));
    stage.position({ x, y });
}

/** Prefetch ±N images around current index */
function prefetchImages(centerIdx) {
    const range = state.prefetchRange;
    for (let offset = -range; offset <= range; offset++) {
        if (offset === 0) continue;
        const idx = centerIdx + offset;
        if (idx < 0 || idx >= state.images.length) continue;
        const fn = state.images[idx].filename;
        void _loadImageObject(fn).then(() => {
            if (state.imageCache.size > range * 6 + 8) {
                const currentUrls = new Set();
                for (let i = Math.max(0, state.currentImageIndex - range * 2); i <= Math.min(state.images.length - 1, state.currentImageIndex + range * 2); i++) {
                    currentUrls.add(_imageCacheKey(state.images[i].filename));
                }
                for (const key of [...state.imageCache.keys()]) {
                    if (!currentUrls.has(key)) state.imageCache.delete(key);
                    if (state.imageCache.size <= range * 4 + 4) break;
                }
            }
        }).catch(() => {});
    }
}

function prefetchAnnotations(centerIdx) {
    const range = state.annotationPrefetchRange;
    for (let offset = -range; offset <= range; offset++) {
        if (offset === 0) continue;
        const idx = centerIdx + offset;
        if (idx < 0 || idx >= state.images.length) continue;
        const fn = state.images[idx].filename;
        void _fetchAnnotationsForImage(fn).catch(() => {});
    }
}


// ══════════════════════════════════════════════════════════════
// Thumbnail strip
// ══════════════════════════════════════════════════════════════

const _thumbStripState = { rendered: false, range: 15, currentCenter: -1 };

function renderThumbnailStrip() {
    const container = document.getElementById('thumb-container');
    if (!container || state.images.length === 0) return;

    const ci = state.currentImageIndex;
    const range = _thumbStripState.range;
    const startIdx = Math.max(0, ci - range);
    const endIdx = Math.min(state.images.length - 1, ci + range);

    if (_thumbStripState.currentCenter === ci && _thumbStripState.rendered) {
        container.querySelectorAll('.thumb-item').forEach(el => {
            const idx = parseInt(el.dataset.idx);
            el.classList.toggle('thumb-active', idx === ci);
        });
        const activeEl = container.querySelector('.thumb-active');
        if (activeEl) activeEl.scrollIntoView({ inline: 'center', behavior: 'smooth', block: 'nearest' });
        return;
    }
    _thumbStripState.currentCenter = ci;
    _thumbStripState.rendered = true;

    let html = '';
    for (let i = startIdx; i <= endIdx; i++) {
        const img = state.images[i];
        const isActive = i === ci;
        const hasLabel = img.has_label;
        const url = _imageUrl(img.filename);
        html += `<div class="thumb-item ${isActive ? 'thumb-active' : ''} ${hasLabel ? 'thumb-labeled' : ''}" data-idx="${i}" title="${img.filename}">
            <img src="${url}" loading="lazy" style="width:auto; height:42px; border-radius:3px; object-fit:cover; pointer-events:none;" onerror="this.style.display='none'">
            <span class="thumb-idx">${i + 1}</span>
        </div>`;
    }
    container.innerHTML = html;

    container.querySelectorAll('.thumb-item').forEach(el => {
        el.onclick = () => selectImage(parseInt(el.dataset.idx));
    });

    requestAnimationFrame(() => {
        const activeEl = container.querySelector('.thumb-active');
        if (activeEl) activeEl.scrollIntoView({ inline: 'center', behavior: 'instant', block: 'nearest' });
    });
}

// ══════════════════════════════════════════════════════════════
// Annotations rendering
// ══════════════════════════════════════════════════════════════

function renderAnnotations(anns) {
    annotLayer.destroyChildren();
    boxLayer.destroyChildren();
    state.annotationCounter = 0;
    state.undoStack = [];
    state.redoStack = [];
    clearSelection();

    anns.forEach(ann => {
        createAnnotationShape(ann.class_id, ann.ann_type, ann.points);
    });

    renderClassList();
    renderAnnotationList();
    renderVisualization();
    setDirty(false);
    renderFewshotPanel();
}

function _getCurrentSnapshot() {
    const snapshot = [];
    annotLayer.getChildren().forEach(group => {
        if (group.name() !== 'annotation') return;
        snapshot.push({
            classId: group.getAttr('classId'),
            annType: group.getAttr('annType'),
            normPoints: group.getAttr('normPoints').slice(),
            uid: group.getAttr('uid') || '',
        });
    });
    return snapshot;
}

function pushUndoState() {
    state.undoStack.push(_getCurrentSnapshot());
    state.redoStack = [];
    if (state.undoStack.length > state.maxUndo) {
        state.undoStack.shift();
    }
    _updateUndoRedoUI();
}

function _restoreSnapshot(snapshot) {
    annotLayer.destroyChildren();
    boxLayer.destroyChildren();
    state.annotationCounter = 0;
    clearSelection();
    snapshot.forEach(s => {
        createAnnotationShape(s.classId, s.annType, s.normPoints);
    });
    setDirty(true);
    scheduleUIUpdate();
}

function undo() {
    if (state.undoStack.length === 0) {
        showStatus('没有可撤销的操作');
        return;
    }
    state.redoStack.push(_getCurrentSnapshot());
    const snapshot = state.undoStack.pop();
    _restoreSnapshot(snapshot);
    _updateUndoRedoUI();
    showStatus(`已撤销 (${state.undoStack.length}步可撤销)`);
}

function redo() {
    if (state.redoStack.length === 0) {
        showStatus('没有可重做的操作');
        return;
    }
    state.undoStack.push(_getCurrentSnapshot());
    const snapshot = state.redoStack.pop();
    _restoreSnapshot(snapshot);
    _updateUndoRedoUI();
    showStatus(`已重做 (${state.redoStack.length}步可重做)`);
}

function _updateUndoRedoUI() {
    const undoCount = state.undoStack.length;
    const redoCount = state.redoStack.length;
    const el = document.getElementById('footer-undo-redo');
    if (el) {
        el.textContent = `撤销:${undoCount} 重做:${redoCount}`;
        el.style.display = (undoCount > 0 || redoCount > 0) ? '' : 'none';
    }
}

/**
 * Create a Konva annotation group.
 * @param {number} classId
 * @param {string} annType - "bbox" or "polygon"
 * @param {number[]} normPoints - bbox: [cx,cy,w,h]; polygon: [x1,y1,x2,y2,...]
 */
function createAnnotationShape(classId, annType, normPoints) {
    const w = state.imageWidth;
    const h = state.imageHeight;
    const color = state.colors[classId] || '#ffffff';

    let pixelPoints;

    if (annType === 'bbox') {
        // Convert cx,cy,w,h to 4 corner points for rendering
        const cx = normPoints[0] * w;
        const cy = normPoints[1] * h;
        const bw = normPoints[2] * w;
        const bh = normPoints[3] * h;
        const x1 = cx - bw / 2;
        const y1 = cy - bh / 2;
        const x2 = cx + bw / 2;
        const y2 = cy + bh / 2;
        pixelPoints = [x1, y1, x2, y1, x2, y2, x1, y2];
    } else {
        pixelPoints = [];
        for (let i = 0; i < normPoints.length; i += 2) {
            pixelPoints.push(normPoints[i] * w);
            pixelPoints.push(normPoints[i + 1] * h);
        }
    }

    const group = new Konva.Group({
        draggable: false,
        name: 'annotation',
    });
    group.setAttr('classId', classId);
    group.setAttr('annType', annType);
    group.setAttr('uid', `ann-${++state.annotationCounter}`);
    // Store original normalized points for bbox to avoid lossy re-conversion
    group.setAttr('normPoints', normPoints.slice());

    const poly = new Konva.Line({
        points: pixelPoints,
        stroke: color,
        strokeWidth: 2,
        closed: true,
        opacity: 1,
        fillEnabled: false,
        visible: state.showBoundary,
    });
    group.add(poly);
    annotLayer.add(group);

    group.on('click', (e) => {
        if (state.mode === 'select' || state.mode === 'adjust') {
            e.cancelBubble = true;
            if (e.evt.ctrlKey || e.evt.metaKey) {
                toggleAnnotationSelection(group);
            } else {
                selectAnnotation(group);
            }
        }
    });

    group.on('mouseenter', () => {
        if (selectedGroups.has(group)) return;
        poly.strokeWidth(2.5 / (stage?.scaleX() || 1));
        poly.opacity(1);
        annotLayer.batchDraw();
    });
    group.on('mouseleave', () => {
        if (selectedGroups.has(group)) return;
        poly.strokeWidth(1.5 / (stage?.scaleX() || 1));
        annotLayer.batchDraw();
    });

    group.on('dragstart', () => {
        pushUndoState();
    });
    group.on('dragend', () => {
        const poly = group.findOne('Line');
        const ox = group.x();
        const oy = group.y();
        if (ox === 0 && oy === 0) return;
        const pts = poly.points();
        for (let i = 0; i < pts.length; i += 2) {
            pts[i] += ox;
            pts[i + 1] += oy;
        }
        poly.points(pts);
        group.position({ x: 0, y: 0 });
        // Update normPoints
        updateNormPoints(group);
        setDirty(true);
        scheduleUIUpdate();
    });
}

function updateNormPoints(group) {
    const poly = group.findOne('Line');
    const pts = poly.points();
    const w = state.imageWidth;
    const h = state.imageHeight;
    const annType = group.getAttr('annType');

    if (annType === 'bbox') {
        // Derive cx,cy,w,h from corner points
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (let i = 0; i < pts.length; i += 2) {
            minX = Math.min(minX, pts[i]);
            minY = Math.min(minY, pts[i + 1]);
            maxX = Math.max(maxX, pts[i]);
            maxY = Math.max(maxY, pts[i + 1]);
        }
        const cx = (minX + maxX) / 2 / w;
        const cy = (minY + maxY) / 2 / h;
        const bw = (maxX - minX) / w;
        const bh = (maxY - minY) / h;
        group.setAttr('normPoints', [cx, cy, bw, bh]);
    } else {
        const norm = [];
        for (let i = 0; i < pts.length; i += 2) {
            norm.push(pts[i] / w);
            norm.push(pts[i + 1] / h);
        }
        group.setAttr('normPoints', norm);
    }
}


// ══════════════════════════════════════════════════════════════
// Selection
// ══════════════════════════════════════════════════════════════

function selectAnnotation(group) {
    clearSelection();
    selectedGroup = group;
    selectedGroups.add(group);

    const poly = group.findOne('Line');
    poly.opacity(1);
    poly.stroke('white');

    if (state.mode === 'adjust') {
        state.showSelectedOnly = true;
        if (els.vizSelectedOnly) els.vizSelectedOnly.checked = true;
        showAdjustHandles(group);
    }

    els.deleteBtn.classList.remove('hidden');
    els.deleteBtn.textContent = selectedGroups.size > 1 ? `删除选中 (${selectedGroups.size})` : '删除选中 (Del)';
    if (els.batchAddSupportBtn) {
        els.batchAddSupportBtn.classList.remove('hidden');
        els.batchAddSupportBtn.textContent = selectedGroups.size > 1 ? `+ 选中添加为 Support (${selectedGroups.size})` : '+ 选中添加为 Support';
    }
    els.classSelect.value = group.getAttr('classId');
    renderAnnotationList();
    renderVisualization();
    renderFewshotPanel();
}

function toggleAnnotationSelection(group) {
    if (selectedGroups.has(group)) {
        selectedGroups.delete(group);
        if (selectedGroup === group) {
            selectedGroup = selectedGroups.size > 0 ? selectedGroups.values().next().value : null;
            if (selectedGroup && state.mode === 'adjust') {
                showAdjustHandles(selectedGroup);
            } else {
                clearAdjustHandles();
            }
        }
    } else {
        selectedGroups.add(group);
        if (!selectedGroup) selectedGroup = group;
        const poly = group.findOne('Line');
        if (poly) poly.opacity(1).stroke('white');
    }
    _updateSelectionHighlight();
    els.deleteBtn.classList.toggle('hidden', selectedGroups.size === 0);
    els.deleteBtn.textContent = selectedGroups.size > 1 ? `删除选中 (${selectedGroups.size})` : '删除选中 (Del)';
    if (els.batchAddSupportBtn) {
        els.batchAddSupportBtn.classList.toggle('hidden', selectedGroups.size === 0);
        els.batchAddSupportBtn.textContent = selectedGroups.size > 1 ? `+ 选中添加为 Support (${selectedGroups.size})` : '+ 选中添加为 Support';
    }
    if (selectedGroup) els.classSelect.value = selectedGroup.getAttr('classId');
    renderAnnotationList();
    renderVisualization();
    renderFewshotPanel();
}

function _updateSelectionHighlight() {
    annotLayer.getChildren().forEach(g => {
        if (g.name() !== 'annotation') return;
        const poly = g.findOne('Line');
        if (!poly) return;
        const cid = g.getAttr('classId');
        const color = state.colors[cid] || '#fff';
        if (selectedGroups.has(g)) {
            poly.opacity(1).stroke('#ffffff');
            poly.strokeWidth(2.5 / (stage?.scaleX() || 1));
            poly.shadowColor(color);
            poly.shadowBlur(12);
            poly.shadowOpacity(0.6);
            poly.shadowEnabled(true);
        } else {
            poly.stroke(color);
            poly.strokeWidth(1.5 / (stage?.scaleX() || 1));
            poly.shadowEnabled(false);
        }
    });
}

function clearSelection() {
    selectedGroups.forEach(g => {
        const poly = g?.findOne?.('Line');
        if (poly) poly.stroke(state.colors[g.getAttr('classId')] || '#ffffff');
    });
    selectedGroup = null;
    selectedGroups.clear();
    clearAdjustHandles();
    els.deleteBtn.classList.add('hidden');
    if (els.batchAddSupportBtn) els.batchAddSupportBtn.classList.add('hidden');
    renderAnnotationList();
    renderFewshotPanel();
}


// ══════════════════════════════════════════════════════════════
// Mouse interactions
// ══════════════════════════════════════════════════════════════

function onMouseDown(e) {
    // Middle mouse button or Right mouse button -> pan
    if (e.evt.button === 1 || e.evt.button === 2) {
        e.evt.preventDefault();
        stage.draggable(true);
        stage.startDrag();
        return;
    }

    if (e.target.getParent()?.name() === 'annotation' && state.mode === 'select') return;
    if (state.mode === 'select') {
        clearSelection();
        return;
    }

    // Polygon mode: add point on click
    if (state.mode === 'polygon') {
        const pos = clampToImage(stageToImage(stage.getPointerPosition()));
        // Check if clicking near first point to close (snap-close)
        if (state.polygonPoints.length >= 3) {
            const first = state.polygonPoints[0];
            const distPx = Math.hypot(pos.x - first.x, pos.y - first.y) * stage.scaleX();
            if (distPx < 10) {
                _finishPolygonDrawing();
                return;
            }
        }
        _addPolygonPoint(pos);
        return;
    }

    if (state.mode === 'box' || state.mode === 'sam') {
        if (tempRect) { tempRect.destroy(); tempRect = null; }
        if (tempLabel) { tempLabel.destroy(); tempLabel = null; }
        state.isDrawing = true;
        const pos = clampToImage(stageToImage(stage.getPointerPosition()));
        state.startPos = pos;
        const color = getCurrentColor();
        const s = stage.scaleX();

        tempRect = new Konva.Rect({
            x: pos.x, y: pos.y,
            width: 0, height: 0,
            stroke: color,
            strokeWidth: 2 / s,
            fill: hexToRgba(color, 0.12),
        });
        tempLabel = new Konva.Text({
            x: pos.x, y: pos.y,
            text: '0×0',
            fontSize: 12 / s,
            fill: '#e2e8f0',
            listening: false,
        });
        tempLayer.add(tempRect);
        tempLayer.add(tempLabel);
        tempLayer.draw();
    }
}

let _mouseMoveRAF = null;
function onMouseMove(e) {
    // Polygon preview line follows cursor
    if (state.mode === 'polygon' && state.polygonPoints.length > 0) {
        if (!_mouseMoveRAF) {
            _mouseMoveRAF = requestAnimationFrame(() => {
                _mouseMoveRAF = null;
                _updatePolygonPreview(stage.getPointerPosition());
            });
        }
        return;
    }

    if ((state.mode === 'box' || state.mode === 'sam') && state.isDrawing) {
        // Use requestAnimationFrame for smooth drawing
        if (_mouseMoveRAF) return;
        _mouseMoveRAF = requestAnimationFrame(() => {
            _mouseMoveRAF = null;
            if (!state.isDrawing || !tempRect) return;

            let end = clampToImage(stageToImage(stage.getPointerPosition()));
            let dx = end.x - state.startPos.x;
            let dy = end.y - state.startPos.y;

            if (e.evt.shiftKey) {
                const size = Math.max(Math.abs(dx), Math.abs(dy));
                dx = Math.sign(dx || 1) * size;
                dy = Math.sign(dy || 1) * size;
                end = clampToImage({ x: state.startPos.x + dx, y: state.startPos.y + dy });
                dx = end.x - state.startPos.x;
                dy = end.y - state.startPos.y;
            }

            const x1 = Math.min(state.startPos.x, state.startPos.x + dx);
            const y1 = Math.min(state.startPos.y, state.startPos.y + dy);
            const rw = Math.abs(dx);
            const rh = Math.abs(dy);

            tempRect.position({ x: x1, y: y1 });
            tempRect.size({ width: rw, height: rh });
            if (tempLabel) {
                tempLabel.text(`${Math.round(rw)}×${Math.round(rh)}`);
                tempLabel.position({ x: x1 + 4 / stage.scaleX(), y: y1 + 4 / stage.scaleX() });
            }
            tempLayer.batchDraw();
        });
    }
}

async function onMouseUp(e) {
    // End pan (middle or right button)
    if (e.evt.button === 1 || e.evt.button === 2) {
        stage.draggable(false);
        stage.stopDrag();
        _clampStagePosition();
        return;
    }

    if (!state.isDrawing) return;
    state.isDrawing = false;

    if (state.mode === 'box' || state.mode === 'sam') {
        if (!tempRect) return;
        const r = tempRect;
        const x1 = r.x(), y1 = r.y();
        const x2 = x1 + r.width(), y2 = y1 + r.height();

        tempRect.destroy(); tempRect = null;
        if (tempLabel) { tempLabel.destroy(); tempLabel = null; }
        tempLayer.draw();

        if (Math.abs(x2 - x1) < 3 || Math.abs(y2 - y1) < 3) return;

        const normBox = [
            x1 / state.imageWidth, y1 / state.imageHeight,
            x2 / state.imageWidth, y2 / state.imageHeight,
        ];

        if (state.mode === 'sam') {
            pushUndoState();
            await runSAM(normBox);
        } else {
            // Box mode: create bbox annotation (cx, cy, w, h)
            pushUndoState();
            const classId = parseInt(els.classSelect.value);
            const cx = (normBox[0] + normBox[2]) / 2;
            const cy = (normBox[1] + normBox[3]) / 2;
            const bw = normBox[2] - normBox[0];
            const bh = normBox[3] - normBox[1];
            createAnnotationShape(classId, 'bbox', [cx, cy, bw, bh]);
            setDirty(true);
            scheduleUIUpdate();
        }
    }
}

function _getSelectedStrategies() {
    const cbs = document.querySelectorAll('.sam-strat-cb:checked');
    return Array.from(cbs).map(cb => cb.value);
}

function _getSamPromptType() {
    const strats = _getSelectedStrategies();
    if (strats.length === 0) return 'grounding';
    if (strats.length === 1) return strats[0];
    return 'multi';
}

function _getSamTextPrompt() {
    return els.samTextPrompt?.value?.trim() || '';
}

function _updateSamPromptUI() {
    const strats = _getSelectedStrategies();
    const hasText = !!_getSamTextPrompt();
    if (els.samBatchBtn) {
        els.samBatchBtn.disabled = strats.length === 0;
        const label = strats.length > 1
            ? `⚡ ${strats.length}策略`
            : '⚡ BBox预测';
        els.samBatchBtn.textContent = label;
    }
    if (els.samTextRun) {
        els.samTextRun.disabled = !hasText;
    }
    if (els.samStatus && !samPreviewData) {
        if (strats.length === 0 && !hasText) {
            els.samStatus.textContent = '选择策略用BBox预测, 或输入文本做纯文本预测';
        } else if (strats.length > 1) {
            els.samStatus.textContent = `${strats.length} 种策略, 每个BBox取最佳`;
        } else if (strats.length === 1) {
            const hints = {
                'grounding': 'Grounding: 框+类名文本',
                'box_inst':  'SAM2: 纯框',
                '13points':  'SAM2: 13点+框',
            };
            els.samStatus.textContent = hints[strats[0]] || '';
        } else {
            els.samStatus.textContent = hasText ? '输入文本后点击"纯文本预测"' : '';
        }
    }
}

async function runSAM(box) {
    const strats = _getSelectedStrategies();
    const textPrompt = _getSamTextPrompt();
    const classId = parseInt(els.classSelect.value);

    if (strats.length <= 1) {
        const promptType = strats[0] || 'grounding';
        showLoading(true, 'SAM3 预测中');
        try {
            const res = await fetch(`${API}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    group_id: state.currentGroup.group_id,
                    label_set_id: state.currentLabelSet.set_id,
                    subset: state.subset,
                    filename: state.images[state.currentImageIndex].filename,
                    class_id: classId,
                    prompt_type: promptType,
                    box, text_prompt: textPrompt || undefined,
                }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            if (data.multi && Array.isArray(data.points)) {
                for (const poly of data.points) createAnnotationShape(classId, 'polygon', poly);
            } else if (data.points?.length >= 6) {
                createAnnotationShape(classId, 'polygon', data.points);
            } else {
                showStatus('SAM3 未生成有效预测');
                return;
            }
            const stratMsg = data.strategy ? ` [${data.strategy}]` : '';
            showStatus(`SAM3 预测成功${stratMsg}`);
            setDirty(true);
            scheduleUIUpdate();
        } catch (e) {
            console.error(e);
            showStatus(`SAM3 预测失败: ${e.message}`, true);
        } finally {
            showLoading(false);
        }
    } else {
        const imgW = currentImageNode.width();
        const imgH = currentImageNode.height();
        showLoading(true, `SAM3 ${strats.length}策略 预测中`);
        try {
            const res = await fetch(`${API}/predict_batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    group_id: state.currentGroup.group_id,
                    label_set_id: state.currentLabelSet.set_id,
                    subset: state.subset,
                    filename: state.images[state.currentImageIndex].filename,
                    boxes: [{ class_id: classId, box }],
                    prompt_types: strats,
                    text_prompt: textPrompt || undefined,
                }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            const best = data.results?.find(r => r.ok && r.points?.length >= 6);
            if (best) {
                createAnnotationShape(best.class_id, 'polygon', best.points);
                setDirty(true);
                scheduleUIUpdate();
                showStatus(`SAM3 预测成功 [${best.strategy}]`);
            } else {
                showStatus('SAM3 多策略均未生成有效预测', true);
            }
        } catch (e) {
            console.error(e);
            showStatus(`SAM3 预测失败: ${e.message}`, true);
        } finally {
            showLoading(false);
        }
    }
}

async function runSAMTextOnly() {
    if (!state.currentGroup || state.currentImageIndex < 0) {
        showStatus('请先选择一张图片', true);
        return;
    }
    const textPrompt = _getSamTextPrompt();
    if (!textPrompt) {
        showStatus('请输入文本提示', true);
        return;
    }
    const classId = parseInt(els.classSelect.value) || 0;

    _clearSamPreview();
    showLoading(true, 'SAM3 纯文本预测中');
    if (els.samTextRun) els.samTextRun.disabled = true;
    const t0 = performance.now();
    try {
        const res = await fetch(`${API}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                group_id: state.currentGroup.group_id,
                label_set_id: state.currentLabelSet.set_id,
                subset: state.subset,
                filename: state.images[state.currentImageIndex].filename,
                class_id: classId,
                prompt_type: 'text',
                text_prompt: textPrompt,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

        let previewResults = [];
        if (data.multi && Array.isArray(data.points)) {
            previewResults = data.points
                .filter(p => Array.isArray(p) && p.length >= 6)
                .map(p => ({ class_id: classId, points: p, strategy: 'text' }));
        } else if (data.points && data.points.length >= 6) {
            previewResults = [{ class_id: classId, points: data.points, strategy: 'text' }];
        }

        if (previewResults.length > 0) {
            samPreviewData = previewResults;
            _drawSamPreview(previewResults);
            if (els.samPreviewActions) els.samPreviewActions.style.display = 'flex';
            if (els.samStatus) els.samStatus.textContent =
                `文本 "${textPrompt}" → ${previewResults.length} 个区域 (${elapsed}s) — 点击接受/放弃`;
        } else {
            if (els.samStatus) els.samStatus.textContent = `文本 "${textPrompt}" 未检测到目标 (${elapsed}s)`;
        }
    } catch (e) {
        console.error('SAM3 text predict error:', e);
        if (els.samStatus) els.samStatus.textContent = `✗ ${e.message}`;
        showStatus(`SAM3 文本预测失败: ${e.message}`, true);
    } finally {
        showLoading(false);
        if (els.samTextRun) els.samTextRun.disabled = !_getSamTextPrompt();
    }
}

// ── SAM3 batch predict from existing bboxes ──

async function runSAMBatch() {
    if (!state.currentGroup || state.currentImageIndex < 0) {
        showStatus('请先选择一张图片', true);
        return;
    }

    const bboxAnns = [];
    const bboxGroups = [];
    annotLayer.getChildren().forEach(g => {
        if (g.name() !== 'annotation') return;
        if (g.getAttr('annType') !== 'bbox') return;
        const pts = g.findOne('Line')?.points();
        if (!pts || pts.length < 4) return;
        const imgW = currentImageNode.width();
        const imgH = currentImageNode.height();
        let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
        for (let i = 0; i < pts.length; i += 2) {
            xmin = Math.min(xmin, pts[i]);   ymin = Math.min(ymin, pts[i+1]);
            xmax = Math.max(xmax, pts[i]);   ymax = Math.max(ymax, pts[i+1]);
        }
        bboxAnns.push({
            class_id: g.getAttr('classId'),
            box: [xmin / imgW, ymin / imgH, xmax / imgW, ymax / imgH],
        });
        bboxGroups.push(g);
    });

    if (bboxAnns.length === 0) {
        showStatus('当前图片没有 BBox 标注可用于预测', true);
        return;
    }

    const strategies = _getSelectedStrategies();
    const textPrompt = _getSamTextPrompt();

    if (strategies.length === 0) {
        showStatus('请至少选择一种预测策略', true);
        return;
    }

    const stratLabel = strategies.length > 1
        ? `${strategies.length}策略`
        : strategies[0];
    const abortSignal = startLongRun();
    showLoading(true, `SAM3 ${stratLabel} 预测 ${bboxAnns.length} 个 BBox`, { stoppable: true });
    if (els.samStatus) els.samStatus.textContent = '正在运行 SAM3…';
    if (els.samBatchBtn) els.samBatchBtn.disabled = true;

    const t0 = Date.now();
    try {
        const body = {
            group_id: state.currentGroup.group_id,
            label_set_id: state.currentLabelSet.set_id,
            subset: state.subset,
            filename: state.images[state.currentImageIndex].filename,
            boxes: bboxAnns,
            prompt_types: strategies,
        };
        if (textPrompt) body.text_prompt = textPrompt;

        const res = await fetch(`${API}/predict_batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: abortSignal,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

        const validResults = data.results
            .map((r, idx) => ({
                ...r,
                _sourceUid: bboxGroups[idx]?.getAttr('uid') || null,
            }))
            .filter(r => r.ok && r.points && r.points.length > 0);
        const failedResults = data.results.filter(r => !r.ok);
        const errMsgs = failedResults.flatMap(r => r.errors || []).filter(Boolean);

        if (validResults.length > 0) {
            samPreviewData = validResults;
            samPreviewSourceBboxGroups = bboxGroups.filter(
                group => validResults.some(result => result._sourceUid && result._sourceUid === group.getAttr('uid'))
            );
            _drawSamPreview(validResults);
            if (els.samPreviewActions) els.samPreviewActions.style.display = 'flex';
            const stratInfo = data.strategies_used ? ` [${data.strategies_used.join('+')}]` : '';
            if (els.samStatus) els.samStatus.textContent =
                `✓ ${validResults.length}/${data.total} 成功 (${elapsed}s)${stratInfo} — 请确认`;
        } else {
            let msg = `未生成有效预测 (${elapsed}s)`;
            if (errMsgs.length > 0) msg += ` | ${errMsgs[0]}`;
            if (els.samStatus) els.samStatus.textContent = msg;
        }
        if (errMsgs.length > 0) {
            console.warn('SAM3 prediction errors:', errMsgs);
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            if (els.samStatus) els.samStatus.textContent = 'SAM3 预测已停止';
            showStatus('已停止 SAM3 批量预测');
        } else {
            console.error('SAM3 batch error:', e);
            if (els.samStatus) els.samStatus.textContent = `✗ ${e.message}`;
            showStatus(`SAM3 批量预测失败: ${e.message}`, true);
        }
    } finally {
        _longRunAbort = null;
        showLoading(false);
        if (els.samBatchBtn) els.samBatchBtn.disabled = false;
    }
}

function _drawSamPreview(results) {
    samPreviewLayer.destroyChildren();
    if (!currentImageNode) return;
    const imgW = currentImageNode.width();
    const imgH = currentImageNode.height();
    const scale = stage?.scaleX() || 1;

    const stratColors = { grounding: null, '13points': '#f59e0b', box_inst: '#8b5cf6', text: '#ec4899' };

    for (const r of results) {
        if (!r.points || r.points.length < 6) continue;
        const baseColor = state.colors[r.class_id] || '#7aa2ff';
        const color = (r.strategy && stratColors[r.strategy]) || baseColor;
        const pts = [];
        for (let i = 0; i < r.points.length; i += 2) {
            pts.push(r.points[i] * imgW);
            pts.push(r.points[i + 1] * imgH);
        }
        samPreviewLayer.add(new Konva.Line({
            points: pts,
            closed: true,
            stroke: color,
            strokeWidth: 2 / scale,
            fill: _hexToRgba(color, 0.18),
            dash: [6 / scale, 3 / scale],
            listening: false,
        }));
    }
    samPreviewLayer.batchDraw();
}

function acceptSAMPreview() {
    if (!samPreviewData || samPreviewData.length === 0) return;
    pushUndoState();
    for (const r of samPreviewData) {
        if (r.points && r.points.length >= 6) {
            createAnnotationShape(r.class_id, 'polygon', r.points);
        }
    }
    if (samPreviewSourceBboxGroups.length > 0) {
        for (const g of samPreviewSourceBboxGroups) {
            if (g && g.getParent()) g.destroy();
        }
        annotLayer.batchDraw();
    }
    setDirty(true);
    scheduleUIUpdate();
    const removedCount = samPreviewSourceBboxGroups.length;
    const msg = removedCount > 0
        ? `✓ 已接受 ${samPreviewData.length} 个 SAM3 预测，已移除 ${removedCount} 个源 BBox`
        : `✓ 已接受 ${samPreviewData.length} 个 SAM3 预测结果`;
    showStatus(msg);
    _clearSamPreview();
}

function rejectSAMPreview() {
    _clearSamPreview();
    showStatus('已放弃 SAM3 预测结果');
}

function _clearSamPreview() {
    samPreviewData = null;
    samPreviewSourceBboxGroups = [];
    samPreviewLayer.destroyChildren();
    samPreviewLayer.batchDraw();
    if (els.samPreviewActions) els.samPreviewActions.style.display = 'none';
    if (els.samStatus) els.samStatus.textContent = '';
    _updateSamPromptUI();
}


// ══════════════════════════════════════════════════════════════
// Adjust mode — corner handles + confirm/cancel
// ══════════════════════════════════════════════════════════════

/**
 * Enter adjust editing for an annotation.
 * - Saves original points as backup (for cancel)
 * - Shows rect + 4 draggable corner handles
 * - Shows confirm/cancel buttons
 * - Real-time preview during drag
 * - Confirm applies, Cancel reverts
 */
function showAdjustHandles(group) {
    clearAdjustHandles();

    const poly = group.findOne('Line');
    if (!poly) return;
    const pts = poly.points();
    if (pts.length < 4) return;

    // Save original state for cancel
    adjustOrigPts = pts.slice();
    const classId = group.getAttr('classId');
    const color = state.colors[classId] || '#ffffff';
    const s = stage.scaleX();

    // Compute original bbox
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (let i = 0; i < pts.length; i += 2) {
        x1 = Math.min(x1, pts[i]);
        y1 = Math.min(y1, pts[i + 1]);
        x2 = Math.max(x2, pts[i]);
        y2 = Math.max(y2, pts[i + 1]);
    }
    adjustOrigBBox = { x1, y1, x2, y2 };

    // Editable rect (draggable body = move)
    adjustRect = new Konva.Rect({
        x: x1, y: y1,
        width: x2 - x1, height: y2 - y1,
        stroke: color,
        strokeWidth: 1.5 / s,
        dash: [6 / s, 3 / s],
        fill: hexToRgba(color, 0.06),
        draggable: true,
        name: 'adjust-rect',
    });
    tempLayer.add(adjustRect);

    adjustRect.on('dragmove', () => {
        // Clamp rect within image
        const rx = clamp(adjustRect.x(), 0, state.imageWidth - adjustRect.width());
        const ry = clamp(adjustRect.y(), 0, state.imageHeight - adjustRect.height());
        adjustRect.position({ x: rx, y: ry });
        syncHandlesToRect();
        livePreviewAdjust();
    });

    // 4 corner handles: [TL, TR, BR, BL]
    const corners = [
        { x: x1, y: y1 },
        { x: x2, y: y1 },
        { x: x2, y: y2 },
        { x: x1, y: y2 },
    ];

    adjustHandles = corners.map((c, idx) => {
        const handle = new Konva.Rect({
            x: c.x - 4 / s, y: c.y - 4 / s,
            width: 8 / s, height: 8 / s,
            fill: 'white',
            stroke: color,
            strokeWidth: 1.5 / s,
            draggable: true,
            name: 'adjust-handle',
            cornerRadius: 1 / s,
        });
        handle.setAttr('cornerIndex', idx);
        tempLayer.add(handle);

        handle.on('dragmove', () => {
            // Get center of the handle as the corner position
            const hx = handle.x() + handle.width() / 2;
            const hy = handle.y() + handle.height() / 2;
            const pos = clampToImage({ x: hx, y: hy });
            onCornerDrag(idx, pos.x, pos.y);
            livePreviewAdjust();
        });

        const cursors = ['nwse-resize', 'nesw-resize', 'nwse-resize', 'nesw-resize'];
        handle.on('mouseenter', () => { document.body.style.cursor = cursors[idx]; });
        handle.on('mouseleave', () => { document.body.style.cursor = ''; });

        return handle;
    });

    // Show confirm/cancel buttons
    els.adjustActions.classList.remove('hidden');
    tempLayer.draw();
}

/**
 * Handle corner drag — the key fix for shrinking.
 * Each corner controls two edges. The opposite corner stays fixed.
 *
 * TL(0): controls x1, y1  — opposite: BR(2)
 * TR(1): controls x2, y1  — opposite: BL(3)
 * BR(2): controls x2, y2  — opposite: TL(0)
 * BL(3): controls x1, y2  — opposite: TR(1)
 */
function onCornerDrag(idx, cx, cy) {
    if (!adjustRect || adjustHandles.length !== 4) return;

    // Current rect bounds from the rect itself
    let x1 = adjustRect.x();
    let y1 = adjustRect.y();
    let x2 = x1 + adjustRect.width();
    let y2 = y1 + adjustRect.height();

    // Update the two edges this corner controls
    switch (idx) {
        case 0: x1 = cx; y1 = cy; break; // TL
        case 1: x2 = cx; y1 = cy; break; // TR
        case 2: x2 = cx; y2 = cy; break; // BR
        case 3: x1 = cx; y2 = cy; break; // BL
    }

    // Ensure min size (prevent flipping)
    const minSize = 4;
    if (x2 - x1 < minSize) { if (idx === 0 || idx === 3) x1 = x2 - minSize; else x2 = x1 + minSize; }
    if (y2 - y1 < minSize) { if (idx === 0 || idx === 1) y1 = y2 - minSize; else y2 = y1 + minSize; }

    // Clamp to image
    x1 = clamp(x1, 0, state.imageWidth);
    y1 = clamp(y1, 0, state.imageHeight);
    x2 = clamp(x2, 0, state.imageWidth);
    y2 = clamp(y2, 0, state.imageHeight);

    // Update rect
    adjustRect.position({ x: x1, y: y1 });
    adjustRect.size({ width: x2 - x1, height: y2 - y1 });

    // Update ALL handle positions
    syncHandlesToRect();
}

/** Sync all 4 corner handles to match the current adjustRect. */
function syncHandlesToRect() {
    if (!adjustRect || adjustHandles.length !== 4) return;
    const s = stage.scaleX();
    const hs = 4 / s; // half handle size
    const x1 = adjustRect.x();
    const y1 = adjustRect.y();
    const x2 = x1 + adjustRect.width();
    const y2 = y1 + adjustRect.height();

    const positions = [
        { x: x1, y: y1 },
        { x: x2, y: y1 },
        { x: x2, y: y2 },
        { x: x1, y: y2 },
    ];
    adjustHandles.forEach((h, i) => {
        h.position({ x: positions[i].x - hs, y: positions[i].y - hs });
    });
    tempLayer.batchDraw();
}

/** Real-time preview: update the polygon shape to match current adjust rect. */
function livePreviewAdjust() {
    if (!selectedGroup || !adjustRect || !adjustOrigPts || !adjustOrigBBox) return;
    const poly = selectedGroup.findOne('Line');
    if (!poly) return;

    const nx1 = adjustRect.x();
    const ny1 = adjustRect.y();
    const nx2 = nx1 + adjustRect.width();
    const ny2 = ny1 + adjustRect.height();

    const annType = selectedGroup.getAttr('annType');
    const { x1: oX1, y1: oY1, x2: oX2, y2: oY2 } = adjustOrigBBox;

    if (annType === 'bbox') {
        poly.points([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2]);
    } else {
        // Scale original polygon proportionally
        const oW = Math.max(1, oX2 - oX1);
        const oH = Math.max(1, oY2 - oY1);
        const nW = Math.max(1, nx2 - nx1);
        const nH = Math.max(1, ny2 - ny1);

        const newPts = [];
        for (let i = 0; i < adjustOrigPts.length; i += 2) {
            newPts.push(nx1 + (adjustOrigPts[i] - oX1) / oW * nW);
            newPts.push(ny1 + (adjustOrigPts[i + 1] - oY1) / oH * nH);
        }
        poly.points(newPts);
    }
    poly.visible(true);
    annotLayer.batchDraw();
}

/** Confirm: apply adjust and close handles. */
function confirmAdjust() {
    if (!selectedGroup || !adjustRect) return;
    pushUndoState();
    // The polygon is already updated by livePreview, just finalize
    updateNormPoints(selectedGroup);
    setDirty(true);
    clearAdjustHandles();
    clearSelection();
    renderAnnotationList();
    renderVisualization();
    showStatus('调整已确认');
}

/** Cancel: revert to original points. */
function cancelAdjust() {
    if (!selectedGroup || !adjustOrigPts) {
        clearAdjustHandles();
        return;
    }
    const poly = selectedGroup.findOne('Line');
    if (poly) {
        poly.points(adjustOrigPts.slice());
    }
    clearAdjustHandles();
    clearSelection();
    renderAnnotationList();
    renderVisualization();
    showStatus('调整已取消');
}

function clearAdjustHandles() {
    if (adjustRect) { adjustRect.destroy(); adjustRect = null; }
    adjustHandles.forEach(h => h.destroy());
    adjustHandles = [];
    adjustOrigPts = null;
    adjustOrigBBox = null;
    els.adjustActions.classList.add('hidden');
    tempLayer.draw();
    document.body.style.cursor = '';
}


// ══════════════════════════════════════════════════════════════
// Save
// ══════════════════════════════════════════════════════════════

async function saveAnnotations() {
    if (!state.currentGroup || !state.currentLabelSet || state.currentImageIndex < 0) return false;
    if (state._annotLoadFailed) {
        showStatus('标注加载失败，无法保存（防止数据丢失）', true);
        return false;
    }

    const anns = [];
    annotLayer.getChildren().forEach(group => {
        if (group.name() !== 'annotation') return;
        anns.push({
            class_id: group.getAttr('classId'),
            ann_type: group.getAttr('annType'),
            points: group.getAttr('normPoints'),
            annotation_uid: group.getAttr('uid'),
        });
    });
    const filename = state.images[state.currentImageIndex]?.filename;
    if (!filename) return false;
    const context = {
        group_id: state.currentGroup.group_id,
        label_set_id: state.currentLabelSet.set_id,
        subset: state.subset,
    };
    const cacheKey = _annotationCacheKeyForContext(context, filename);
    const requestId = ++saveRequestCounter;

    try {
        const res = await fetch(`${API}/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                group_id: context.group_id,
                label_set_id: context.label_set_id,
                subset: context.subset,
                filename,
                annotations: anns,
            }),
        });
        if (!res.ok) throw new Error('Save failed');
        state.annotationCache.set(cacheKey, cloneAnnotations(anns));
        const stillCurrent = (
            state.currentGroup?.group_id === context.group_id &&
            state.currentLabelSet?.set_id === context.label_set_id &&
            state.subset === context.subset &&
            state.images[state.currentImageIndex]?.filename === filename
        );
        if (stillCurrent && requestId === saveRequestCounter) {
            setDirty(false);
            state.images[state.currentImageIndex].has_label = anns.length > 0;
            updateImageListHasLabel(state.currentImageIndex, anns.length > 0);
        } else {
            const imageIndex = state.images.findIndex(img => img.filename === filename);
            if (imageIndex >= 0) {
                state.images[imageIndex].has_label = anns.length > 0;
                updateImageListHasLabel(imageIndex, anns.length > 0);
            }
        }
        showStatus('已保存');
        return true;
    } catch (e) {
        console.error(e);
        showStatus('保存失败!', true);
        return false;
    }
}

/** Save current image then switch to next (Save & Next). */
async function saveNextAndGo() {
    if (!state.currentGroup || !state.currentLabelSet || state.currentImageIndex < 0) return;
    const ok = await saveAnnotations();
    if (!ok) return;
    if (state.currentImageIndex + 1 < state.images.length) {
        await selectImage(state.currentImageIndex + 1);
    }
}

// Auto-save: every 45s if dirty
const AUTO_SAVE_INTERVAL_MS = 45000;
let autoSaveTimerId = null;

function startAutoSaveTimer() {
    if (autoSaveTimerId) return;
    autoSaveTimerId = setInterval(async () => {
        if (!state.dirty || state.currentImageIndex < 0) return;
        await saveAnnotations();
        showStatus('自动保存');
        if (els.footerAutosave) {
            els.footerAutosave.classList.add('flash');
            setTimeout(() => els.footerAutosave.classList.remove('flash'), 800);
        }
    }, AUTO_SAVE_INTERVAL_MS);
}


// ══════════════════════════════════════════════════════════════
// Visualization
// ══════════════════════════════════════════════════════════════

function renderVisualization() {
    const mode = state.vizModeValue || 'both';
    annotLayer.getChildren().forEach(group => {
        if (group.name() !== 'annotation') return;
        const poly = group.findOne('Line');
        if (!poly) return;

        if (state.mode === 'adjust') {
            poly.visible(group === selectedGroup);
            return;
        }
        if (state.showSelectedOnly && selectedGroup && group !== selectedGroup) {
            poly.visible(false);
            return;
        }
        const isBbox = group.getAttr('annType') === 'bbox';
        if (mode === 'none') { poly.visible(false); return; }
        if (isBbox) { poly.visible(state.showBoxes || mode === 'filled' || mode === 'mask'); return; }
        poly.visible(state.showBoundary || mode === 'filled' || mode === 'mask');

        // Apply fill/stroke styling based on mode
        const classId = group.getAttr('classId');
        const color = state.colors[classId] || '#7aa2ff';
        if (mode === 'mask') {
            poly.fill(_hexToRgba(color, 0.35));
            poly.strokeWidth(0);
        } else if (mode === 'filled') {
            poly.fill(_hexToRgba(color, 0.18));
            poly.strokeWidth(2);
        } else {
            poly.fill(_hexToRgba(color, 0.06));
            poly.strokeWidth(2);
        }
    });

    renderBoxOverlays();
    renderAnnotationLabels();
    annotLayer.batchDraw();
}

function renderAnnotationLabels() {
    if (!labelLayer || !annotLayer) return;
    labelLayer.destroyChildren();
    const s = stage?.scaleX() || 1;
    const items = annotLayer.getChildren().filter(g => g.name() === 'annotation');

    items.forEach((group, idx) => {
        const poly = group.findOne('Line');
        if (!poly) return;

        let visible = true;
        if (state.mode === 'adjust') {
            visible = group === selectedGroup;
        } else if (state.showSelectedOnly && selectedGroup && group !== selectedGroup) {
            visible = false;
        } else {
            const isBbox = group.getAttr('annType') === 'bbox';
            visible = isBbox ? state.showBoxes : state.showBoundary;
        }
        if (!visible) return;

        const pts = poly.points();
        if (!pts || pts.length < 4) return;
        let minX = Infinity, minY = Infinity;
        for (let i = 0; i < pts.length; i += 2) {
            minX = Math.min(minX, pts[i]);
            minY = Math.min(minY, pts[i + 1]);
        }

        const classId = group.getAttr('classId');
        const color = state.colors[classId] || '#7aa2ff';
        const className = state.classes.find(c => c.ID === classId)?.Name || '';
        const isSelected = selectedGroups.has(group) || group === selectedGroup;
        const labelText = className ? `${idx + 1} ${className}` : String(idx + 1);

        const label = new Konva.Label({
            x: minX,
            y: Math.max(2, minY - (isSelected ? 16 : 12)),
            listening: false,
        });
        label.add(new Konva.Tag({
            fill: color,
            opacity: isSelected ? 1.0 : 0.85,
            cornerRadius: 2,
            stroke: isSelected ? '#fff' : undefined,
            strokeWidth: isSelected ? 1 / s : 0,
        }));
        label.add(new Konva.Text({
            text: labelText,
            fontSize: Math.max(8, (isSelected ? 11 : 10) / s),
            fontFamily: 'system-ui, sans-serif',
            fontStyle: '600',
            padding: 2,
            fill: '#fff',
        }));
        labelLayer.add(label);
    });
    labelLayer.batchDraw();
}

function renderBoxOverlays() {
    boxLayer.destroyChildren();
    if (!state.showBoxes || state.mode === 'adjust') {
        boxLayer.batchDraw();
        return;
    }

    const s = stage.scaleX();
    annotLayer.getChildren().forEach(group => {
        if (group.name() !== 'annotation') return;
        if (state.showSelectedOnly && selectedGroup && group !== selectedGroup) return;
        if (group.getAttr('annType') === 'bbox') return;

        const poly = group.findOne('Line');
        if (!poly) return;
        const pts = poly.points();
        if (pts.length < 4) return;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (let i = 0; i < pts.length; i += 2) {
            minX = Math.min(minX, pts[i]);
            minY = Math.min(minY, pts[i + 1]);
            maxX = Math.max(maxX, pts[i]);
            maxY = Math.max(maxY, pts[i + 1]);
        }

        const color = state.colors[group.getAttr('classId')] || '#ffffff';
        const rect = new Konva.Rect({
            x: minX, y: minY,
            width: Math.max(1, maxX - minX),
            height: Math.max(1, maxY - minY),
            stroke: color,
            strokeWidth: 1.2 / s,
            dash: [5 / s, 3 / s],
            fillEnabled: false,
            listening: false,
        });
        boxLayer.add(rect);
    });
    boxLayer.batchDraw();
}

function setVisualizationMode(mode) {
    state.showBoxes = mode === 'both' || mode === 'bbox';
    state.showBoundary = mode === 'both' || mode === 'boundary' || mode === 'filled';
    state.showFill = mode === 'filled' || mode === 'mask';
    state.showStroke = mode !== 'mask' && mode !== 'none';
    state.vizModeValue = mode;
    renderVisualization();
}


// ══════════════════════════════════════════════════════════════
// UI Panels
// ══════════════════════════════════════════════════════════════

let uiUpdateTimer = null;
function scheduleUIUpdate() {
    if (uiUpdateTimer) return;
    uiUpdateTimer = requestAnimationFrame(() => {
        uiUpdateTimer = null;
        renderClassList();
        renderAnnotationList();
        renderVisualization();
        renderFewshotPanel();
    });
}

function renderClassList() {
    if (!els.classList) return;
    const counts = {};
    annotLayer.getChildren().forEach(g => {
        if (g.name() !== 'annotation') return;
        const id = g.getAttr('classId');
        counts[id] = (counts[id] || 0) + 1;
    });

    if (!state.classes.length) {
        els.classList.innerHTML = `<div class="empty-state" style="min-height:72px;">
            <strong>暂无类别</strong>
            <span>当前图片集还没有可用类别，请先检查数据集配置。</span>
        </div>`;
        return;
    }

    els.classList.innerHTML = state.classes.map(c => {
        const color = state.colors[c.ID] || '#fff';
        const active = parseInt(els.classSelect.value) === parseInt(c.ID);
        const count = counts[c.ID] || 0;
        return `<button class="flex items-center justify-between gap-1 px-2 py-1 rounded text-xs"
                    style="border:1px solid ${active ? 'var(--accent)' : 'var(--border)'}; background:${active ? '#1b2134' : 'var(--bg-3)'}"
                    data-id="${c.ID}">
                    <span class="flex items-center gap-1">
                        <span class="w-2 h-2 rounded-full" style="background:${color}"></span>
                        ${c.Name}
                    </span>
                    <span style="color:var(--text-2)">${count}</span>
                </button>`;
    }).join('');

    els.classList.querySelectorAll('button').forEach(btn => {
        btn.onclick = () => {
            els.classSelect.value = btn.dataset.id;
            if (selectedGroup) {
                pushUndoState();
                selectedGroup.setAttr('classId', parseInt(btn.dataset.id));
                const poly = selectedGroup.findOne('Line');
                poly.stroke('white');
                setDirty(true);
            }
            renderClassList();
            renderAnnotationList();
            renderVisualization();
            renderFewshotPanel();
        };
    });
}

function renderAnnotationList() {
    if (!els.annotationList) return;
    const items = annotLayer.getChildren().filter(g => g.name() === 'annotation');
    els.annCount.textContent = items.length;

    if (items.length === 0) {
        els.annotationList.innerHTML = `<div class="empty-state">
            <strong>当前图片还没有标注</strong>
            <span>可直接运行 CellposeSAM，或用画框 / SAM3 开始标注。</span>
        </div>`;
        return;
    }

    // Build class distribution summary
    const classCounts = {};
    items.forEach(g => {
        const cid = g.getAttr('classId');
        classCounts[cid] = (classCounts[cid] || 0) + 1;
    });
    const summaryChips = Object.entries(classCounts).map(([cid, cnt]) => {
        const cls = state.classes.find(c => parseInt(c.ID) === parseInt(cid));
        const name = cls ? cls.Name : `#${cid}`;
        const color = state.colors[cid] || '#888';
        const isActive = state._annFilterClass == null || String(state._annFilterClass) === String(cid);
        return `<span class="ann-class-chip" data-class-id="${cid}" style="display:inline-flex;align-items:center;gap:3px;padding:1px 6px;
            background:${isActive ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.01)'};border-radius:999px;font-size:9px;
            color:${isActive ? 'var(--text-1)' : 'var(--text-3)'};cursor:pointer;border:1px solid ${isActive ? color + '40' : 'transparent'};
            transition:all .15s;opacity:${isActive ? 1 : 0.5};">
            <span style="width:6px;height:6px;border-radius:50%;background:${color};flex-shrink:0;"></span>
            ${name} <strong style="color:${isActive ? 'var(--text-0)' : 'var(--text-3)'};">${cnt}</strong></span>`;
    }).join('');

    const filterVal = state._annFilterText || '';
    let listHtml = `<div style="padding:3px 6px 0;"><input id="ann-filter-input" type="text" placeholder="搜索标注 (类别名/编号)..."
        value="${filterVal}" style="width:100%;background:var(--bg-3);border:1px solid var(--border);color:var(--text-0);
        border-radius:4px;padding:3px 6px;font-size:10px;outline:none;" /></div>
    <div style="display:flex;flex-wrap:wrap;gap:3px;padding:3px 6px 4px;border-bottom:1px solid rgba(255,255,255,0.04);">${summaryChips}</div>`;

    // Apply filters
    let filteredItems = items;
    if (state._annFilterClass != null) {
        filteredItems = filteredItems.filter(g => String(g.getAttr('classId')) === String(state._annFilterClass));
    }
    if (state._annFilterText) {
        const q = state._annFilterText.toLowerCase();
        filteredItems = filteredItems.filter(g => {
            const cls = state.classes.find(c => parseInt(c.ID) === parseInt(g.getAttr('classId')));
            const name = cls ? cls.Name.toLowerCase() : '';
            return name.includes(q) || String(items.indexOf(g) + 1).includes(q);
        });
    }

    listHtml += filteredItems.map((g, _fi) => {
        const idx = items.indexOf(g);
        const cls = state.classes.find(c => parseInt(c.ID) === parseInt(g.getAttr('classId')));
        const name = cls ? cls.Name : `Class ${g.getAttr('classId')}`;
        const active = selectedGroups.has(g);
        const annType = g.getAttr('annType');
        const typeTag = annType === 'bbox' ? '□' : '◇';
        const color = state.colors[g.getAttr('classId')] || '#fff';
        const pts = g.getAttr('normPoints') || [];
        const ptCount = annType === 'bbox' ? '' : `${pts.length / 2}pt`;
        return `<div class="ann-item ${active ? 'active' : ''}" data-uid="${g.getAttr('uid')}" data-class-id="${g.getAttr('classId')}">
                    <input type="checkbox" class="ann-item-checkbox" ${active ? 'checked' : ''} data-uid="${g.getAttr('uid')}" title="多选">
                    <span class="ann-item-label">
                        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0;margin-right:3px;vertical-align:middle;box-shadow:0 0 4px ${color}40;"></span>
                        <span style="color:${color};font-size:11px;">${typeTag}</span> ${idx + 1}. ${name}
                    </span>
                    <span class="ann-item-right">
                        ${ptCount ? `<span class="ann-type-tag">${ptCount}</span>` : ''}
                        <span class="ann-type-tag">${annType}</span>
                        <span class="ann-del-btn" data-uid="${g.getAttr('uid')}" title="删除">×</span>
                    </span>
                </div>`;
    }).join('');

    // Batch class change control (visible when items selected)
    if (selectedGroups.size > 0) {
        const classOptions = state.classes.map(c =>
            `<option value="${c.ID}">${c.Name}</option>`
        ).join('');
        listHtml += `<div style="padding:4px 6px;border-top:1px solid rgba(255,255,255,0.04);
            display:flex;align-items:center;gap:4px;font-size:10px;color:var(--text-2);">
            <span>批量改类:</span>
            <select id="ann-batch-class" style="flex:1;background:var(--bg-3);border:1px solid var(--border);
                color:var(--text-0);border-radius:4px;padding:2px 4px;font-size:10px;">
                ${classOptions}
            </select>
            <button class="btn btn-primary" style="font-size:9px;padding:2px 6px;" onclick="_batchChangeClass()">应用</button>
        </div>`;
    }

    els.annotationList.innerHTML = listHtml;

    els.annotationList.querySelectorAll('.ann-item-checkbox').forEach(cb => {
        cb.onclick = (e) => {
            e.stopPropagation();
            const uid = cb.dataset.uid;
            const target = annotLayer.getChildren().find(g => g.name() === 'annotation' && g.getAttr('uid') === uid);
            if (!target) return;
            toggleAnnotationSelection(target);
        };
    });

    els.annotationList.querySelectorAll('.ann-item').forEach(el => {
        el.onclick = (e) => {
            if (e.target.classList.contains('ann-del-btn') || e.target.classList.contains('ann-item-checkbox')) return;
            const uid = el.dataset.uid;
            const target = annotLayer.getChildren().find(g => g.name() === 'annotation' && g.getAttr('uid') === uid);
            if (!target) return;
            if (e.ctrlKey || e.metaKey) {
                toggleAnnotationSelection(target);
            } else if (selectedGroups.has(target) && selectedGroups.size === 1) {
                clearSelection();
                renderAnnotationList();
                renderVisualization();
            } else {
                selectAnnotation(target);
            }
        };
    });

    els.annotationList.querySelectorAll('.ann-del-btn').forEach(btn => {
        btn.onclick = (e) => {
            e.stopPropagation();
            const uid = btn.dataset.uid;
            const target = annotLayer.getChildren().find(g => g.name() === 'annotation' && g.getAttr('uid') === uid);
            if (!target) return;
            pushUndoState();
            selectedGroups.delete(target);
            if (selectedGroup === target) selectedGroup = selectedGroups.size > 0 ? selectedGroups.values().next().value : null;
            if (selectedGroups.size === 0) {
                els.deleteBtn.classList.add('hidden');
                if (els.batchAddSupportBtn) els.batchAddSupportBtn.classList.add('hidden');
            }
            target.destroy();
            setDirty(true);
            scheduleUIUpdate();
        };
    });

    // Bind search/filter events
    const filterInput = document.getElementById('ann-filter-input');
    if (filterInput) {
        filterInput.oninput = () => {
            state._annFilterText = filterInput.value;
            renderAnnotationList();
            filterInput.focus();
        };
    }
    els.annotationList.querySelectorAll('.ann-class-chip').forEach(chip => {
        chip.onclick = () => {
            const cid = chip.dataset.classId;
            state._annFilterClass = (String(state._annFilterClass) === String(cid)) ? null : cid;
            renderAnnotationList();
        };
    });
}

function _batchChangeClass() {
    const sel = document.getElementById('ann-batch-class');
    if (!sel || selectedGroups.size === 0) return;
    const newClassId = parseInt(sel.value);
    pushUndoState();
    selectedGroups.forEach(g => {
        g.setAttr('classId', newClassId);
        const color = state.colors[newClassId] || '#fff';
        const poly = g.findOne('Line');
        if (poly) {
            poly.stroke(color);
            poly.fill(hexToRgba(color, 0.18));
        }
    });
    setDirty(true);
    scheduleUIUpdate();
    showStatus(`已将 ${selectedGroups.size} 个标注修改为类别 ${newClassId}`);
}


// ══════════════════════════════════════════════════════════════
// Mode management
// ══════════════════════════════════════════════════════════════

function setMode(m) {
    // Cancel ongoing polygon drawing if switching away
    if (state.mode === 'polygon' && m !== 'polygon') {
        _cancelPolygonDrawing();
    }

    state.mode = m;
    const btns = { select: els.modeSelect, adjust: els.modeAdjust, box: els.modeBox, polygon: els.modePolygon, sam: els.modeSam };
    Object.entries(btns).forEach(([key, btn]) => {
        if (!btn) return;
        btn.className = key === m ? 'btn btn-primary text-xs' : 'btn btn-ghost text-xs';
    });

    // Disable dragging on annotations
    annotLayer.getChildren().forEach(g => {
        if (g.name() === 'annotation') g.draggable(false);
    });

    if (m === 'adjust') {
        state.prevViz = {
            showBoxes: state.showBoxes,
            showBoundary: state.showBoundary,
            showSelectedOnly: state.showSelectedOnly,
            vizMode: els.vizMode?.value || 'both',
            vizSelectedOnly: els.vizSelectedOnly?.checked || false,
        };
        state.showBoxes = true;
        state.showBoundary = false;
        state.showSelectedOnly = true;
        if (els.vizMode) els.vizMode.value = 'bbox';
        if (els.vizSelectedOnly) els.vizSelectedOnly.checked = true;
    } else if (state.prevViz) {
        state.showBoxes = state.prevViz.showBoxes;
        state.showBoundary = state.prevViz.showBoundary;
        state.showSelectedOnly = state.prevViz.showSelectedOnly;
        if (els.vizMode) els.vizMode.value = state.prevViz.vizMode;
        if (els.vizSelectedOnly) els.vizSelectedOnly.checked = state.prevViz.vizSelectedOnly;
        state.prevViz = null;
    }

    if (m !== 'adjust') {
        clearAdjustHandles();
    }

    clearSelection();
    updateModeHint();
    renderVisualization();
    updateStatusBar();
    _updateSamPromptUI();
}

function updateModeHint() {
    const labels = { select: '选择', adjust: '调整BBox', box: '画框', polygon: '多边形', sam: 'SAM分割' };
    els.modeHint.textContent = labels[state.mode] || '';
    els.canvasContainer.style.cursor = (state.mode === 'box' || state.mode === 'sam' || state.mode === 'polygon') ? 'crosshair' : 'default';
}


// ══════════════════════════════════════════════════════════════
// Event listeners
// ══════════════════════════════════════════════════════════════

function setupEventListeners() {
    // Image group switch
    els.groupSelect.onchange = async () => {
        const previousGroupId = state.currentGroup?.group_id || '';
        const ok = await selectGroup(els.groupSelect.value);
        if (!ok) els.groupSelect.value = previousGroupId;
    };

    // Label set switch
    els.labelsetSelect.onchange = async () => {
        const previousSetId = state.currentLabelSet?.set_id || '';
        const ok = await selectLabelSet(els.labelsetSelect.value);
        if (!ok) {
            els.labelsetSelect.value = previousSetId;
            return;
        }
        await loadImages();
    };

    // Subset switch
    els.subsetSelect.onchange = async () => {
        const previousSubset = state.subset;
        if (!(await _ensureContextSaved('切换子集'))) {
            els.subsetSelect.value = previousSubset;
            return;
        }
        state.subset = els.subsetSelect.value;
        await loadImages();
    };

    // Generate bbox label set from polygon
    if (els.genBboxBtn) els.genBboxBtn.onclick = async () => {
        if (!state.currentGroup || !state.currentLabelSet) {
            showStatus('请先选择标注集', true); return;
        }
        const srcId = state.currentLabelSet.set_id;
        const srcFmt = state.currentLabelSet.label_format;
        if (srcFmt === 'bbox') {
            showStatus('当前已是BBox标注集，无需转换', true); return;
        }
        if (typeof _dmPrompt === 'function') {
            _dmPrompt(`生成BBox标注集 (源: ${state.currentLabelSet.set_name})`, `${srcId}_bbox`, async (name) => {
                showLoading(true, '生成BBox标注集');
                try {
                    const res = await fetch(`${API}/generate_bbox_set`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            group_id: state.currentGroup.group_id,
                            source_set_id: srcId,
                            target_name: name,
                        }),
                    });
                    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
                    const data = await res.json();
                    await loadGroups();
                    await selectGroup(state.currentGroup.group_id);
                    await selectLabelSet(data.set_id);
                    await loadImages();
                    showStatus(`✓ 已生成BBox标注集: ${data.files} 文件, ${data.boxes} 个框`);
                } catch (e) {
                    showStatus(`生成BBox失败: ${e.message}`, true);
                } finally {
                    showLoading(false);
                }
            });
        }
    };

    // Directory search — label sets
    if (els.scanLabelsetsBtn) els.scanLabelsetsBtn.onclick = () => {
        const panel = els.labelsetSearchPanel;
        if (!panel) return;
        const visible = panel.style.display !== 'none';
        panel.style.display = visible ? 'none' : 'block';
        if (!visible) {
            els.labelsetSearchInput.value = '';
            els.labelsetSearchInput.focus();
            _searchLabelDirs('');
        }
    };
    if (els.labelsetSearchInput) els.labelsetSearchInput.oninput = debounce(() => {
        _searchLabelDirs(els.labelsetSearchInput.value);
    }, 250);

    // Directory search — image groups
    if (els.scanGroupsBtn) els.scanGroupsBtn.onclick = () => {
        const panel = els.groupSearchPanel;
        if (!panel) return;
        const visible = panel.style.display !== 'none';
        panel.style.display = visible ? 'none' : 'block';
        if (!visible) {
            els.groupSearchInput.value = '';
            els.groupSearchInput.focus();
            _searchImageDirs('');
        }
    };
    if (els.groupSearchInput) els.groupSearchInput.oninput = debounce(() => {
        _searchImageDirs(els.groupSearchInput.value);
    }, 250);

    // Add class
    els.addClassBtn.onclick = () => {
        els.addClassForm.classList.remove('hidden');
        els.newClassName.value = '';
        els.newClassName.focus();
    };
    els.addClassNo.onclick = () => {
        els.addClassForm.classList.add('hidden');
    };
    els.addClassOk.onclick = () => {
        const name = els.newClassName.value;
        if (name.trim()) {
            addClass(name);
            els.addClassForm.classList.add('hidden');
        }
    };
    els.newClassName.onkeydown = (e) => {
        if (e.key === 'Enter') els.addClassOk.click();
        if (e.key === 'Escape') els.addClassNo.click();
    };

    // Mode buttons
    els.modeSelect.onclick = () => setMode('select');
    els.modeAdjust.onclick = () => setMode('adjust');
    els.modeBox.onclick = () => setMode('box');
    if (els.modePolygon) els.modePolygon.onclick = () => setMode('polygon');
    els.modeSam.onclick = () => setMode('sam');

    // SAM3 strategy checkboxes
    document.querySelectorAll('.sam-strat-cb').forEach(cb => {
        cb.onchange = () => _updateSamPromptUI();
    });
    if (els.samTextPrompt) els.samTextPrompt.oninput = debounce(() => _updateSamPromptUI(), 300);
    if (els.samTextRun) els.samTextRun.onclick = runSAMTextOnly;
    if (els.samBatchBtn) els.samBatchBtn.onclick = runSAMBatch;
    if (els.samAcceptBtn) els.samAcceptBtn.onclick = acceptSAMPreview;
    if (els.samRejectBtn) els.samRejectBtn.onclick = rejectSAMPreview;

    // Smart classification panel
    if (els.classifyMethod) els.classifyMethod.onchange = () => {
        state.fewshot.classifyMethod = els.classifyMethod.value;
        updateClassifyMethodUI();
        renderFewshotPanel();
    };
    if (els.fewshotAddSelectedBtn) els.fewshotAddSelectedBtn.onclick = addSelectedAnnotationToFewshot;
    if (els.batchAddSupportBtn) els.batchAddSupportBtn.onclick = addSelectedAnnotationsAsSupports;
    if (els.fewshotRemoveSelectedBtn) els.fewshotRemoveSelectedBtn.onclick = removeCurrentSelectedFromFewshot;
    if (els.fewshotClearBtn) els.fewshotClearBtn.onclick = () => clearFewshotSupports(true);
    if (els.fewshotRunCurrentBtn) els.fewshotRunCurrentBtn.onclick = runFewshotCurrentPrediction;
    if (els.fewshotEvalSubsetBtn) els.fewshotEvalSubsetBtn.onclick = runFewshotSubsetEvaluation;
    if (els.fewshotBatchClassifyBtn) els.fewshotBatchClassifyBtn.onclick = runFewshotBatchClassifyAndSave;
    if (els.fewshotApplyBtn) els.fewshotApplyBtn.onclick = applyFewshotResult;
    if (els.fewshotRevertBtn) els.fewshotRevertBtn.onclick = revertFewshotPreview;
    if (els.fewshotSaveSupportsBtn) els.fewshotSaveSupportsBtn.onclick = () => {
        persistSupports();
        setFewshotStatus('Support 已保存');
        renderFewshotPanel();
    };
    if (els.hybridEnableSizeRefiner) els.hybridEnableSizeRefiner.onchange = () => {
        const enabled = els.hybridEnableSizeRefiner.checked;
        if (els.hybridSrMargin) els.hybridSrMargin.disabled = !enabled;
        if (els.hybridSrSeparation) els.hybridSrSeparation.disabled = !enabled;
        if (els.hybridSrScale) els.hybridSrScale.disabled = !enabled;
        if (els.hybridSrMaxAdjust) els.hybridSrMaxAdjust.disabled = !enabled;
    };
    if (els.hybridPromptTemplate) els.hybridPromptTemplate.oninput = () => {
        state.fewshot.hybridPromptTemplate = els.hybridPromptTemplate.value;
        persistHybridPromptNames();
    };
    if (els.hybridPromptClearBtn) els.hybridPromptClearBtn.onclick = () => {
        state.fewshot.hybridPromptTemplate = '';
        state.fewshot.hybridTextPromptNames = {};
        if (els.hybridPromptTemplate) els.hybridPromptTemplate.value = '';
        persistHybridPromptNames();
        renderFewshotPanel();
    };
    if (els.fewshotSupportFilter) els.fewshotSupportFilter.onchange = () => {
        state.fewshot.supportFilterClass = els.fewshotSupportFilter.value;
        renderFewshotPanel();
    };
    if (els.fewshotEvalScope) els.fewshotEvalScope.onchange = () => {
        state.fewshot.evalScope = els.fewshotEvalScope.value;
        if (state.fewshot.evalScope === 'current_to_end' && state.currentImageIndex >= 0) {
            state.fewshot.rangeStart = String(state.currentImageIndex);
            state.fewshot.rangeEnd = String(Math.max(state.currentImageIndex, state.images.length - 1));
        }
        renderFewshotPanel();
    };
    if (els.fewshotRangeStart) els.fewshotRangeStart.onchange = () => {
        state.fewshot.rangeStart = els.fewshotRangeStart.value;
        if (Number(state.fewshot.rangeEnd) < Number(state.fewshot.rangeStart)) {
            state.fewshot.rangeEnd = state.fewshot.rangeStart;
        }
        renderFewshotPanel();
    };
    if (els.fewshotRangeEnd) els.fewshotRangeEnd.onchange = () => {
        state.fewshot.rangeEnd = els.fewshotRangeEnd.value;
        if (Number(state.fewshot.rangeEnd) < Number(state.fewshot.rangeStart)) {
            state.fewshot.rangeStart = state.fewshot.rangeEnd;
        }
        renderFewshotPanel();
    };
    if (els.fewshotUsePrompts) els.fewshotUsePrompts.onchange = () => {
        state.fewshot.usePrompts = els.fewshotUsePrompts.checked;
        renderFewshotPanel();
    };
    if (els.fewshotPromptMode) els.fewshotPromptMode.onchange = () => {
        state.fewshot.promptMode = els.fewshotPromptMode.value;
        renderFewshotPanel();
    };
    if (els.fewshotImageProtoWeight) els.fewshotImageProtoWeight.oninput = () => {
        state.fewshot.imageProtoWeight = Math.max(0, Math.min(1, Number(els.fewshotImageProtoWeight.value) || 0.5));
    };
    if (els.fewshotTextProtoWeight) els.fewshotTextProtoWeight.oninput = () => {
        state.fewshot.textProtoWeight = Math.max(0, Math.min(1, Number(els.fewshotTextProtoWeight.value) || 0.5));
    };
    if (els.fewshotPromptResetBtn) els.fewshotPromptResetBtn.onclick = () => {
        state.fewshot.promptEnsembles = {};
        state.fewshot.promptMode = 'auto';
        state.fewshot.usePrompts = false;
        state.fewshot.imageProtoWeight = 0.5;
        state.fewshot.textProtoWeight = 0.5;
        renderFewshotPanel();
        setFewshotStatus('已恢复默认提示设置');
    };

    // Save
    els.saveBtn.onclick = saveAnnotations;
    // Save & Next
    if (els.saveNextBtn) els.saveNextBtn.onclick = saveNextAndGo;

    // Adjust confirm/cancel
    els.adjustConfirm.onclick = confirmAdjust;
    els.adjustCancel.onclick = cancelAdjust;

    // Delete (with undo support)
    els.deleteBtn.onclick = () => {
        if (selectedGroups.size === 0) return;
        pushUndoState();
        const toDestroy = [...selectedGroups];
        toDestroy.forEach(g => g.destroy());
        selectedGroup = null;
        selectedGroups.clear();
        els.deleteBtn.classList.add('hidden');
        if (els.batchAddSupportBtn) els.batchAddSupportBtn.classList.add('hidden');
        setDirty(true);
        scheduleUIUpdate();
    };

    // Clear all annotations
    const clearAllBtn = $('clear-all-btn');
    if (clearAllBtn) clearAllBtn.onclick = clearAllAnnotations;

    // Navigation
    els.prevBtn.onclick = () => selectImage(state.currentImageIndex - 1);
    els.nextBtn.onclick = () => selectImage(state.currentImageIndex + 1);

    // Image list: click to select image (event delegation)
    els.imageList.addEventListener('click', (e) => {
        const item = e.target.closest('.img-item');
        if (!item) return;
        const idx = item.getAttribute('data-idx');
        if (idx !== null) {
            const i = parseInt(idx, 10);
            if (!isNaN(i) && i >= 0 && i < state.images.length) selectImage(i);
        }
    });

    // Search: debounce to avoid re-render on every keystroke
    els.searchInput.oninput = debounce(() => renderImageList(true), 220);
    if (els.imageStatusFilter) els.imageStatusFilter.onchange = () => renderImageList(true);

    // Virtual list: on scroll update visible slice (throttled)
    els.imageList.addEventListener('scroll', throttle(() => {
        if (_filteredIndices.length >= 500) _renderImageListVirtual();
    }, 50));

    // Class change on selected annotation (with undo)
    els.classSelect.onchange = () => {
        if (selectedGroup) {
            pushUndoState();
            selectedGroup.setAttr('classId', parseInt(els.classSelect.value));
            const poly = selectedGroup.findOne('Line');
            poly.stroke('white');
            setDirty(true);
        }
        renderClassList();
        renderAnnotationList();
        renderVisualization();
        renderFewshotPanel();
    };

    // Visualization controls
    els.vizMode.onchange = () => setVisualizationMode(els.vizMode.value);
    els.vizSelectedOnly.onchange = () => {
        state.showSelectedOnly = els.vizSelectedOnly.checked;
        renderVisualization();
    };

    // ── Comprehensive Keyboard Shortcuts System ─────────────────
    document.addEventListener('keydown', (e) => {
        const inInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT';
        const ctrl = e.ctrlKey || e.metaKey;

        // Global shortcuts that work even in inputs
        if (ctrl && e.key === 's') { e.preventDefault(); saveAnnotations(); return; }
        if (ctrl && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
        if (ctrl && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); redo(); return; }
        if (ctrl && e.key === 'Enter') { e.preventDefault(); saveNextAndGo(); return; }

        if (inInput) return;

        // ── Modifier combos ──

        if (ctrl && e.key === 'g') { e.preventDefault(); runCellposeSegment(); return; }

        // Ctrl+C: copy selected annotations to clipboard
        if (ctrl && e.key === 'c' && selectedGroups.size > 0) {
            e.preventDefault();
            state._clipboard = [];
            selectedGroups.forEach(g => {
                state._clipboard.push({
                    classId: g.getAttr('classId'),
                    annType: g.getAttr('annType'),
                    normPoints: g.getAttr('normPoints').slice(),
                });
            });
            showStatus(`已复制 ${state._clipboard.length} 个标注`);
            return;
        }

        // Ctrl+V: paste annotations from clipboard
        if (ctrl && e.key === 'v' && state._clipboard && state._clipboard.length > 0
            && state.currentImageIndex >= 0 && state.currentGroup) {
            e.preventDefault();
            pushUndoState();
            const offset = 0.015;
            state._clipboard.forEach(item => {
                const pts = item.normPoints.slice();
                if (item.annType === 'bbox') {
                    pts[0] = Math.min(1, pts[0] + offset);
                    pts[1] = Math.min(1, pts[1] + offset);
                } else {
                    for (let i = 0; i < pts.length; i += 2) {
                        pts[i] = Math.min(1, pts[i] + offset);
                        pts[i + 1] = Math.min(1, pts[i + 1] + offset);
                    }
                }
                createAnnotationShape(item.classId, item.annType, pts);
            });
            setDirty(true);
            scheduleUIUpdate();
            showStatus(`已粘贴 ${state._clipboard.length} 个标注`);
            return;
        }

        // Ctrl+A: select all annotations
        if (ctrl && e.key === 'a') {
            e.preventDefault();
            annotLayer.getChildren().forEach(g => {
                if (g.name() === 'annotation') selectedGroups.add(g);
            });
            if (selectedGroups.size > 0) {
                selectedGroup = [...selectedGroups][0];
                renderVisualization();
                renderAnnotationList();
                els.deleteBtn.classList.remove('hidden');
                showStatus(`已选中 ${selectedGroups.size} 个标注`);
            }
            return;
        }

        // Ctrl+D: duplicate with offset
        if (ctrl && e.key === 'd' && selectedGroup) {
            e.preventDefault();
            pushUndoState();
            const classId = selectedGroup.getAttr('classId');
            const annType = selectedGroup.getAttr('annType');
            const pts = selectedGroup.getAttr('normPoints').slice();
            const offset = 0.02;
            if (annType === 'bbox') {
                pts[0] = Math.min(1, pts[0] + offset);
                pts[1] = Math.min(1, pts[1] + offset);
            } else {
                for (let i = 0; i < pts.length; i += 2) {
                    pts[i] = Math.min(1, pts[i] + offset);
                    pts[i + 1] = Math.min(1, pts[i + 1] + offset);
                }
            }
            createAnnotationShape(classId, annType, pts);
            setDirty(true);
            scheduleUIUpdate();
            showStatus('已复制标注');
            return;
        }

        // Ctrl+0: fit to screen
        if (ctrl && e.key === '0') { e.preventDefault(); if (state.imageWidth > 0) fitImageToView(); return; }

        // Ctrl+1: actual pixels (1:1 zoom)
        if (ctrl && e.key === '1') {
            e.preventDefault();
            if (state.imageWidth > 0) {
                stage.scale({ x: 1, y: 1 });
                stage.position({ x: 0, y: 0 });
                _clampStagePosition();
                updateZoomUI(1);
                stage.batchDraw();
                showStatus('实际像素 (100%)');
            }
            return;
        }

        // Ctrl+= / Ctrl+-: zoom in/out
        if (ctrl && (e.key === '=' || e.key === '+')) {
            e.preventDefault();
            const s = Math.min(20, stage.scaleX() * 1.25);
            stage.scale({ x: s, y: s });
            _clampStagePosition();
            updateZoomUI(s);
            stage.batchDraw();
            return;
        }
        if (ctrl && e.key === '-') {
            e.preventDefault();
            const s = Math.max(0.05, stage.scaleX() * 0.8);
            stage.scale({ x: s, y: s });
            _clampStagePosition();
            updateZoomUI(s);
            stage.batchDraw();
            return;
        }

        // Ctrl+Shift+Arrow: jump to flagged images (must be checked before Ctrl+Arrow)
        if (ctrl && e.shiftKey && e.key === 'ArrowLeft') {
            e.preventDefault();
            _jumpToFlaggedImage(-1);
            return;
        }
        if (ctrl && e.shiftKey && e.key === 'ArrowRight') {
            e.preventDefault();
            _jumpToFlaggedImage(1);
            return;
        }

        // Ctrl+ArrowLeft/Right: jump to unlabeled image
        if (ctrl && !e.shiftKey && e.key === 'ArrowLeft') {
            e.preventDefault();
            for (let i = state.currentImageIndex - 1; i >= 0; i--) {
                if (!state.images[i].has_label) { selectImage(i); return; }
            }
            showStatus('没有更多未标注图像');
            return;
        }
        if (ctrl && !e.shiftKey && e.key === 'ArrowRight') {
            e.preventDefault();
            for (let i = state.currentImageIndex + 1; i < state.images.length; i++) {
                if (!state.images[i].has_label) { selectImage(i); return; }
            }
            showStatus('没有更多未标注图像');
            return;
        }

        // ── Single key shortcuts ──

        // Delete / Backspace: delete selected (but not during polygon drawing)
        if ((e.key === 'Delete' || e.key === 'Backspace') && selectedGroups.size > 0
            && !(state.mode === 'polygon' && state.polygonPoints.length > 0)) {
            e.preventDefault();
            els.deleteBtn.click();
            return;
        }

        // Enter: confirm adjust
        if (e.key === 'Enter' && state.mode === 'adjust' && adjustRect) { confirmAdjust(); return; }

        // Escape: multi-purpose cancel
        if (e.key === 'Escape') {
            // Cancel polygon drawing
            if (state.mode === 'polygon' && state.polygonPoints.length > 0) {
                _cancelPolygonDrawing();
                showStatus('多边形绘制已取消');
                return;
            }
            // Close shortcut help if open
            const helpOverlay = document.getElementById('shortcut-help-overlay');
            if (helpOverlay && helpOverlay.classList.contains('show')) {
                toggleShortcutHelp();
                return;
            }
            // Close data manager if open
            const dmModal = document.getElementById('dataset-manager-modal');
            if (dmModal && dmModal.style.display !== 'none') {
                dmModal.style.display = 'none';
                return;
            }
            if (state.mode === 'adjust' && adjustRect) { cancelAdjust(); return; }
            if (state.isDrawing) {
                state.isDrawing = false;
                if (tempRect) { tempRect.destroy(); tempRect = null; }
                if (tempLabel) { tempLabel.destroy(); tempLabel = null; }
                tempLayer.draw();
                return;
            }
            if (selectedGroup) { clearSelection(); renderVisualization(); return; }
            return;
        }

        // Space: hold for pan mode (press once without drag → fit to view)
        if (e.key === ' ' && !state._spaceHeld) {
            e.preventDefault();
            state._spaceHeld = true;
            state._prePanMode = state.mode;
            els.canvasContainer.style.cursor = 'grab';
            stage.draggable(true);
            return;
        }

        // 1-9: quick class switch (Shift+1-9: set class on selected)
        if (e.key >= '1' && e.key <= '9') {
            const idx = parseInt(e.key, 10) - 1;
            if (state.classes[idx]) {
                if (e.shiftKey && selectedGroups.size > 0) {
                    pushUndoState();
                    const newCid = parseInt(state.classes[idx].ID);
                    const newColor = state.colors[newCid] || '#fff';
                    selectedGroups.forEach(g => {
                        g.setAttr('classId', newCid);
                        const poly = g.findOne('Line');
                        if (poly) {
                            poly.stroke('#ffffff');
                            poly.fill(hexToRgba(newColor, 0.18));
                        }
                    });
                    setDirty(true);
                    showStatus(`已将 ${selectedGroups.size} 个标注设为: ${state.classes[idx].Name}`);
                } else {
                    els.classSelect.value = state.classes[idx].ID;
                    if (selectedGroup) {
                        pushUndoState();
                        selectedGroup.setAttr('classId', parseInt(state.classes[idx].ID));
                        const poly = selectedGroup.findOne('Line');
                        if (poly) poly.stroke('white');
                        setDirty(true);
                    }
                    showStatus(`类别: ${state.classes[idx].Name}`);
                }
                renderClassList();
                renderAnnotationList();
                renderVisualization();
            }
            return;
        }

        // D: duplicate selected (no ctrl)
        if (e.key === 'd' && selectedGroup && !ctrl) {
            pushUndoState();
            const classId = selectedGroup.getAttr('classId');
            const annType = selectedGroup.getAttr('annType');
            const pts = selectedGroup.getAttr('normPoints').slice();
            const offset = 0.02;
            if (annType === 'bbox') {
                pts[0] = Math.min(1, pts[0] + offset);
                pts[1] = Math.min(1, pts[1] + offset);
            } else {
                for (let i = 0; i < pts.length; i += 2) {
                    pts[i] = Math.min(1, pts[i] + offset);
                    pts[i + 1] = Math.min(1, pts[i + 1] + offset);
                }
            }
            createAnnotationShape(classId, annType, pts);
            setDirty(true);
            scheduleUIUpdate();
            showStatus('已复制标注');
            return;
        }

        // Polygon-specific shortcuts (before mode switch)
        if (state.mode === 'polygon' && state.polygonPoints.length > 0) {
            if (e.key === 'Enter') { e.preventDefault(); _finishPolygonDrawing(); return; }
            if (e.key === 'Backspace') { e.preventDefault(); _undoLastPolygonPoint(); return; }
        }

        // Mode shortcuts
        if (e.key === 'v') { setMode('select'); return; }
        if (e.key === 'a') { setMode('adjust'); return; }
        if (e.key === 'b') { setMode('box'); return; }
        if (e.key === 'p') { setMode('polygon'); return; }
        if (e.key === 's' && !ctrl) { setMode('sam'); return; }
        if (e.key === 'r') { setMode('box'); return; }
        if (e.key === 'h') { setMode('select'); return; }

        // Navigation
        if (e.key === 'ArrowLeft') { selectImage(state.currentImageIndex - 1); return; }
        if (e.key === 'ArrowRight') { selectImage(state.currentImageIndex + 1); return; }
        if (e.key === 'Home') { e.preventDefault(); selectImage(0); return; }
        if (e.key === 'End') { e.preventDefault(); selectImage(state.images.length - 1); return; }

        // (fullscreen toggle removed — sidebars always visible)

        // ?: show shortcut help
        if (e.key === '?' || (e.shiftKey && e.key === '/')) {
            e.preventDefault();
            toggleShortcutHelp();
            return;
        }

        // M: toggle data manager
        if (e.key === 'm' && !ctrl) {
            const dmModal = document.getElementById('dataset-manager-modal');
            if (dmModal) {
                dmModal.style.display = dmModal.style.display === 'none' ? 'flex' : 'none';
                if (dmModal.style.display === 'flex' && typeof DatasetManager !== 'undefined') {
                    DatasetManager.init();
                }
            }
            return;
        }

        // G: mark current image as done/review
        if (e.key === 'g' && !ctrl) {
            toggleImageFlag('review_status');
            return;
        }
    });

    // Space release: end pan mode
    document.addEventListener('keyup', (e) => {
        if (e.key === ' ' && state._spaceHeld) {
            state._spaceHeld = false;
            stage.draggable(false);
            stage.stopDrag();
            _clampStagePosition();
            updateModeHint();
        }
    });

    // Mouse-centered zoom with Ctrl+scroll
    if (els.canvasContainer) {
        els.canvasContainer.addEventListener('wheel', (e) => {
            if (!e.ctrlKey) return;
            e.preventDefault();
            const oldScale = stage.scaleX();
            const pointer = stage.getPointerPosition();
            if (!pointer) return;
            const factor = e.deltaY < 0 ? 1.1 : 0.9;
            const newScale = Math.max(0.1, Math.min(5, oldScale * factor));
            const mousePointTo = {
                x: (pointer.x - stage.x()) / oldScale,
                y: (pointer.y - stage.y()) / oldScale,
            };
            stage.scale({ x: newScale, y: newScale });
            stage.position({
                x: pointer.x - mousePointTo.x * newScale,
                y: pointer.y - mousePointTo.y * newScale,
            });
            _clampStagePosition();
            updateZoomUI(newScale);
            stage.batchDraw();
        }, { passive: false });
    }

    // CellposeSAM visualization
    if (els.cellposeRunBtn) els.cellposeRunBtn.onclick = runCellposeSegment;
    if (els.cellposeClearBtn) els.cellposeClearBtn.onclick = clearCellposeOverlay;
    if (els.cellposeAcceptBtn) els.cellposeAcceptBtn.onclick = acceptCellposePreview;
    if (els.cellposeRejectBtn) els.cellposeRejectBtn.onclick = rejectCellposePreview;
    if (els.cellposeBatchAllBtn) els.cellposeBatchAllBtn.onclick = () => {
        if (!state.images.length) return;
        els.cellposeBatchStart.value = '0';
        els.cellposeBatchEnd.value = String(state.images.length - 1);
    };
    if (els.cellposeBatchTailBtn) els.cellposeBatchTailBtn.onclick = () => {
        if (!state.images.length || state.currentImageIndex < 0) return;
        els.cellposeBatchStart.value = String(state.currentImageIndex);
        els.cellposeBatchEnd.value = String(state.images.length - 1);
    };
    if (els.cellposeBatchRunBtn) els.cellposeBatchRunBtn.onclick = runCellposeBatchSegment;
    if (els.cellposeBatchCancelBtn) els.cellposeBatchCancelBtn.onclick = cancelCellposeBatchSegment;
    if (els.cellposeBatchSaveBtn) els.cellposeBatchSaveBtn.onclick = confirmCellposeBatchSave;
    if (els.cellposeBatchDiscardBtn) els.cellposeBatchDiscardBtn.onclick = discardCellposeBatchSave;
    if (els.cellposeBatchStart) els.cellposeBatchStart.onchange = updateCellposeBatchRunState;
    if (els.cellposeBatchEnd) els.cellposeBatchEnd.onchange = updateCellposeBatchRunState;
    if (els.cellposeBatchSkip) els.cellposeBatchSkip.onchange = () => {
        if (els.cellposeBatchSkip.checked && els.cellposeBatchOverwrite) {
            els.cellposeBatchOverwrite.checked = false;
        }
        updateCellposeBatchRunState();
    };
    if (els.cellposeBatchOverwrite) els.cellposeBatchOverwrite.onchange = () => {
        if (els.cellposeBatchOverwrite.checked && els.cellposeBatchSkip) {
            els.cellposeBatchSkip.checked = false;
        }
        updateCellposeBatchRunState();
    };
    if (els.cellposeVisible) els.cellposeVisible.onchange = () => {
        cellposeOverlayLayer.visible(els.cellposeVisible.checked);
        cellposeOverlayLayer.batchDraw();
    };
    if (els.cellposeColor) els.cellposeColor.onchange = () => {
        const color = _getCellposeColor();
        cellposeOverlayLayer.getChildren().forEach(line => {
            line.stroke(color);
            line.fill(_hexToRgba(color, 0.12));
        });
        cellposeOverlayLayer.batchDraw();
    };
    if (els.fewshotManageBtn) {
        els.fewshotManageBtn.onclick = () => {
            const isVisible = els.fewshotSupportPopover && !els.fewshotSupportPopover.classList.contains('collapsed-panel');
            setFewshotSupportPopoverVisible(!isVisible);
        };
    }
    if (els.fewshotPopoverClose) {
        els.fewshotPopoverClose.onclick = () => setFewshotSupportPopoverVisible(false);
    }
    if (els.evalSegmentMethod) els.evalSegmentMethod.onchange = () => renderResearchEvalPanel();
    if (els.evalClassifyGoldBtn) els.evalClassifyGoldBtn.onclick = runGoldClassificationEvaluation;
    if (els.evalSegmentGoldBtn) els.evalSegmentGoldBtn.onclick = runGoldSegmentationEvaluation;

    // Disable context menu on canvas (right-click is used for pan)
    els.canvasContainer.addEventListener('contextmenu', (e) => e.preventDefault());

    // Warn before leaving with unsaved changes
    window.addEventListener('beforeunload', (e) => {
        if (state.dirty) {
            e.preventDefault();
            e.returnValue = '';
        }
    });
}

// ══════════════════════════════════════════════════════════════
// Polygon Drawing Tool
// ══════════════════════════════════════════════════════════════

function _cancelPolygonDrawing() {
    state.polygonPoints = [];
    if (state.polygonKonvaLine) { state.polygonKonvaLine.destroy(); state.polygonKonvaLine = null; }
    if (state.polygonPreviewLine) { state.polygonPreviewLine.destroy(); state.polygonPreviewLine = null; }
    state.polygonKonvaDots.forEach(d => d.destroy());
    state.polygonKonvaDots = [];
    tempLayer.batchDraw();
}

function _addPolygonPoint(pos) {
    const color = getCurrentColor();
    const s = stage.scaleX();
    state.polygonPoints.push({ x: pos.x, y: pos.y });

    // Draw vertex dot
    const dot = new Konva.Circle({
        x: pos.x, y: pos.y,
        radius: 4 / s,
        fill: color,
        stroke: '#fff',
        strokeWidth: 1 / s,
        listening: false,
    });
    tempLayer.add(dot);
    state.polygonKonvaDots.push(dot);

    // Update polyline
    const flatPts = state.polygonPoints.flatMap(p => [p.x, p.y]);
    if (state.polygonKonvaLine) {
        state.polygonKonvaLine.points(flatPts);
    } else {
        state.polygonKonvaLine = new Konva.Line({
            points: flatPts,
            stroke: color,
            strokeWidth: 2 / s,
            fill: hexToRgba(color, 0.08),
            closed: false,
            listening: false,
        });
        tempLayer.add(state.polygonKonvaLine);
    }

    // Create/update preview line (from last point to cursor)
    if (!state.polygonPreviewLine) {
        state.polygonPreviewLine = new Konva.Line({
            points: [pos.x, pos.y, pos.x, pos.y],
            stroke: color,
            strokeWidth: 1.5 / s,
            dash: [6 / s, 4 / s],
            opacity: 0.6,
            listening: false,
        });
        tempLayer.add(state.polygonPreviewLine);
    }
    state.polygonPreviewLine.points([pos.x, pos.y, pos.x, pos.y]);

    tempLayer.batchDraw();
    showStatus(`多边形: ${state.polygonPoints.length} 个顶点 (Enter完成, Esc取消, Backspace回退)`);
}

function _undoLastPolygonPoint() {
    if (state.polygonPoints.length === 0) return;
    state.polygonPoints.pop();
    const lastDot = state.polygonKonvaDots.pop();
    if (lastDot) lastDot.destroy();

    if (state.polygonPoints.length > 0) {
        const flatPts = state.polygonPoints.flatMap(p => [p.x, p.y]);
        if (state.polygonKonvaLine) state.polygonKonvaLine.points(flatPts);
        const last = state.polygonPoints[state.polygonPoints.length - 1];
        if (state.polygonPreviewLine) state.polygonPreviewLine.points([last.x, last.y, last.x, last.y]);
    } else {
        if (state.polygonKonvaLine) { state.polygonKonvaLine.destroy(); state.polygonKonvaLine = null; }
        if (state.polygonPreviewLine) { state.polygonPreviewLine.destroy(); state.polygonPreviewLine = null; }
    }
    tempLayer.batchDraw();
    showStatus(state.polygonPoints.length > 0 ?
        `多边形: ${state.polygonPoints.length} 个顶点` : '多边形: 点击放置第一个顶点');
}

function _finishPolygonDrawing() {
    if (state.polygonPoints.length < 3) {
        showStatus('多边形至少需要3个顶点', true);
        return;
    }

    pushUndoState();
    const classId = parseInt(els.classSelect.value);
    const normPoints = [];
    for (const p of state.polygonPoints) {
        normPoints.push(p.x / state.imageWidth, p.y / state.imageHeight);
    }
    createAnnotationShape(classId, 'polygon', normPoints);
    setDirty(true);
    _cancelPolygonDrawing();
    scheduleUIUpdate();
    showStatus('多边形标注已创建');
}

function _updatePolygonPreview(stagePos) {
    if (state.mode !== 'polygon' || state.polygonPoints.length === 0 || !state.polygonPreviewLine) return;
    const imgPos = clampToImage(stageToImage(stagePos));
    const last = state.polygonPoints[state.polygonPoints.length - 1];
    state.polygonPreviewLine.points([last.x, last.y, imgPos.x, imgPos.y]);
    tempLayer.batchDraw();
}

let _isFullscreenCanvas = false;
function toggleFullscreenCanvas() { /* disabled — sidebars always visible */ }
window.toggleFullscreenCanvas = toggleFullscreenCanvas;


// ══════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════

function stageToImage(stagePos) {
    const t = stage.getAbsoluteTransform().copy();
    t.invert();
    return t.point(stagePos);
}

function clampToImage(pos) {
    return {
        x: clamp(pos.x, 0, state.imageWidth),
        y: clamp(pos.y, 0, state.imageHeight),
    };
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function getCurrentColor() {
    const classId = parseInt(els.classSelect.value);
    return state.colors[classId] || '#ffffff';
}

function hexToRgba(hex, alpha) {
    return _hexToRgba(hex, alpha);
}

function _updateEmptyState() {
    const overlay = document.getElementById('canvas-empty-state');
    if (!overlay) return;
    const hasContent = state.currentGroup && state.currentImageIndex >= 0;
    overlay.classList.toggle('hidden', !!hasContent);

    if (hasContent) return;

    const iconEl = overlay.querySelector('.canvas-empty-state__icon');
    const titleEl = overlay.querySelector('.canvas-empty-state__title');
    const descEl = overlay.querySelector('.canvas-empty-state__desc');

    if (!state.groups || state.groups.length === 0) {
        if (iconEl) iconEl.innerHTML = '<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.7;"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/><line x1="12" y1="11" x2="12" y2="17"/><line x1="9" y1="14" x2="15" y2="14"/></svg>';
        if (titleEl) titleEl.textContent = '暂无数据集';
        if (descEl) descEl.textContent = '系统未发现可用的数据集。请通过数据管理中心添加新的数据集目录。';
    } else if (!state.currentGroup) {
        if (titleEl) titleEl.textContent = '请选择数据集';
        if (descEl) descEl.textContent = '从左侧图片集下拉菜单选择一个数据集，或通过数据管理中心打开。';
    } else if (state.images.length === 0) {
        if (iconEl) iconEl.innerHTML = '<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--warn)" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.7;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>';
        if (titleEl) titleEl.textContent = '当前子集暂无图片';
        if (descEl) descEl.textContent = '该数据集的当前子集没有图片。请尝试切换子集（Train/Val），或选择其他数据集。';
    } else {
        if (iconEl) iconEl.innerHTML = '<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.7;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>';
        if (titleEl) titleEl.textContent = '尚未加载数据集';
        if (descEl) descEl.textContent = '选择一个数据集开始标注，或从数据管理中心浏览所有可用数据集。';
    }
}

function showLoading(show, message, opts = {}) {
    els.loading.classList.toggle('show', show);
    const textEl = els.loading.querySelector('.loader-text');
    if (textEl) {
        const msg = message || '加载中';
        textEl.innerHTML = msg + '<span class="loader-dots"></span>';
    }
    let stopBtn = els.loading.querySelector('.loading-stop-btn');
    if (show && opts.stoppable) {
        if (!stopBtn) {
            stopBtn = document.createElement('button');
            stopBtn.className = 'loading-stop-btn';
            stopBtn.innerHTML = '⏹ 停止';
            els.loading.appendChild(stopBtn);
        }
        stopBtn.style.display = '';
        stopBtn.disabled = false;
        stopBtn.onclick = () => {
            requestStopLongRun();
            stopBtn.disabled = true;
            stopBtn.textContent = '正在停止…';
            if (textEl) textEl.innerHTML = '正在停止，等待当前处理完成…';
        };
    } else if (stopBtn) {
        stopBtn.style.display = 'none';
    }
}

function showStatus(msg, isError = false) {
    if (!els.status) return;
    els.status.textContent = msg;
    els.status.style.color = isError ? 'var(--danger)' : 'var(--warn)';
    els.status.classList.add('visible');
    clearTimeout(showStatus._hideTimer);
    showStatus._hideTimer = setTimeout(() => {
        els.status.textContent = '';
        els.status.classList.remove('visible');
    }, 3200);
}

function setDirty(dirty) {
    state.dirty = dirty;
    els.unsavedDot.classList.toggle('visible', dirty);
    updateStatusBar();
    if (dirty) _debouncedSync();
}

const _debouncedSync = debounce(async () => {
    if (state.dirty && state.currentImageIndex >= 0) {
        await saveAnnotations();
    }
}, 600);

/** Sync footer status bar with current state */
function updateStatusBar() {
    if (els.footerMode) {
        const labels = { select: '选择', adjust: '调整', box: '画框', polygon: '多边形', sam: 'SAM' };
        els.footerMode.textContent = labels[state.mode] || state.mode;
    }
    if (els.footerImage) {
        const n = state.images.length;
        const i = state.currentImageIndex + 1;
        els.footerImage.textContent = n ? `${i} / ${n}` : '-';
    }
    if (els.footerAnns) {
        const count = annotLayer ? annotLayer.getChildren().filter(g => g.name() === 'annotation').length : 0;
        els.footerAnns.textContent = `标注: ${count}`;
    }
    const hasImage = state.currentImageIndex >= 0 && state.images.length > 0;
    if (els.footerDirty) els.footerDirty.style.display = (state.dirty && hasImage) ? '' : 'none';
    if (els.footerSaved) els.footerSaved.style.display = (!state.dirty && hasImage) ? '' : 'none';
    if (els.footerAutosave) {
        els.footerAutosave.textContent = state.dirty ? '等待自动保存' : '自动保存';
    }
    if (els.quickGroupName) {
        const text = state.currentGroup?.group_name || '未选择';
        els.quickGroupName.textContent = text;
        els.quickGroupName.title = text;
    }
    if (els.quickLabelsetName) {
        const text = state.currentLabelSet ? `${state.currentLabelSet.set_name} (${state.currentLabelSet.label_format})` : '-';
        els.quickLabelsetName.textContent = text;
        els.quickLabelsetName.title = text;
    }
    if (els.quickImageName) {
        const text = state.images[state.currentImageIndex]?.filename || '-';
        els.quickImageName.textContent = text;
        els.quickImageName.title = text;
    }
    if (els.quickSupportCount) {
        els.quickSupportCount.textContent = String(state.fewshot?.supports?.length || 0);
    }
}

function updateZoomUI(scale) {
    els.zoomLevel.textContent = `${Math.round(scale * 100)}%`;
}

function updateScaledSizes() {
    const s = stage.scaleX();
    if (tempRect) tempRect.strokeWidth(2 / s);
    if (tempLabel) tempLabel.fontSize(12 / s);

    // Adjust handles
    if (adjustRect) {
        adjustRect.strokeWidth(1.5 / s);
        adjustRect.dash([6 / s, 3 / s]);
    }
    adjustHandles.forEach(h => {
        h.size({ width: 8 / s, height: 8 / s });
        h.strokeWidth(1.5 / s);
        h.cornerRadius(1 / s);
    });
    if (adjustRect && adjustHandles.length === 4) {
        syncHandlesToRect();
    }

    // Update annotation polygon stroke widths
    annotLayer.getChildren().forEach(group => {
        if (group.name() !== 'annotation') return;
        const poly = group.findOne('Line');
        if (poly) poly.strokeWidth(2 / s);
    });

    // Update box overlay stroke widths
    boxLayer.getChildren().forEach(rect => {
        rect.strokeWidth(1.5 / s);
        if (rect.dash()) rect.dash([5 / s, 3 / s]);
    });

    // Update CellposeSAM overlay stroke widths
    cellposeOverlayLayer.getChildren().forEach(line => {
        line.strokeWidth(1.5 / s);
    });

    // Update SAM3 preview stroke widths
    samPreviewLayer.getChildren().forEach(line => {
        line.strokeWidth(2 / s);
        if (line.dash()) line.dash([6 / s, 3 / s]);
    });

    renderAnnotationLabels();
}


// ══════════════════════════════════════════════════════════════
// CellposeSAM Visualization Overlay (view-only, NOT saved as labels)
// ══════════════════════════════════════════════════════════════

function _getCellposeColor() {
    return els.cellposeColor?.value || '#00ff88';
}

function _drawCellposeOverlay(polygons) {
    cellposeOverlayLayer.destroyChildren();
    if (!currentImageNode) return;
    const imgW = currentImageNode.width();
    const imgH = currentImageNode.height();
    const color = _getCellposeColor();

    for (const poly of polygons) {
        const pts = [];
        for (let i = 0; i < poly.points.length; i += 2) {
            pts.push(poly.points[i] * imgW);
            pts.push(poly.points[i + 1] * imgH);
        }
        cellposeOverlayLayer.add(new Konva.Line({
            points: pts,
            closed: true,
            stroke: color,
            strokeWidth: 1.5 / (stage?.scaleX() || 1),
            fill: _hexToRgba(color, 0.12),
            opacity: 0.85,
            listening: false,
        }));
    }
    cellposeOverlayLayer.visible(els.cellposeVisible?.checked ?? true);
    cellposeOverlayLayer.batchDraw();
}

function _hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

async function runCellposeSegment() {
    if (!state.currentGroup || state.currentImageIndex < 0) {
        showStatus('请先选择一张图片', true);
        return;
    }

    const diameters = _getCellposeDiameters();
    if (diameters.length === 0) {
        showStatus('请输入有效的直径值', true);
        return;
    }

    const gpu = els.cellposeGpu?.checked ?? true;
    const img = state.images[state.currentImageIndex];

    _clearCellposePreview();
    const abortSignal = startLongRun();
    showLoading(true, 'CellposeSAM 分割中', { stoppable: true });
    if (els.cellposeStatus) els.cellposeStatus.textContent = '正在运行 CellposeSAM…';
    if (els.cellposeRunBtn) els.cellposeRunBtn.disabled = true;

    const t0 = Date.now();
    try {
        const res = await fetch(`${API}/cellpose_segment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                group_id: state.currentGroup.group_id,
                subset: state.subset,
                filename: img.filename,
                diameters: diameters,
                gpu: gpu,
                class_id: parseInt(els.classSelect.value) || 0,
                min_area: 100,
            }),
            signal: abortSignal,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

        if (data.annotations && data.annotations.length > 0) {
            cellposePreviewData = data.annotations;
            _drawCellposeOverlay(data.annotations);
            if (els.cellposeAcceptActions) els.cellposeAcceptActions.style.display = 'flex';
            const msg = `${data.count} 个细胞 (${elapsed}s) — 可保存或放弃`;
            if (els.cellposeStatus) els.cellposeStatus.textContent = msg;
            showStatus(`CellposeSAM: ${data.count} 个细胞轮廓`);
        } else {
            cellposeOverlayLayer.destroyChildren();
            cellposeOverlayLayer.batchDraw();
            const msg = `未检测到细胞 (${elapsed}s)`;
            if (els.cellposeStatus) els.cellposeStatus.textContent = msg;
            showStatus(msg);
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            if (els.cellposeStatus) els.cellposeStatus.textContent = 'CellposeSAM 已停止';
            showStatus('已停止 CellposeSAM 分割');
        } else {
            console.error('CellposeSAM error:', e);
            if (els.cellposeStatus) els.cellposeStatus.textContent = `✗ ${e.message}`;
            showStatus(`CellposeSAM 失败: ${e.message}`, true);
        }
    } finally {
        _longRunAbort = null;
        showLoading(false);
        if (els.cellposeRunBtn) els.cellposeRunBtn.disabled = false;
    }
}

async function runCellposeBatchSegment() {
    if (!state.currentGroup || !state.currentLabelSet) {
        showStatus('请先选择图片集和标注集', true);
        return;
    }
    if (!state.images.length) {
        showStatus('当前没有可处理图片', true);
        return;
    }
    if (state.currentLabelSet.label_format === 'bbox') {
        showStatus('当前标注集是 bbox，批量 CellposeSAM 仅支持 polygon 标注集', true);
        return;
    }
    if (activeCellposeBatchJobId) {
        showStatus('已有批量任务在运行', true);
        return;
    }

    const diameters = _getCellposeDiameters();
    if (diameters.length === 0) {
        showStatus('请输入有效的直径值', true);
        return;
    }

    const startIndex = Number(els.cellposeBatchStart?.value ?? -1);
    const endIndex = Number(els.cellposeBatchEnd?.value ?? -1);
    if (!Number.isInteger(startIndex) || !Number.isInteger(endIndex) || startIndex < 0 || endIndex < startIndex || endIndex >= state.images.length) {
        showStatus('请选择有效的起止图片范围', true);
        return;
    }

    if (state.dirty) {
        const ok = await saveAnnotations();
        if (!ok) return;
    }

    _clearCellposePreview();
    _clearSamPreview();

    const overwriteExisting = !!els.cellposeBatchOverwrite?.checked;
    const skipExisting = overwriteExisting ? false : !!els.cellposeBatchSkip?.checked;
    if (!skipExisting && !overwriteExisting) {
        showStatus('请明确选择“跳过已有标注”或“覆盖已有标注”', true);
        return;
    }
    const imgCount = endIndex - startIndex + 1;
    const rangeLabel = _formatBatchRangeLabel(startIndex, endIndex);

    showLoading(true, '创建批量分割任务');
    try {
        const res = await fetch(`${API}/cellpose_batch/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                group_id: state.currentGroup.group_id,
                label_set_id: state.currentLabelSet.set_id,
                subset: state.subset,
                start_index: startIndex,
                end_index: endIndex,
                diameters: diameters,
                gpu: els.cellposeGpu?.checked ?? true,
                class_id: parseInt(els.classSelect.value, 10) || 0,
                min_area: 100,
                skip_existing: skipExisting,
                overwrite_existing: overwriteExisting,
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        activeCellposeBatchJobId = data.job_id;
        cellposeBatchPhase = 'queued';
        _setCellposeBatchProgress({
            total: data.total || imgCount,
            processed: 0,
            current: data.start_filename || '任务排队中',
            saved: 0,
            skipped: 0,
            failed: 0,
            status: 'queued',
        });
        if (els.cellposeBatchStatus) {
            els.cellposeBatchStatus.textContent = `已启动 ${imgCount} 张图片的批量分割 | ${rangeLabel}`;
        }
        updateCellposeBatchRunState();
        showStatus(`批量分割已启动: ${imgCount} 张`);
        pollCellposeBatchStatus();
    } catch (e) {
        console.error('Cellpose batch start error:', e);
        cellposeBatchPhase = 'failed';
        _setCellposeBatchProgress({
            total: imgCount,
            processed: 0,
            current: '启动失败',
            saved: 0,
            skipped: 0,
            failed: 0,
            status: 'failed',
        });
        updateCellposeBatchRunState();
        if (els.cellposeBatchStatus) els.cellposeBatchStatus.textContent = `✗ ${e.message}`;
        showStatus(`批量分割启动失败: ${e.message}`, true);
    } finally {
        showLoading(false);
    }
}

async function pollCellposeBatchStatus() {
    if (!activeCellposeBatchJobId) return;
    _stopCellposeBatchPolling();

    try {
        const res = await fetch(`${API}/cellpose_batch/status?job_id=${encodeURIComponent(activeCellposeBatchJobId)}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const job = await res.json();
        cellposeBatchPhase = job.status || 'running';
        const total = job.total || 0;
        const processed = job.processed || 0;
        const current = job.current_filename || '';
        const summary = `已分割 ${job.saved || 0} / 跳过 ${job.skipped || 0} / 失败 ${job.failed || 0}`;

        _setCellposeBatchProgress({
            total,
            processed,
            current,
            saved: job.saved || 0,
            skipped: job.skipped || 0,
            failed: job.failed || 0,
            status: job.status || 'running',
        });
        _updateCellposeBatchActionButtons();

        if (els.cellposeBatchStatus) {
            if (job.status === 'queued') {
                els.cellposeBatchStatus.textContent = job.message || '任务排队中';
            } else if (job.status === 'cancel_requested') {
                els.cellposeBatchStatus.textContent = job.message || '正在取消，等待当前图片处理完成';
            } else if (job.status === 'running') {
                els.cellposeBatchStatus.textContent = `${job.message || '批量分割执行中'} | ${summary}`;
            } else if (job.status === 'awaiting_save') {
                const recent = Array.isArray(job.results) && job.results.length
                    ? job.results.slice(-3).map(item => `${item.filename}:${item.status}`).join(' | ')
                    : '';
                els.cellposeBatchStatus.textContent = `${job.message || '批量分割已结束，等待确认保存'}${recent ? ` | 最近: ${recent}` : ''}`;
            } else if (job.status === 'saved') {
                els.cellposeBatchStatus.textContent = job.message || '已保存全部分割结果';
            } else {
                const recent = Array.isArray(job.results) && job.results.length
                    ? job.results.slice(-3).map(item => `${item.filename}:${item.status}`).join(' | ')
                    : '';
                els.cellposeBatchStatus.textContent = `${job.message || ''}${recent ? ` | 最近: ${recent}` : ''}`;
            }
        }

        if (job.status === 'queued' || job.status === 'running' || job.status === 'cancel_requested') {
            cellposeBatchPollTimer = setTimeout(pollCellposeBatchStatus, 1200);
            return;
        }

        if (job.status === 'awaiting_save') {
            _setCellposeBatchProgress({
                total,
                processed,
                current: '浏览预览后确认保存',
                saved: job.saved || 0,
                skipped: job.skipped || 0,
                failed: job.failed || 0,
                status: 'awaiting_save',
            });
            _updateCellposeBatchActionButtons();
            showStatus('批量分割完成 — 切换图片可预览结果，确认后点击保存');
            _loadBatchPreviewForCurrentImage();
            return;
        }

        if (job.status === 'saved') {
            activeCellposeBatchJobId = null;
            cellposeBatchPhase = 'idle';
            _setCellposeBatchProgress({
                total,
                processed: job.processed || total,
                current: '已全部保存',
                saved: job.committed || job.saved || 0,
                skipped: job.skipped || 0,
                failed: job.failed || 0,
                status: 'saved',
            });
            updateCellposeBatchRunState();
            await reloadImagesKeepSelection();
            showStatus(job.message || '已保存全部分割结果');
        } else if (job.status === 'failed' && activeCellposeBatchJobId) {
            cellposeBatchPhase = 'failed';
            _setCellposeBatchProgress({
                total,
                processed,
                current: '保存失败，可重试',
                saved: job.committed || job.saved || 0,
                skipped: job.skipped || 0,
                failed: job.failed || 0,
                status: 'failed',
            });
            _updateCellposeBatchActionButtons();
            showStatus(job.message || '保存失败，可重试或放弃', true);
        } else {
            activeCellposeBatchJobId = null;
            cellposeBatchPhase = 'failed';
            _setCellposeBatchProgress({
                total,
                processed,
                current: '任务失败',
                saved: job.saved || 0,
                skipped: job.skipped || 0,
                failed: job.failed || 0,
                status: 'failed',
            });
            updateCellposeBatchRunState();
            showStatus(job.message || '批量分割失败', true);
        }
    } catch (e) {
        console.error('Cellpose batch poll error:', e);
        activeCellposeBatchJobId = null;
        cellposeBatchPhase = 'failed';
        _setCellposeBatchProgress({
            total: 0,
            processed: 0,
            current: '状态读取失败',
            saved: 0,
            skipped: 0,
            failed: 0,
            status: 'failed',
        });
        updateCellposeBatchRunState();
        if (els.cellposeBatchStatus) {
            els.cellposeBatchStatus.textContent = `✗ 读取批量任务状态失败: ${e.message}`;
        }
        showStatus(`读取批量任务状态失败: ${e.message}`, true);
    }
}

async function cancelCellposeBatchSegment() {
    if (!activeCellposeBatchJobId) return;
    try {
        const res = await fetch(`${API}/cellpose_batch/cancel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: activeCellposeBatchJobId }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        cellposeBatchPhase = 'cancel_requested';
        _updateCellposeBatchActionButtons();
        if (els.cellposeBatchStatus) {
            els.cellposeBatchStatus.textContent = '已请求取消，等待当前图片处理完成后停止';
        }
        showStatus('已请求取消批量任务');
    } catch (e) {
        showStatus(`取消失败: ${e.message}`, true);
    }
}

async function confirmCellposeBatchSave() {
    if (!activeCellposeBatchJobId) return;
    if (els.cellposeBatchSaveBtn) els.cellposeBatchSaveBtn.disabled = true;
    if (els.cellposeBatchDiscardBtn) els.cellposeBatchDiscardBtn.disabled = true;
    showLoading(true, '正在保存分割结果');
    try {
        const res = await fetch(`${API}/cellpose_batch/commit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: activeCellposeBatchJobId }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const job = await res.json();
        const total = job.total || 0;
        cellposeBatchPhase = job.status || 'saved';
        _setCellposeBatchProgress({
            total,
            processed: job.processed || total,
            current: job.status === 'saved' ? '已全部保存' : '保存失败',
            saved: job.committed || job.saved || 0,
            skipped: job.skipped || 0,
            failed: job.failed || 0,
            status: job.status || 'saved',
        });
        if (els.cellposeBatchStatus) {
            els.cellposeBatchStatus.textContent = job.message || '已确认保存';
        }
        if (job.status === 'saved') {
            activeCellposeBatchJobId = null;
            cellposeBatchPhase = 'idle';
            _clearCellposePreview();
            updateCellposeBatchRunState();
            await reloadImagesKeepSelection();
            showStatus(job.message || '已保存全部分割结果');
        } else {
            cellposeBatchPhase = job.status || 'failed';
            _updateCellposeBatchActionButtons();
            showStatus(job.message || '保存失败', true);
        }
    } catch (e) {
        showStatus(`确认保存失败: ${e.message}`, true);
        _updateCellposeBatchActionButtons();
    } finally {
        showLoading(false);
    }
}

async function discardCellposeBatchSave() {
    if (!activeCellposeBatchJobId) return;
    showLoading(true, '正在丢弃未保存结果');
    try {
        const res = await fetch(`${API}/cellpose_batch/discard`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: activeCellposeBatchJobId }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const job = await res.json();
        const total = job.total || 0;
        activeCellposeBatchJobId = null;
        cellposeBatchPhase = 'idle';
        _setCellposeBatchProgress({
            total,
            processed: job.processed || 0,
            current: '已不保存',
            saved: 0,
            skipped: job.skipped || 0,
            failed: job.failed || 0,
            status: 'idle',
        });
        updateCellposeBatchRunState();
        _clearCellposePreview();
        showStatus(job.message || '已放弃未保存的分割结果');
    } catch (e) {
        showStatus(`不保存失败: ${e.message}`, true);
    } finally {
        showLoading(false);
    }
}

function acceptCellposePreview() {
    if (!cellposePreviewData || cellposePreviewData.length === 0) return;
    pushUndoState();
    let saved = 0;
    for (const ann of cellposePreviewData) {
        if (ann.points && ann.points.length >= 6) {
            createAnnotationShape(ann.class_id, 'polygon', ann.points);
            saved++;
        }
    }
    setDirty(true);
    scheduleUIUpdate();
    showStatus(`✓ 已保存 ${saved} 个 CellposeSAM 标注`);
    _clearCellposePreview();
}

function rejectCellposePreview() {
    _clearCellposePreview();
    showStatus('已放弃 CellposeSAM 预测结果');
}

function _clearCellposePreview() {
    cellposePreviewData = null;
    cellposeOverlayLayer.destroyChildren();
    cellposeOverlayLayer.batchDraw();
    if (els.cellposeAcceptActions) els.cellposeAcceptActions.style.display = 'none';
    if (els.cellposeStatus && cellposeBatchPhase !== 'awaiting_save') {
        els.cellposeStatus.textContent = '';
    }
}

function clearCellposeOverlay() {
    _clearCellposePreview();
    showStatus('已清除 CellposeSAM 预览');
}

let _batchPreviewToken = 0;
async function _loadBatchPreviewForCurrentImage() {
    if (!activeCellposeBatchJobId || cellposeBatchPhase !== 'awaiting_save') return;
    if (state.currentImageIndex < 0 || !state.images[state.currentImageIndex]) return;
    const filename = state.images[state.currentImageIndex].filename;
    const token = ++_batchPreviewToken;
    try {
        const res = await fetch(
            `${API}/cellpose_batch/preview?job_id=${encodeURIComponent(activeCellposeBatchJobId)}&filename=${encodeURIComponent(filename)}`
        );
        if (!res.ok || token !== _batchPreviewToken) return;
        const data = await res.json();
        if (token !== _batchPreviewToken) return;
        if (data.found && data.annotations && data.annotations.length > 0) {
            cellposePreviewData = data.annotations;
            _drawCellposeOverlay(data.annotations);
            if (els.cellposeStatus) {
                els.cellposeStatus.textContent = `预览: ${filename} — ${data.count} 个细胞 (批量结果，未保存)`;
            }
        } else {
            cellposeOverlayLayer.destroyChildren();
            cellposeOverlayLayer.batchDraw();
            if (els.cellposeStatus) {
                els.cellposeStatus.textContent = data.found === false
                    ? `${filename}: 无待保存结果（可能被跳过）`
                    : `${filename}: 未检测到细胞`;
            }
        }
    } catch (e) {
        if (token === _batchPreviewToken) console.warn('Batch preview load error:', e);
    }
}

function clearAllAnnotations() {
    const count = annotLayer.getChildren().filter(g => g.name() === 'annotation').length;
    if (count === 0) return;
    const doIt = () => {
        pushUndoState();
        annotLayer.destroyChildren();
        boxLayer.destroyChildren();
        state.annotationCounter = 0;
        clearSelection();
        setDirty(true);
        scheduleUIUpdate();
        showStatus(`已清空 ${count} 条标注`);
    };
    if (typeof _dmConfirm === 'function') {
        _dmConfirm('清空标注', `确定清空当前 ${count} 条标注？（可撤销）`, doIt);
    } else {
        if (confirm(`确定清空当前 ${count} 条标注？（可撤销）`)) doIt();
    }
}


// ══════════════════════════════════════════════════════════════
// Directory search
// ══════════════════════════════════════════════════════════════

async function _searchLabelDirs(query) {
    if (!state.currentGroup) return;
    const container = els.labelsetSearchResults;
    if (!container) return;
    try {
        const gid = encodeURIComponent(state.currentGroup.group_id);
        const q = encodeURIComponent(query);
        const res = await fetch(`${API}/scan_label_dirs?group_id=${gid}&query=${q}`);
        const dirs = await res.json();
        container.innerHTML = '';
        if (dirs.length === 0) {
            container.innerHTML = '<div style="font-size:11px; color:var(--text-2); padding:4px;">未找到匹配的标注目录</div>';
            return;
        }
        for (const d of dirs) {
            const item = document.createElement('div');
            item.className = 'dir-result-item';
            const tagClass = d.already_loaded ? 'dir-tag' : 'dir-tag new';
            const tagText = d.already_loaded ? '已加载' : '可添加';
            const info = `${d.n_train}t/${d.n_val}v`;
            item.innerHTML = `<span class="dir-path" title="${d.train_dir}">${d.dir_name}</span>
                <span style="font-size:10px; color:var(--text-2);">${info}</span>
                <span class="${tagClass}">${tagText}</span>`;
            if (d.already_loaded) {
                item.onclick = async () => {
                    const ok = await selectLabelSet(d.set_id);
                    if (!ok) return;
                    await loadImages();
                    els.labelsetSearchPanel.style.display = 'none';
                };
            } else {
                item.onclick = () => _addLabelDir(d.dir_name, d.set_id);
            }
            container.appendChild(item);
        }
    } catch (e) {
        container.innerHTML = `<div style="font-size:11px; color:var(--danger); padding:4px;">搜索失败: ${e.message}</div>`;
    }
}

async function _addLabelDir(dirName, setId) {
    try {
        const res = await fetch(`${API}/add_label_set`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                group_id: state.currentGroup.group_id,
                dir_name: dirName,
            }),
        });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
        const data = await res.json();
        await loadGroups();
        await selectGroup(state.currentGroup.group_id);
        await selectLabelSet(data.set_id);
        await loadImages();
        if (els.labelsetSearchPanel) els.labelsetSearchPanel.style.display = 'none';
        showStatus(`✓ 已加载标注集: ${data.set_name}`);
    } catch (e) {
        showStatus(`加载标注集失败: ${e.message}`, true);
    }
}

async function _searchImageDirs(query) {
    const container = els.groupSearchResults;
    if (!container) return;
    try {
        const q = encodeURIComponent(query);
        const res = await fetch(`${API}/scan_image_dirs?query=${q}`);
        const dirs = await res.json();
        container.innerHTML = '';
        if (dirs.length === 0) {
            container.innerHTML = '<div style="font-size:11px; color:var(--text-2); padding:4px;">未找到匹配的图片目录</div>';
            return;
        }
        for (const d of dirs) {
            const item = document.createElement('div');
            item.className = 'dir-result-item';
            item.style.cursor = 'pointer';
            const tagClass = d.already_loaded ? 'dir-tag' : 'dir-tag';
            const tagText = d.already_loaded ? '已加载' : `${d.img_count}张`;
            item.innerHTML = `<span class="dir-path" title="${d.path}">${d.rel_path}</span>
                <span class="${tagClass}">${tagText}</span>`;
            item.onclick = async () => {
                if (d.already_loaded) {
                    const ok = await selectGroup(d.path);
                    if (!ok) return;
                    await loadImages();
                    els.groupSearchPanel.style.display = 'none';
                } else {
                    try {
                        const res = await fetch(`${API}/load_image_dir`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ path: d.path }),
                        });
                        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || '加载失败');
                        const data = await res.json();
                        await loadGroups();
                        const ok = await selectGroup(data.group_id);
                        if (!ok) return;
                        await loadImages();
                        els.groupSearchPanel.style.display = 'none';
                        showStatus(`✓ 已加载图片目录`);
                    } catch (e) {
                        showStatus(`加载失败: ${e.message}`, true);
                    }
                }
            };
            container.appendChild(item);
        }
    } catch (e) {
        container.innerHTML = `<div style="font-size:11px; color:var(--text-2); padding:4px;">搜索失败: ${e.message}</div>`;
    }
}


// ══════════════════════════════════════════════════════════════
// Shortcut Help Panel
// ══════════════════════════════════════════════════════════════

function toggleShortcutHelp() {
    const overlay = document.getElementById('shortcut-help-overlay');
    if (!overlay) return;
    overlay.classList.toggle('show');
}


// ══════════════════════════════════════════════════════════════
// Image Flags
// ══════════════════════════════════════════════════════════════

async function _loadFlaggedImages() {
    state._flaggedImages = new Set();
    if (!state.currentGroup || !state.currentLabelSet) return;
    try {
        const resp = await fetch(`${API}/image_flags?group_id=${encodeURIComponent(state.currentGroup.group_id)}&label_set_id=${encodeURIComponent(state.currentLabelSet.set_id)}&subset=${state.subset}`);
        const flags = await resp.json();
        flags.forEach(f => state._flaggedImages.add(f.filename));
    } catch {}
}

async function _jumpToFlaggedImage(direction) {
    if (!state.currentGroup || !state.currentLabelSet) return;
    try {
        const resp = await fetch(`${API}/image_flags?group_id=${encodeURIComponent(state.currentGroup.group_id)}&label_set_id=${encodeURIComponent(state.currentLabelSet.set_id)}&subset=${state.subset}`);
        const flags = await resp.json();
        const flaggedNames = new Set(flags.map(f => f.filename));
        if (flaggedNames.size === 0) { showStatus('没有已标记的图像'); return; }
        const start = state.currentImageIndex + direction;
        const end = direction > 0 ? state.images.length : -1;
        for (let i = start; i !== end; i += direction) {
            if (flaggedNames.has(state.images[i]?.filename)) { selectImage(i); return; }
        }
        showStatus('没有更多已标记的图像');
    } catch { showStatus('获取标记失败'); }
}

async function toggleImageFlag(flagType) {
    if (!state.currentGroup || !state.currentLabelSet || state.currentImageIndex < 0) return;
    const filename = state.images[state.currentImageIndex]?.filename;
    if (!filename) return;

    try {
        const existing = await fetch(`${API}/image_flags?group_id=${encodeURIComponent(state.currentGroup.group_id)}&label_set_id=${encodeURIComponent(state.currentLabelSet.set_id)}&subset=${state.subset}&filename=${encodeURIComponent(filename)}`);
        const flags = await existing.json();
        const hasFlag = flags.some(f => f.flag_type === flagType);

        if (hasFlag) {
            await fetch(`${API}/image_flags?group_id=${encodeURIComponent(state.currentGroup.group_id)}&label_set_id=${encodeURIComponent(state.currentLabelSet.set_id)}&subset=${state.subset}&filename=${encodeURIComponent(filename)}&flag_type=${flagType}`, { method: 'DELETE' });
            state._flaggedImages.delete(filename);
            renderImageList(true);
            showStatus('已取消标记');
        } else {
            await fetch(`${API}/image_flags`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    group_id: state.currentGroup.group_id,
                    label_set_id: state.currentLabelSet.set_id,
                    subset: state.subset,
                    filename: filename,
                    flag_type: flagType,
                    flag_value: 'done',
                }),
            });
            state._flaggedImages.add(filename);
            renderImageList(true);
            showStatus('已标记为完成 ✓');
        }
    } catch (e) {
        showStatus('标记失败', true);
    }
}


// ══════════════════════════════════════════════════════════════
// Session Persistence
// ══════════════════════════════════════════════════════════════

let _sessionId = null;

function _getSessionId() {
    if (_sessionId) return _sessionId;
    _sessionId = localStorage.getItem('balf_session_id');
    if (!_sessionId) {
        _sessionId = 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
        localStorage.setItem('balf_session_id', _sessionId);
    }
    return _sessionId;
}

async function saveSessionState() {
    if (!state.currentGroup) return;
    try {
        await fetch(`${API}/session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: _getSessionId(),
                group_id: state.currentGroup?.group_id || '',
                label_set_id: state.currentLabelSet?.set_id || '',
                subset: state.subset,
                current_image: state.images[state.currentImageIndex]?.filename || '',
                current_image_index: state.currentImageIndex,
            }),
        });
    } catch (e) { /* silent */ }
}

async function restoreSession() {
    try {
        const res = await fetch(`${API}/session/${_getSessionId()}`);
        if (!res.ok) return false;
        const session = await res.json();
        if (!session.group_id) return false;

        const groupIdx = state.groups.findIndex(g => g.group_id === session.group_id);
        if (groupIdx < 0) return false;

        els.groupSelect.value = session.group_id;
        state.currentGroup = state.groups[groupIdx];
        await loadClasses();

        if (session.label_set_id) {
            const ls = state.currentGroup.label_sets?.find(l => l.set_id === session.label_set_id);
            if (ls) {
                state.currentLabelSet = ls;
                els.labelsetSelect.value = session.label_set_id;
            }
        }
        if (session.subset) {
            state.subset = session.subset;
            els.subsetSelect.value = session.subset;
        }

        await loadImages();

        if (session.current_image_index >= 0 && session.current_image_index < state.images.length) {
            await selectImage(session.current_image_index);
        }
        showStatus('已恢复上次会话');
        return true;
    } catch (e) {
        return false;
    }
}

// Save session periodically
setInterval(saveSessionState, 30000);


// ══════════════════════════════════════════════════════════════
// Boot
// ══════════════════════════════════════════════════════════════

const _originalInit = typeof init === 'function' ? init : null;

window.onload = async function() {
    if (_originalInit) await _originalInit();
    // Try to restore session after initial load
    setTimeout(async () => {
        if (state.currentImageIndex < 0 && state.groups.length > 0) {
            await restoreSession();
        }
    }, 500);
};
