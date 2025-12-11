// BioLaySumm RAG Demo - with Multi-Summary Comparison

const API_BASE = '/api';
let selectedStyle = 'plain';
let currentRAGProcess = null;
let currentMainText = '';
let summaries = {}; // Store summaries by style {formal: {...}, plain: {...}, ...}
let currentArticle = ''; // Track current article to detect changes

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initStyleSelector();
    initGenerateButton();
    initClearButton();
    initPinButton();
    initResizeHandle();
});

// Style selector
function initStyleSelector() {
    const styleBtns = document.querySelectorAll('.style-btn');
    styleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            styleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedStyle = btn.dataset.style;
        });
    });
}

// Generate button
function initGenerateButton() {
    const generateBtn = document.getElementById('generateBtn');
    const titleInput = document.getElementById('title');
    const articleInput = document.getElementById('article');
    
    generateBtn.addEventListener('click', async () => {
        const title = titleInput.value.trim();
        const article = articleInput.value.trim();
        
        if (!article) {
            showError('Please paste an article first');
            return;
        }
        
        // Check if article changed - if yes, clear previous summaries
        if (article !== currentArticle) {
            currentArticle = article;
            summaries = {};
        }
        
        await generateSummary(title, article, selectedStyle);
    });
}

// Clear button
function initClearButton() {
    const clearBtn = document.getElementById('clearBtn');
    clearBtn.addEventListener('click', () => {
        if (confirm('Clear all generated summaries?')) {
            summaries = {};
            currentArticle = '';
            updateSummariesDisplay();
            document.getElementById('ragCard').style.display = 'none';
        }
    });
}

// Pin button
function initPinButton() {
    const pinBtn = document.getElementById('pinBtn');
    const summaryCard = document.getElementById('summaryCard');
    const resizeHandle = document.getElementById('resizeHandle');
    let isPinned = false;
    
    pinBtn.addEventListener('click', () => {
        isPinned = !isPinned;
        
        if (isPinned) {
            summaryCard.classList.add('pinned');
            pinBtn.classList.add('pinned');
            pinBtn.querySelector('.pin-text').textContent = 'Unpin';
            pinBtn.title = 'Click to unpin (allow scrolling)';
            resizeHandle.style.display = 'flex';
            
            // Set default height
            if (!summaryCard.style.maxHeight) {
                summaryCard.style.maxHeight = '50vh';
            }
        } else {
            summaryCard.classList.remove('pinned');
            pinBtn.classList.remove('pinned');
            pinBtn.querySelector('.pin-text').textContent = 'Pin';
            pinBtn.title = 'Click to pin (keep visible while scrolling)';
            resizeHandle.style.display = 'none';
            summaryCard.style.maxHeight = '';
        }
    });
}

// Resize handle - drag to adjust height
function initResizeHandle() {
    const resizeHandle = document.getElementById('resizeHandle');
    const summaryCard = document.getElementById('summaryCard');
    let isResizing = false;
    let startY = 0;
    let startHeight = 0;
    
    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startHeight = summaryCard.offsetHeight;
        
        // Change cursor globally
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        
        // Visual feedback
        resizeHandle.style.background = 'rgba(59, 130, 246, 0.3)';
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const deltaY = e.clientY - startY;
        const newHeight = startHeight + deltaY;
        
        // Constrain height between 200px and 80vh
        const minHeight = 200;
        const maxHeight = window.innerHeight * 0.8;
        const clampedHeight = Math.max(minHeight, Math.min(newHeight, maxHeight));
        
        summaryCard.style.maxHeight = clampedHeight + 'px';
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            resizeHandle.style.background = '';
        }
    });
}

// Generate summary
async function generateSummary(title, article, style) {
    const generateBtn = document.getElementById('generateBtn');
    const btnText = generateBtn.querySelector('.btn-text');
    const spinner = generateBtn.querySelector('.spinner');
    const error = document.getElementById('error');
    
    error.classList.add('hidden');
    
    // Loading state
    generateBtn.disabled = true;
    btnText.textContent = `⏳ Generating ${style}...`;
    spinner.classList.remove('hidden');
    
    try {
        const response = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: title || 'Untitled Article',
                article: article,
                style: style
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Generation failed');
        }
        
        const data = await response.json();
        
        // Store summary by style
        summaries[style] = {
            summary: data.summary,
            word_count: data.word_count,
            chunks_used: data.chunks_used,
            queries: data.queries,
            style: style
        };
        
        // Store RAG process info (shared across styles)
        currentMainText = data.main_text || '';
        currentRAGProcess = data.rag_process || null;
        
        // Update display
        updateSummariesDisplay();
        
        // Show RAG visualization if available
        if (currentRAGProcess) {
            displayRAGProcess(currentRAGProcess);
            document.getElementById('ragCard').style.display = 'block';
        }
        
    } catch (err) {
        showError(err.message);
    } finally {
        generateBtn.disabled = false;
        btnText.textContent = '🚀 Generate Summary';
        spinner.classList.add('hidden');
    }
}

// Update summaries display
function updateSummariesDisplay() {
    const placeholder = document.getElementById('placeholder');
    const container = document.getElementById('summariesContainer');
    const clearBtn = document.getElementById('clearBtn');
    const pinBtn = document.getElementById('pinBtn');
    
    const numSummaries = Object.keys(summaries).length;
    
    if (numSummaries === 0) {
        placeholder.classList.remove('hidden');
        container.classList.add('hidden');
        clearBtn.classList.add('hidden');
        pinBtn.classList.add('hidden');
        return;
    }
    
    placeholder.classList.add('hidden');
    container.classList.remove('hidden');
    clearBtn.classList.remove('hidden');
    pinBtn.classList.remove('hidden');
    
    // Set grid layout based on number of summaries
    container.className = 'summaries-container';
    if (numSummaries === 1) {
        container.classList.add('cols-1');
    } else if (numSummaries === 2) {
        container.classList.add('cols-2');
    } else {
        container.classList.add('cols-3');
    }
    
    // Render summaries
    const styleOrder = ['formal', 'plain', 'high_readability'];
    const styleNames = {
        'formal': 'Formal',
        'plain': 'Plain',
        'high_readability': 'Simple'
    };
    const styleIcons = {
        'formal': '📚',
        'plain': '📖',
        'high_readability': '✨'
    };
    
    let html = '';
    styleOrder.forEach(style => {
        if (!summaries[style]) return;
        
        const data = summaries[style];
        
        html += `
            <div class="summary-card">
                <div class="summary-header ${style}">
                    <div class="summary-style-badge">
                        <span>${styleIcons[style]}</span>
                        <span>${styleNames[style]}</span>
                    </div>
                    <button class="summary-remove" onclick="removeSummary('${style}')" title="Remove">
                        ×
                    </button>
                </div>
                <div class="summary-stats">
                    ${data.word_count} words • ${data.chunks_used} chunks • ${data.queries} queries
                </div>
                <div class="summary-content">
                    ${escapeHtml(data.summary)}
                </div>
                <button class="summary-copy" onclick="copySummary('${style}')">
                    📋 Copy ${styleNames[style]} Summary
                </button>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Remove single summary
function removeSummary(style) {
    delete summaries[style];
    updateSummariesDisplay();
    
    // Hide RAG card if no summaries
    if (Object.keys(summaries).length === 0) {
        document.getElementById('ragCard').style.display = 'none';
    }
}

// Copy single summary
function copySummary(style) {
    if (!summaries[style]) return;
    
    const text = summaries[style].summary;
    navigator.clipboard.writeText(text).then(() => {
        // Visual feedback
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.style.background = '#d1fae5';
        
        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
        }, 2000);
    });
}

// Display RAG process
function displayRAGProcess(ragProcess) {
    // Step 1: Query generation
    document.getElementById('queryInfo').innerHTML = `
        Generated <strong>${ragProcess.num_queries}</strong> queries from abstract
    `;
    
    // Step 2: Dense retrieval with deduplication info
    const dedupInfo = ragProcess.deduplication_info || {};
    const before = dedupInfo.before || ragProcess.total_retrievals || (ragProcess.num_queries * 3);
    const after = dedupInfo.after || ragProcess.unique_candidates || ragProcess.dense_candidates.length;
    const removed = dedupInfo.removed || (before - after);
    
    document.getElementById('denseInfo').innerHTML = `
        Retrieved <strong>${before}</strong> results 
        (${ragProcess.num_queries} queries × 3 chunks)<br>
        → Deduplicated to <strong>${after}</strong> unique chunks 
        <span style="color: #6b7280; font-size: 11px;">(removed ${removed} duplicates)</span>
    `;
    
    // Step 3: Reranking
    const avgRerankScore = ragProcess.reranked_results.length > 0
        ? (ragProcess.reranked_results.reduce((sum, r) => sum + r.rerank_score, 0) / ragProcess.reranked_results.length).toFixed(3)
        : 0;
    document.getElementById('rerankInfo').innerHTML = `
        Reranked ${after} candidates (avg score: <strong>${avgRerankScore}</strong>)
    `;
    
    // Step 4: Final selection
    document.getElementById('finalInfo').innerHTML = `
        Selected top <strong>${ragProcess.final_chunks.length}</strong> most relevant chunks
    `;
    
    // Display overlap analysis
    displayOverlapAnalysis(ragProcess);
    
    // Display detailed chunk info
    displayChunkDetails(ragProcess);
}

// Display overlap analysis
function displayOverlapAnalysis(ragProcess) {
    const chunks = ragProcess.reranked_results || [];
    
    // Count query overlaps
    const overlapCounts = {};
    chunks.forEach(chunk => {
        const count = chunk.query_count || 0;
        overlapCounts[count] = (overlapCounts[count] || 0) + 1;
    });
    
    let html = `
        <div style="margin-top: 16px; padding: 12px; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <div style="font-weight: 600; margin-bottom: 8px; color: #1e40af;">
                📊 Query Overlap Analysis
            </div>
            <div style="font-size: 13px; color: #374151;">
    `;
    
    // Show distribution
    const maxCount = Math.max(...chunks.map(c => c.query_count || 0));
    for (let i = maxCount; i >= 1; i--) {
        const count = overlapCounts[i] || 0;
        if (count > 0) {
            const plural = i === 1 ? 'query' : 'queries';
            html += `
                <div style="margin-bottom: 4px;">
                    <span style="display: inline-block; width: 120px;">
                        Selected by <strong>${i}</strong> ${plural}:
                    </span>
                    <span style="color: #667eea; font-weight: 600;">${count} chunks</span>
                </div>
            `;
        }
    }
    
    html += `
            </div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #bfdbfe; font-size: 12px; color: #6b7280;">
                💡 Chunks selected by more queries are considered more important
            </div>
        </div>
    `;
    
    // Insert after step 2
    const denseInfoDiv = document.getElementById('denseInfo').parentElement;
    const existingOverlap = denseInfoDiv.querySelector('.overlap-analysis');
    if (existingOverlap) {
        existingOverlap.remove();
    }
    
    const overlapDiv = document.createElement('div');
    overlapDiv.className = 'overlap-analysis';
    overlapDiv.innerHTML = html;
    denseInfoDiv.appendChild(overlapDiv);
}

// Display chunk details
function displayChunkDetails(ragProcess) {
    const container = document.getElementById('chunkDetails');
    
    let html = '<div style="margin-bottom: 16px; font-weight: 600; color: #374151;">Reranked Chunks (with Query Overlap):</div>';
    
    ragProcess.reranked_results.forEach((chunk, idx) => {
        const selectedClass = chunk.selected ? 'selected' : '';
        const selectedBadge = chunk.selected ? '<span style="color: #10b981; font-weight: 600;">✓ SELECTED</span>' : '';
        
        // Query overlap badge
        const queryCount = chunk.query_count || 0;
        const queryBadge = queryCount > 0 
            ? `<span style="background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">
                ${queryCount} ${queryCount === 1 ? 'query' : 'queries'}
               </span>`
            : '';
        
        // Queries that selected this chunk
        const queriesInfo = chunk.selected_by_queries && chunk.selected_by_queries.length > 0
            ? `<div style="margin-top: 6px; font-size: 11px; color: #6b7280;">
                Selected by queries: ${chunk.selected_by_queries.join(', ')}
               </div>`
            : '';
        
        html += `
            <div class="chunk-item ${selectedClass}">
                <div class="chunk-header">
                    <span>Rank ${chunk.rank} ${selectedBadge}</span>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        ${queryBadge}
                        <span style="color: #667eea;">Score: ${chunk.rerank_score.toFixed(3)}</span>
                    </div>
                </div>
                ${queriesInfo}
                <div class="chunk-text">
                    ${escapeHtml(chunk.chunk_text.substring(0, 150))}...
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Toggle highlighted text view
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('toggleTextBtn');
    const highlightedText = document.getElementById('highlightedText');
    
    toggleBtn.addEventListener('click', () => {
        if (highlightedText.classList.contains('hidden')) {
            highlightedText.classList.remove('hidden');
            toggleBtn.textContent = '🙈 Hide Original Text';
            displayHighlightedText();
        } else {
            highlightedText.classList.add('hidden');
            toggleBtn.textContent = '👁️ Show Original Text with Highlights';
        }
    });
});

// Display highlighted text
function displayHighlightedText() {
    if (!currentRAGProcess || !currentMainText) return;
    
    const textContent = document.getElementById('textContent');
    
    // Get all chunks with their categories
    const chunksToHighlight = [];
    
    // Collect all chunks with their types and query counts
    currentRAGProcess.reranked_results.forEach(chunk => {
        const chunkWords = chunk.chunk_text.split(/\s+/).slice(0, 15).join(' ');
        const pos = currentMainText.toLowerCase().indexOf(chunkWords.toLowerCase());
        if (pos !== -1) {
            chunksToHighlight.push({
                start: pos,
                end: pos + chunk.chunk_text.length,
                type: chunk.selected ? 'final' : 'candidate',
                text: chunk.chunk_text,
                rank: chunk.rank,
                score: chunk.rerank_score,
                query_count: chunk.query_count || 1,
                selected_by: chunk.selected_by_queries || []
            });
        }
    });
    
    // Sort by position
    chunksToHighlight.sort((a, b) => a.start - b.start);
    
    // Build highlighted HTML
    if (chunksToHighlight.length === 0) {
        textContent.innerHTML = `
            <p style="color: #6b7280; font-style: italic;">
                Unable to map chunks to original text positions.
            </p>
        `;
        return;
    }
    
    let html = '';
    let lastPos = 0;
    chunksToHighlight.forEach((chunk, idx) => {
        // Add text before chunk
        if (chunk.start > lastPos) {
            html += escapeHtml(currentMainText.substring(lastPos, chunk.start));
        }
        
        // Add highlighted chunk with intensity based on query count
        const intensity = chunk.query_count || 1;
        const highlightClass = chunk.type === 'final' ? 'highlight-final' : 'highlight-candidate';
        const intensityClass = `intensity-${Math.min(intensity, 5)}`;
        
        const queriesStr = chunk.selected_by.length > 0 ? chunk.selected_by.join(', ') : 'N/A';
        const title = `Rank ${chunk.rank} | Score: ${chunk.score.toFixed(3)} | ${chunk.type === 'final' ? 'SELECTED' : 'Candidate'} | Selected by ${intensity} ${intensity === 1 ? 'query' : 'queries'} (${queriesStr})`;
        
        html += `<mark class="${highlightClass} ${intensityClass}" title="${title}">${escapeHtml(chunk.text)}</mark>`;
        
        lastPos = chunk.end;
    });
    
    // Add remaining text
    if (lastPos < currentMainText.length) {
        const remaining = currentMainText.substring(lastPos, Math.min(lastPos + 1000, currentMainText.length));
        html += escapeHtml(remaining);
        if (lastPos + 1000 < currentMainText.length) {
            html += '<span style="color: #9ca3af;">... (text truncated)</span>';
        }
    }
    
    textContent.innerHTML = html;
}

// Show error
function showError(message) {
    const error = document.getElementById('error');
    error.classList.remove('hidden');
    error.textContent = `❌ ${message}`;
    
    setTimeout(() => {
        error.classList.add('hidden');
    }, 5000);
}

// Utility
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
