// Global state
let currentTab = 'episodic';
let sleepCycleInterval = null;
let processLogInterval = null;
let isResizing = false;
let lastLogCount = 0;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeChat();
    initializeTabs();
    initializeSleepMonitor();
    initializeProcessLogMonitor();
    loadMemoryData();
    setupSleepTrigger();
    setupResetButton();
    initializeResizer();
});

// ============ CHAT FUNCTIONALITY ============
function initializeChat() {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Clear input
        chatInput.value = '';
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Send to backend
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Add AI response
                addMessageToChat('assistant', data.response, data.timestamp);
                
                // Immediately refresh episodic memory
                loadEpisodicMemory();
            }
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToChat('assistant', 'Sorry, there was an error processing your message.', null, true);
        }
    });
}

function addMessageToChat(role, content, timestamp = null, isError = false) {
    const chatMessages = document.getElementById('chatMessages');
    
    // Remove welcome message if it exists
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const roleSpan = document.createElement('div');
    roleSpan.className = 'message-role';
    roleSpan.textContent = role === 'user' ? 'You' : 'AI Agent';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = content;
    
    contentDiv.appendChild(roleSpan);
    contentDiv.appendChild(textDiv);
    
    if (timestamp || !isError) {
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
        contentDiv.appendChild(timestampDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============ TABS FUNCTIONALITY ============
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    currentTab = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-content`);
    });
}

// ============ MEMORY DATA LOADING ============
async function loadMemoryData() {
    await Promise.all([
        loadEpisodicMemory(),
        loadConsolidatedMemory(),
        loadSchemaMemory(),
    ]);
}

async function loadEpisodicMemory() {
    try {
        const response = await fetch('/api/memory/episodic');
        const data = await response.json();
        
        const container = document.getElementById('episodicMemoryList');
        container.innerHTML = '';
        
        if (!data.memories || data.memories.length === 0) {
            container.innerHTML = '<div class="loading">No episodic memories yet</div>';
            return;
        }
        
        // Add header with total stats
        const header = document.createElement('div');
        header.className = 'memory-stats-header';
        header.innerHTML = `
            <span class="stat-item">Total: ${data.count}</span>
            <span class="stat-item">Size: ${data.total_size_formatted}</span>
        `;
        container.appendChild(header);
        
        data.memories.forEach(memory => {
            const item = createEpisodicMemoryItem(memory);
            container.appendChild(item);
        });
    } catch (error) {
        console.error('Error loading episodic memory:', error);
    }
}

async function loadConsolidatedMemory() {
    try {
        const response = await fetch('/api/memory/consolidated');
        const data = await response.json();
        
        const container = document.getElementById('consolidatedMemoryList');
        container.innerHTML = '';
        
        if (!data.memories || data.memories.length === 0) {
            container.innerHTML = '<div class="loading">No consolidated memories yet</div>';
            return;
        }
        
        // Add header with total stats
        const header = document.createElement('div');
        header.className = 'memory-stats-header';
        header.innerHTML = `
            <span class="stat-item">Total: ${data.count}</span>
            <span class="stat-item">Size: ${data.total_size_formatted}</span>
        `;
        container.appendChild(header);
        
        data.memories.forEach(memory => {
            const item = createConsolidatedMemoryItem(memory);
            container.appendChild(item);
        });
    } catch (error) {
        console.error('Error loading consolidated memory:', error);
    }
}

async function loadSchemaMemory() {
    try {
        const response = await fetch('/api/memory/schema');
        const data = await response.json();
        
        const container = document.getElementById('schemaMemoryList');
        container.innerHTML = '';
        
        if (!data.memories || data.memories.length === 0) {
            container.innerHTML = '<div class="loading">No schema memories yet</div>';
            return;
        }
        
        // Add header with total stats
        const header = document.createElement('div');
        header.className = 'memory-stats-header';
        header.innerHTML = `
            <span class="stat-item">Total: ${data.count}</span>
            <span class="stat-item">Size: ${data.total_size_formatted}</span>
        `;
        container.appendChild(header);
        
        data.memories.forEach(schema => {
            const item = createSchemaMemoryItem(schema);
            container.appendChild(item);
        });
    } catch (error) {
        console.error('Error loading schema memory:', error);
    }
}

// ============ MEMORY ITEM CREATORS ============
function createEpisodicMemoryItem(memory) {
    const item = document.createElement('div');
    item.className = 'memory-item collapsible-item';
    item.dataset.expanded = 'false';
    
    const consolidatedBadge = memory.consolidated 
        ? '<span class="consolidated-badge">✓ Consolidated</span>' 
        : '';
    
    const sizeFormatted = memory.size ? formatSizeJS(memory.size) : '0 B';
    
    // Parse content to extract user input and agent response
    const userInputMatch = memory.content.match(/^User: (.+?)(?:\nAgent:|$)/s);
    const agentResponseMatch = memory.content.match(/Agent: (.+)$/s);
    const userInput = userInputMatch ? userInputMatch[1].trim() : memory.content;
    const agentResponse = agentResponseMatch ? agentResponseMatch[1].trim() : '';
    
    item.innerHTML = `
        <div class="collapsible-header">
            <span class="expand-arrow">▶</span>
            <div class="collapsed-content">
                <strong>User:</strong> ${userInput}
            </div>
        </div>
        <div class="expanded-content" style="display: none;">
            <div class="memory-item-header">
                <span class="memory-item-title">Episode ${consolidatedBadge}</span>
                <span class="memory-item-meta">${memory.timestamp}</span>
            </div>
            <div class="dialogue-section">
                <div class="user-input"><strong>User:</strong> ${userInput}</div>
                <div class="agent-response"><strong>Agent:</strong> ${agentResponse}</div>
            </div>
            <div class="memory-item-stats">
                <span class="stat-badge">Size: ${sizeFormatted}</span>
                <span class="stat-badge">Importance: ${(memory.importance * 100).toFixed(0)}%</span>
                <span class="stat-badge">Novelty: ${(memory.novelty * 100).toFixed(0)}%</span>
                <span class="stat-badge">Accessed: ${memory.access_count}x</span>
            </div>
            <div class="importance-bar">
                <div class="importance-fill" style="width: ${memory.importance * 100}%"></div>
            </div>
        </div>
    `;
    
    // Add click handler to toggle expand/collapse
    const header = item.querySelector('.collapsible-header');
    const arrow = item.querySelector('.expand-arrow');
    const expandedContent = item.querySelector('.expanded-content');
    
    header.addEventListener('click', () => {
        const isExpanded = item.dataset.expanded === 'true';
        item.dataset.expanded = !isExpanded ? 'true' : 'false';
        expandedContent.style.display = isExpanded ? 'none' : 'block';
        arrow.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(90deg)';
    });
    
    return item;
}

function createConsolidatedMemoryItem(memory) {
    const item = document.createElement('div');
    item.className = 'memory-item';
    
    const conceptsHTML = memory.key_concepts
        .map(concept => `<span class="connection-tag">${concept}</span>`)
        .join('');
    
    const sizeFormatted = memory.size ? formatSizeJS(memory.size) : '0 B';
    
    item.innerHTML = `
        <div class="memory-item-header">
            <span class="memory-item-title">Consolidated Memory</span>
            <span class="memory-item-meta">${memory.timestamp}</span>
        </div>
        <div class="memory-item-content">${memory.summary}</div>
        <div class="connections">${conceptsHTML}</div>
        <div class="memory-item-stats">
            <span class="stat-badge">Size: ${sizeFormatted}</span>
            <span class="stat-badge">Sources: ${memory.source_count} episodes</span>
            <span class="stat-badge">Strength: ${(memory.importance * 100).toFixed(0)}%</span>
        </div>
        <div class="importance-bar">
            <div class="importance-fill" style="width: ${memory.importance * 100}%"></div>
        </div>
    `;
    
    return item;
}

function createSchemaMemoryItem(schema) {
    const item = document.createElement('div');
    item.className = 'memory-item';
    
    const conceptsHTML = schema.core_concepts
        .map(concept => `<span class="connection-tag">${concept}</span>`)
        .join('');
    
    const examplesHTML = schema.examples && schema.examples.length > 0
        ? `<div class="schema-examples">${schema.examples.slice(0, 2).join('; ')}</div>`
        : '';
    
    const sizeFormatted = schema.size ? formatSizeJS(schema.size) : '0 B';
    
    item.innerHTML = `
        <div class="memory-item-header">
            <span class="memory-item-title">${schema.name}</span>
            <span class="memory-item-meta">${schema.timestamp}</span>
        </div>
        <div class="memory-item-content">${schema.description}</div>
        <div class="connections">${conceptsHTML}</div>
        ${examplesHTML}
        <div class="memory-item-stats">
            <span class="stat-badge">Size: ${sizeFormatted}</span>
            <span class="stat-badge">Memories: ${schema.memory_count}</span>
            <span class="stat-badge">Confidence: ${(schema.confidence * 100).toFixed(0)}%</span>
        </div>
    `;
    
    return item;
}

// ============ SLEEP CYCLE MONITORING ============
function initializeSleepMonitor() {
    // Poll sleep status every 2 seconds
    sleepCycleInterval = setInterval(updateSleepStatus, 2000);
    
    // Initial update
    updateSleepStatus();
}

async function updateSleepStatus() {
    try {
        const response = await fetch('/api/sleep/status');
        const status = await response.json();
        
        // Update status display
        document.getElementById('sleepStage').textContent = 
            status.stage.charAt(0).toUpperCase() + status.stage.slice(1);
        document.getElementById('sleepDetails').textContent = status.details;
        document.getElementById('progressFill').style.width = `${status.progress}%`;
        
        // Update episode count
        if (status.episodes_since_sleep !== undefined) {
            document.getElementById('episodeCount').textContent = 
                `${status.episodes_since_sleep} / ${status.total_episodes || 0}`;
        }
        
        // Update stage indicators
        document.querySelectorAll('.stage-item').forEach(item => {
            item.classList.toggle('active', item.dataset.stage === status.stage);
        });
        
        // Refresh memory displays after sleep completes
        if (!status.active && status.progress === 100) {
            await loadMemoryData();
        }
    } catch (error) {
        console.error('Error updating sleep status:', error);
    }
}

// ============ PROCESS LOG MONITORING ============
function initializeProcessLogMonitor() {
    // Poll process log every 1 second
    processLogInterval = setInterval(updateProcessLog, 1000);
    
    // Initial update
    updateProcessLog();
}

async function updateProcessLog() {
    try {
        const response = await fetch('/api/process/log');
        const logs = await response.json();
        
        if (logs.length === 0) return;
        
        // Update badge count
        const badge = document.getElementById('logBadge');
        badge.textContent = logs.length;
        
        // Only update if new logs arrived
        if (logs.length === lastLogCount) return;
        lastLogCount = logs.length;
        
        const logContainer = document.getElementById('processLog');
        
        // Clear and rebuild (or append new ones)
        if (logs.length > 0) {
            // Clear initial message
            if (logContainer.querySelector('.log-init')) {
                logContainer.innerHTML = '';
            }
            
            // Add new logs
            const newLogs = logs.slice(-20); // Show last 20
            newLogs.forEach(log => {
                // Check if log already exists
                const existingLog = logContainer.querySelector(`[data-log-time="${log.timestamp}"]`);
                if (existingLog) return;
                
                const logEntry = createLogEntry(log);
                logContainer.appendChild(logEntry);
            });
            
            // Auto-scroll to bottom
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Refresh memory data after certain log types
            if (logs[logs.length - 1].step.includes('Complete') || 
                logs[logs.length - 1].step.includes('Stored')) {
                loadMemoryData();
            }
        }
    } catch (error) {
        console.error('Error updating process log:', error);
    }
}

function createLogEntry(log) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.setAttribute('data-log-time', log.timestamp);
    
    // Add class based on step type
    if (log.step.includes('Error') || log.step.includes('❌')) {
        entry.classList.add('log-error');
    } else if (log.step.includes('Complete') || log.step.includes('✅')) {
        entry.classList.add('log-success');
    } else if (log.step.includes('Warning') || log.step.includes('⚠️')) {
        entry.classList.add('log-warning');
    }
    
    const time = new Date(log.timestamp).toLocaleTimeString();
    const icon = log.step.match(/[^\w\s]/)?.[0] || 'ℹ️';
    
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-icon">${icon}</span>
        <span class="log-message">${log.step}: ${log.details}</span>
    `;
    
    return entry;
}

function setupSleepTrigger() {
    const triggerBtn = document.getElementById('triggerSleepBtn');
    
    triggerBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/sleep/trigger', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Show feedback
                triggerBtn.textContent = 'Sleep Cycle Active...';
                triggerBtn.disabled = true;
                
                // Re-enable after 10 seconds
                setTimeout(() => {
                    triggerBtn.textContent = 'Trigger Sleep Cycle';
                    triggerBtn.disabled = false;
                }, 10000);
                
                // Refresh status immediately
                updateSleepStatus();
            }
        } catch (error) {
            console.error('Error triggering sleep cycle:', error);
        }
    });
}

function setupResetButton() {
    const resetBtn = document.getElementById('resetMemoryBtn');
    
    resetBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to reset all memory? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch('/api/memory/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Show feedback
                resetBtn.textContent = 'Memory Reset!';
                setTimeout(() => {
                    resetBtn.textContent = 'Reset Memory';
                }, 2000);
                
                // Reload all memory displays
                loadEpisodicMemory();
                loadConsolidatedMemory();
                loadSchemaMemory();
                updateSleepStatus();
                
                // Clear chat messages except welcome
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.innerHTML = `
                    <div class="welcome-message">
                        <h2>Welcome!</h2>
                        <p>Start a conversation to see the memory system in action.</p>
                    </div>
                `;
            } else {
                alert('Error resetting memory: ' + data.error);
            }
        } catch (error) {
            console.error('Error resetting memory:', error);
            alert('Error resetting memory: ' + error.message);
        }
    });
}
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function formatSizeJS(bytes) {
    if (bytes < 1024) {
        return `${bytes} B`;
    } else if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(2)} KB`;
    } else {
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
}

// ============ RESIZABLE PANELS ============
function initializeResizer() {
    const divider = document.getElementById('divider');
    const chatPanel = document.getElementById('chatPanel');
    const memoryPanel = document.getElementById('memoryPanel');
    const container = document.querySelector('.container');
    
    divider.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        e.preventDefault();
        
        const containerRect = container.getBoundingClientRect();
        const offsetX = e.clientX - containerRect.left;
        const containerWidth = containerRect.width;
        const dividerWidth = 4; // Width of divider in pixels
        
        // Calculate width in pixels (with limits)
        let leftWidth = offsetX;
        const minWidth = containerWidth * 0.3;
        const maxWidth = containerWidth * 0.7;
        
        leftWidth = Math.max(minWidth, Math.min(maxWidth, leftWidth));
        
        // Apply width - only set chat panel width, let memory panel flex naturally
        chatPanel.style.flex = `0 0 ${leftWidth}px`;
        memoryPanel.style.flex = '1 1 auto';
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (sleepCycleInterval) {
        clearInterval(sleepCycleInterval);
    }
    if (processLogInterval) {
        clearInterval(processLogInterval);
    }
});
