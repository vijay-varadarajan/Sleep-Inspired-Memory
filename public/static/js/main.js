// Global state
let currentTab = 'episodic';
let sleepCycleInterval = null;
let isResizing = false;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeChat();
    initializeTabs();
    initializeSleepMonitor();
    loadMemoryData();
    setupSleepTrigger();
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
        const memories = await response.json();
        
        const container = document.getElementById('episodicMemoryList');
        container.innerHTML = '';
        
        if (memories.length === 0) {
            container.innerHTML = '<div class="loading">No episodic memories yet</div>';
            return;
        }
        
        memories.forEach(memory => {
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
        const memories = await response.json();
        
        const container = document.getElementById('consolidatedMemoryList');
        container.innerHTML = '';
        
        if (memories.length === 0) {
            container.innerHTML = '<div class="loading">No consolidated memories yet</div>';
            return;
        }
        
        memories.forEach(memory => {
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
        const schemas = await response.json();
        
        const container = document.getElementById('schemaMemoryList');
        container.innerHTML = '';
        
        if (schemas.length === 0) {
            container.innerHTML = '<div class="loading">No schema memories yet</div>';
            return;
        }
        
        schemas.forEach(schema => {
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
    item.className = 'memory-item';
    
    item.innerHTML = `
        <div class="memory-item-header">
            <span class="memory-item-title">Episode</span>
            <span class="memory-item-meta">${memory.timestamp}</span>
        </div>
        <div class="memory-item-content">${memory.content}</div>
        <div class="importance-bar">
            <div class="importance-fill" style="width: ${memory.importance * 100}%"></div>
        </div>
    `;
    
    return item;
}

function createConsolidatedMemoryItem(memory) {
    const item = document.createElement('div');
    item.className = 'memory-item';
    
    const connectionsHTML = memory.connections
        .map(conn => `<span class="connection-tag">${conn}</span>`)
        .join('');
    
    item.innerHTML = `
        <div class="memory-item-header">
            <span class="memory-item-title">${memory.concept}</span>
            <span class="memory-item-meta">Strength: ${(memory.strength * 100).toFixed(0)}%</span>
        </div>
        <div class="connections">${connectionsHTML}</div>
        <div class="importance-bar">
            <div class="importance-fill" style="width: ${memory.strength * 100}%"></div>
        </div>
    `;
    
    return item;
}

function createSchemaMemoryItem(schema) {
    const item = document.createElement('div');
    item.className = 'memory-item';
    
    const structureHTML = JSON.stringify(schema.structure, null, 2);
    
    item.innerHTML = `
        <div class="memory-item-header">
            <span class="memory-item-title">${schema.schema}</span>
        </div>
        <div class="schema-structure">${structureHTML}</div>
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
        
        // Update stage indicators
        document.querySelectorAll('.stage-item').forEach(item => {
            item.classList.toggle('active', item.dataset.stage === status.stage);
        });
        
        // Simulate progress if active (this would be real in production)
        if (status.active && status.progress < 100) {
            simulateSleepProgress(status.stage);
        }
    } catch (error) {
        console.error('Error updating sleep status:', error);
    }
}

function simulateSleepProgress(currentStage) {
    // This is just for demo - in production, the backend would handle this
    const stages = ['compression', 'consolidation', 'replay', 'decay'];
    const currentIndex = stages.indexOf(currentStage);
    
    if (currentIndex < stages.length - 1) {
        // Simulate moving to next stage after a delay
        setTimeout(() => {
            // This would be handled by the backend in production
        }, 3000);
    }
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

// ============ UTILITY FUNCTIONS ============
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
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
});
