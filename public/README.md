# Sleep-Inspired Memory System - Web Interface

A sleek, modern web interface for visualizing and interacting with the sleep-inspired memory consolidation system.

## Features

### 🎨 Modern Dark Mode Interface
- Split-panel layout (resizable divider)
- Chat interface with real-time responses
- Live process logging
- Memory visualization tabs

### 💬 Chat System
- Natural conversation with the AI agent
- Memory-augmented responses
- Real-time interaction tracking

### 🧠 Memory Visualization
Three types of memory displayed in tabs:
- **Episodic Memory**: Raw interaction episodes with importance/novelty scores
- **Consolidated Memory**: Compressed summaries with key concepts
- **Schema Memory**: Abstract patterns and knowledge structures

### 😴 Sleep Cycle Tracking
- Real-time status monitoring
- 4-phase visualization:
  1. **Compression** - LLM-based episode compression
  2. **Consolidation** - Creating long-term memories
  3. **Replay** - Prioritized episode selection
  4. **Decay** - Forgetting low-value memories
- Manual sleep trigger
- Auto-sleep after threshold (5 interactions)

### 📊 Process Log
Real-time display of every system operation:
- User inputs
- Memory retrievals
- LLM generation steps
- Memory storage
- Sleep cycle phases
- Errors and warnings

## Setup

### 1. Install Dependencies
```bash
pip install flask langchain-google-genai
```

### 2. Set API Key
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. Run the Server
```bash
cd public
python app.py
```

### 4. Open Browser
Navigate to: http://localhost:5000

## Architecture

### Backend (Flask)
- `app.py` - Main Flask application
- Integrates with:
  - `agent/agent.py` - Memory agent with LLM
  - `memory/` - Episodic, consolidated, schema stores
  - `sleep/` - Consolidation, compression, replay

### Frontend
- `templates/index.html` - Main UI structure
- `static/css/style.css` - Dark mode styling
- `static/js/main.js` - Interactive functionality

## API Endpoints

- `POST /api/chat` - Send message, get response
- `GET /api/memory/episodic` - Fetch episodic memories
- `GET /api/memory/consolidated` - Fetch consolidated memories
- `GET /api/memory/schema` - Fetch schemas
- `GET /api/sleep/status` - Get sleep cycle status
- `POST /api/sleep/trigger` - Manually trigger sleep
- `GET /api/process/log` - Get process log entries

## Real-Time Tracking

Every function call is logged and displayed:

1. **User sends prompt** → Logged with timestamp
2. **Memory retrieval** → Shows concept extraction
3. **LLM generation** → Tracks query status
4. **Response stored** → Confirms episodic memory creation
5. **Sleep triggered** → All 4 phases tracked
6. **Memory updates** → Automatic refresh

## Troubleshooting

### "Agent not initialized"
- Ensure `GOOGLE_API_KEY` is set
- Check terminal for initialization errors

### No memories displayed
- Send some chat messages first
- Trigger sleep cycle to create consolidated memories

### Process log not updating
- Check browser console for errors
- Verify `/api/process/log` endpoint returns data
