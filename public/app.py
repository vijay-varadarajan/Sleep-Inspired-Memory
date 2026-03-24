from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import sys
import os
from threading import Lock

# Add parent directory to path to import agent modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import MemoryAgent

app = Flask(__name__)

# Create data directory for persistence
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

EPISODIC_FILE = os.path.join(DATA_DIR, 'episodic_memory.json')
CONSOLIDATED_FILE = os.path.join(DATA_DIR, 'consolidated_memory.json')
SCHEMA_FILE = os.path.join(DATA_DIR, 'schema_memory.json')

# Global process log for real-time tracking
process_log = []
process_log_lock = Lock()

# Global current process state
current_process = {
    "active": False,
    "stage": "idle",
    "progress": 0,
    "details": "Awaiting sleep cycle trigger"
}

# Initialize Memory Agent
agent = None

def calculate_size(obj):
    """Calculate size of object in bytes"""
    json_str = json.dumps(obj)
    return len(json_str.encode('utf-8'))

def format_size(size_bytes):
    """Format bytes to KB or MB"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def save_episodic_memory():
    """Save episodic memory to JSON file"""
    if agent is None:
        return
    
    episodes = agent.episodic_store.get_all_episodes()
    data = []
    for ep in episodes:
        ep_dict = ep.to_dict()
        ep_dict['size'] = calculate_size(ep_dict)
        data.append(ep_dict)
    
    with open(EPISODIC_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def save_consolidated_memory():
    """Save consolidated memory to JSON file"""
    if agent is None:
        return
    
    memories = agent.consolidated_store.get_all_memories()
    data = []
    for mem in memories:
        mem_dict = mem.to_dict()
        mem_dict['size'] = calculate_size(mem_dict)
        data.append(mem_dict)
    
    with open(CONSOLIDATED_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def save_schema_memory():
    """Save schema memory to JSON file"""
    if agent is None:
        return
    
    schemas = agent.schema_store.get_all_schemas()
    data = []
    for schema in schemas:
        schema_dict = schema.to_dict()
        schema_dict['size'] = calculate_size(schema_dict)
        data.append(schema_dict)
    
    with open(SCHEMA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def log_process(step, details, stage=None):
    """Log a process step for real-time display"""
    with process_log_lock:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "details": details,
            "stage": stage
        }
        process_log.append(entry)
        # Keep only last 50 entries to prevent memory bloat
        if len(process_log) > 50:
            process_log.pop(0)
    return entry

def init_agent():
    """Initialize the memory agent with error handling"""
    global agent
    if agent is None:
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                log_process("⚠️ Warning", "GOOGLE_API_KEY not set. Agent will not function.", "init")
                return None
            
            agent = MemoryAgent(
                api_key=api_key,
                auto_sleep_threshold=5  # Trigger sleep after 5 interactions
            )
            log_process("✅ Initialized", "Memory Agent successfully created", "init")
            return agent
        except Exception as e:
            log_process("❌ Error", f"Failed to initialize agent: {str(e)}", "init")
            return None
    return agent

@app.route('/')
def index():
    # Initialize agent on first request
    init_agent()
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with full process tracking"""
    global current_process
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'success': False, 'error': 'Empty message'})
    
    # Check if agent is initialized
    if agent is None:
        return jsonify({
            'success': False,
            'error': 'Agent not initialized. Please set GOOGLE_API_KEY environment variable.',
            'response': 'I cannot respond without proper API configuration.'
        })
    
    try:
        # Step 1: Log user input
        log_process("📝 Received User Prompt", f"{user_message[:100]}{'...' if len(user_message) > 100 else ''}", "input")
        
        # Step 2: Extracting concepts
        log_process("🔍 Extracting Concepts", "Analyzing user prompt for key concepts...", "retrieval")
        concepts = agent.compressor.extract_concepts_from_text(user_message)
        log_process("📊 Concepts Extracted", f"Found {len(concepts)} concepts: {', '.join(concepts[:5])}", "retrieval")
        
        # Step 3: Check consolidated memories
        log_process("🔎 Checking Consolidated Memories", "Searching for related memories...", "retrieval")
        consolidated_matches = agent.consolidated_store.search_by_concepts(concepts)
        if consolidated_matches:
            log_process("✅ Found Consolidated Memories", f"Retrieved {len(consolidated_matches)} related memories", "retrieval")
            for mem in consolidated_matches[:3]:
                agent.consolidated_store.mark_accessed(mem.id)
        else:
            log_process("ℹ️ No Consolidated Memories", "No related memories found in consolidated store", "retrieval")
        
        # Step 4: Check schemas
        log_process("🔎 Checking Schemas", "Searching for related patterns...", "retrieval")
        schema_matches = agent.schema_store.find_by_concepts(concepts, min_overlap=1)
        if schema_matches:
            log_process("✅ Found Schemas", f"Retrieved {len(schema_matches)} related schemas", "retrieval")
            for schema in schema_matches[:2]:
                agent.schema_store.mark_accessed(schema.id)
        else:
            log_process("ℹ️ No Schemas", "No related schemas found", "retrieval")
        
        # Step 5: Building system prompt
        memory_context = agent._retrieve_relevant_memories(user_message)
        persona = None
        if memory_context:
            log_process("📝 Built System Prompt", "Constructed prompt WITH memory context", "generation")
        else:
            log_process("📝 Built System Prompt", "Constructed prompt WITHOUT memory context", "generation")
        
        # Step 6: Querying LLM
        log_process("🤖 Querying LLM", "Sending request to Gemini with conversation history...", "generation")
        ai_response = agent.interact(user_message, use_memory=True)
        log_process("✅ Obtained LLM Response", f"{ai_response[:80]}{'...' if len(ai_response) > 80 else ''}", "generation")
        
        # Step 7: Storing interaction
        log_process("💾 Storing Interaction", "Creating episodic memory entry...", "storage")
        
        # Get the last episode (just created)
        recent_episodes = agent.episodic_store.get_recent(1)
        if recent_episodes:
            episode = recent_episodes[0]
            log_process("📊 Estimated Importance", f"Score: {episode.importance:.2f} (based on length and content)", "storage")
            log_process("📊 Calculated Novelty", f"Score: {episode.novelty:.2f} (compared to existing concepts)", "storage")
            log_process("✅ Added to Episode Store", f"Episode ID: {episode.id[:8]}...", "storage")
            
            # Save episodic memory immediately
            save_episodic_memory()
            log_process("💾 Saved to File", "Episodic memory persisted to episodic_memory.json", "storage")
        
        # Step 8: Check sleep threshold
        if agent.episodes_since_sleep >= agent.auto_sleep_threshold:
            log_process("😴 Sleep Threshold Exceeded", f"Reached {agent.auto_sleep_threshold} episodes - auto-sleep will trigger", "sleep-trigger")
        else:
            log_process("ℹ️ Sleep Threshold Not Exceeded", f"Episodes since sleep: {agent.episodes_since_sleep}/{agent.auto_sleep_threshold}", "storage")
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'timestamp': datetime.now().isoformat(),
            'memory_stats': agent.get_memory_summary()
        })
        
    except Exception as e:
        log_process("❌ Error", f"Chat error: {str(e)}", "error")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': 'Sorry, an error occurred while processing your message.'
        })

@app.route('/api/memory/episodic', methods=['GET'])
def get_episodic_memory():
    """Get episodic memory data from JSON file"""
    try:
        if os.path.exists(EPISODIC_FILE):
            with open(EPISODIC_FILE, 'r') as f:
                data = json.load(f)
                # Calculate total size
                total_size = sum(item.get('size', 0) for item in data)
                return jsonify({
                    'memories': data,
                    'total_size': total_size,
                    'total_size_formatted': format_size(total_size),
                    'count': len(data)
                })
        return jsonify({'memories': [], 'total_size': 0, 'total_size_formatted': '0 B', 'count': 0})
    except Exception as e:
        log_process("❌ Error", f"Failed to fetch episodic memory: {str(e)}", "error")
        return jsonify({'memories': [], 'total_size': 0, 'total_size_formatted': '0 B', 'count': 0})

@app.route('/api/memory/consolidated', methods=['GET'])
def get_consolidated_memory():
    """Get consolidated memory data from JSON file"""
    try:
        if os.path.exists(CONSOLIDATED_FILE):
            with open(CONSOLIDATED_FILE, 'r') as f:
                data = json.load(f)
                # Calculate total size
                total_size = sum(item.get('size', 0) for item in data)
                return jsonify({
                    'memories': data,
                    'total_size': total_size,
                    'total_size_formatted': format_size(total_size),
                    'count': len(data)
                })
        return jsonify({'memories': [], 'total_size': 0, 'total_size_formatted': '0 B', 'count': 0})
    except Exception as e:
        log_process("❌ Error", f"Failed to fetch consolidated memory: {str(e)}", "error")
        return jsonify({'memories': [], 'total_size': 0, 'total_size_formatted': '0 B', 'count': 0})

@app.route('/api/memory/schema', methods=['GET'])
def get_schema_memory():
    """Get schema memory data from JSON file"""
    try:
        if os.path.exists(SCHEMA_FILE):
            with open(SCHEMA_FILE, 'r') as f:
                data = json.load(f)
                # Calculate total size
                total_size = sum(item.get('size', 0) for item in data)
                return jsonify({
                    'memories': data,
                    'total_size': total_size,
                    'total_size_formatted': format_size(total_size),
                    'count': len(data)
                })
        return jsonify({'memories': [], 'total_size': 0, 'total_size_formatted': '0 B', 'count': 0})
    except Exception as e:
        log_process("❌ Error", f"Failed to fetch schema memory: {str(e)}", "error")
        return jsonify({'memories': [], 'total_size': 0, 'total_size_formatted': '0 B', 'count': 0})

@app.route('/api/sleep/status', methods=['GET'])
def get_sleep_status():
    """Get current sleep cycle status"""
    if agent is None:
        return jsonify(current_process)
    
    # Update from agent state
    status = {
        "active": current_process["active"],
        "stage": current_process["stage"],
        "progress": current_process["progress"],
        "details": current_process["details"],
        "episodes_since_sleep": agent.episodes_since_sleep,
        "total_episodes": len(agent.episodic_store.episodes),
        "sleep_cycles": agent.sleep_cycle.cycle_count
    }
    return jsonify(status)

@app.route('/api/sleep/trigger', methods=['POST'])
def trigger_sleep():
    """Manually trigger sleep cycle with full detailed process tracking"""
    global current_process
    
    if agent is None:
        return jsonify({
            'success': False,
            'message': 'Agent not initialized'
        })
    
    try:
        log_process("😴 Sleep Cycle Initiated", "Starting memory consolidation process...", "sleep")
        current_process.update({
            "active": True,
            "stage": "replay",
            "progress": 5,
            "details": "Phase 1: Prioritized Replay"
        })
        
        # ===== PHASE 1: REPLAY =====
        log_process("🔄 Starting Replay Phase", "Selecting high-priority episodes for consolidation...", "replay")
        
        unconsolidated = agent.episodic_store.get_unconsolidated()
        log_process("📊 Replay: Episode Count", f"Found {len(unconsolidated)} unconsolidated episodes", "replay")
        
        if len(unconsolidated) > 0:
            log_process("🔍 Replay: Calculating Priorities", "Computing importance, novelty, recency scores...", "replay")
            
            # Calculate priorities for selection
            from sleep.replay import calculate_replay_priority
            from datetime import datetime
            current_time = datetime.now()
            
            priorities = []
            for ep in unconsolidated:
                priority = calculate_replay_priority(ep, current_time)
                priorities.append((ep, priority))
            
            priorities.sort(key=lambda x: x[1], reverse=True)
            selected_count = min(len(priorities), agent.sleep_cycle.replay_batch_size)
            
            log_process("✅ Replay: Episodes Selected", f"Selected top {selected_count} episodes for consolidation", "replay")
            
            if selected_count > 0:
                avg_priority = sum(p[1] for p in priorities[:selected_count]) / selected_count
                log_process("📊 Replay Stats", f"Avg priority: {avg_priority:.3f}, Batch size: {selected_count}", "replay")
        else:
            log_process("ℹ️ Replay: No Episodes", "No unconsolidated episodes to process", "replay")
        
        current_process.update({
            "stage": "compression",
            "progress": 25,
            "details": "Phase 2: Generative Compression & Consolidation"
        })
        
        # ===== PHASE 2: COMPRESSION & CONSOLIDATION =====
        log_process("🗜️ Starting Compression Phase", "Compressing episodes with LLM...", "compression")
        log_process("🔍 Compression: Grouping Episodes", f"Creating batches of {agent.sleep_cycle.consolidation_batch_size} episodes", "compression")
        
        # Run actual sleep cycle
        stats = agent.sleep(verbose=False)
        
        log_process("✅ Compression Complete", f"Generated {stats.get('memories_consolidated', 0)} compressed summaries", "compression")
        
        if stats.get('memories_consolidated', 0) > 0:
            log_process("💾 Consolidation: Creating Memories", "Storing consolidated memories with key concepts...", "consolidation")
            log_process("📊 Consolidation: Extracting Concepts", "LLM extracted key concepts from episodes", "consolidation")
            log_process("✅ Consolidation Complete", f"Created {stats['memories_consolidated']} consolidated memories", "consolidation")
            
            # Save consolidated memories
            save_consolidated_memory()
            log_process("💾 Saved Consolidated", "Persisted to consolidated_memory.json", "consolidation")
            
            # Update episodic memory (mark as consolidated)
            save_episodic_memory()
            log_process("💾 Updated Episodic", "Marked episodes as consolidated", "consolidation")
        
        current_process.update({
            "stage": "schema",
            "progress": 60,
            "details": "Phase 3: Schema Formation"
        })
        
        # ===== PHASE 3: SCHEMA FORMATION =====
        log_process("🧠 Starting Schema Formation", "Analyzing patterns across consolidated memories...", "schema")
        
        all_consolidated = agent.consolidated_store.get_all_memories()
        log_process("📊 Schema: Memory Count", f"Analyzing {len(all_consolidated)} consolidated memories", "schema")
        
        if len(all_consolidated) >= agent.sleep_cycle.schema_min_memories:
            log_process("🔍 Schema: Grouping by Concepts", "Finding memories with shared concepts...", "schema")
            log_process("🔍 Schema: Pattern Detection", "Identifying recurring themes and relationships...", "schema")
            
            if stats['schemas_formed'] > 0:
                log_process("✅ Schema: Formation Complete", f"Induced {stats['schemas_formed']} new schemas", "schema")
                log_process("📊 Schema Stats", f"Total schemas: {len(agent.schema_store.get_all_schemas())}", "schema")
                
                # Save schemas
                save_schema_memory()
                log_process("💾 Saved Schemas", "Persisted to schema_memory.json", "schema")
            else:
                log_process("ℹ️ Schema: No New Patterns", "No new schemas formed (insufficient shared concepts)", "schema")
        else:
            log_process("ℹ️ Schema: Insufficient Data", f"Need {agent.sleep_cycle.schema_min_memories} consolidated memories, have {len(all_consolidated)}", "schema")
        
        current_process.update({
            "stage": "decay",
            "progress": 85,
            "details": "Phase 4: Memory Decay & Cleanup"
        })
        
        # ===== PHASE 4: DECAY =====
        log_process("⏳ Starting Decay Phase", "Evaluating low-value memories for removal...", "decay")
        log_process("🔍 Decay: Checking Age & Importance", "Identifying old, low-importance episodes...", "decay")
        
        if stats['episodes_forgotten'] > 0:
            log_process("🧹 Decay: Removed Episodes", f"Forgot {stats['episodes_forgotten']} low-value episodes", "decay")
            log_process("📊 Decay Stats", f"Freed up memory space from forgotten episodes", "decay")
            
            # Update episodic memory file
            save_episodic_memory()
            log_process("💾 Updated Episodic Store", "Removed forgotten episodes from storage", "decay")
        else:
            log_process("ℹ️ Decay: No Removals", "All episodes retained (sufficient importance/recency)", "decay")
        
        current_process.update({
            "active": False,
            "stage": "idle",
            "progress": 100,
            "details": "Sleep cycle complete"
        })
        
        # Final summary
        log_process("✅ Sleep Cycle Complete", f"Cycle #{stats['cycle_number']} finished successfully", "complete")
        log_process("📊 Final Stats", 
                   f"Replayed: {stats.get('episodes_replayed', 0)}, " +
                   f"Consolidated: {stats['memories_consolidated']}, " +
                   f"Schemas: {stats['schemas_formed']}, " +
                   f"Forgotten: {stats['episodes_forgotten']}", 
                   "complete")
        
        return jsonify({
            'success': True,
            'message': 'Sleep cycle completed',
            'stats': stats
        })
        
    except Exception as e:
        log_process("❌ Sleep Error", f"Error during sleep cycle: {str(e)}", "error")
        current_process.update({
            "active": False,
            "stage": "idle",
            "progress": 0,
            "details": f"Error: {str(e)}"
        })
        return jsonify({
            'success': False,
            'message': f'Sleep cycle failed: {str(e)}'
        })

@app.route('/api/process/log', methods=['GET'])
def get_process_log():
    """Get recent process log entries"""
    with process_log_lock:
        # Return last 50 entries
        return jsonify(process_log[-50:])

@app.route('/api/memory/reset', methods=['POST'])
def reset_memory():
    """Reset all memory files (delete contents)"""
    try:
        # Clear all memory files
        for filepath in [EPISODIC_FILE, CONSOLIDATED_FILE, SCHEMA_FILE]:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Reset the agent's memory stores
        if agent:
            agent.episodic_store.clear()
            agent.consolidated_store.clear()
            agent.schema_store.clear()
            agent.conversation_history = []
            agent.interaction_count = 0
            agent.episodes_since_sleep = 0
        
        log_process("🔄 Reset", "All memory systems cleared", "system")
        
        return jsonify({
            'success': True,
            'message': 'All memory cleared successfully'
        })
    except Exception as e:
        log_process("❌ Reset Error", f"Failed to reset memory: {str(e)}", "error")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
