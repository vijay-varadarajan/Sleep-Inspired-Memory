from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json

app = Flask(__name__)

# Store conversation history (in production, use a database)
conversation_history = []

# Sample memory data structures
episodic_memory = [
    {"timestamp": "2026-02-08 10:30", "content": "User discussed neural networks", "importance": 0.8},
    {"timestamp": "2026-02-08 10:45", "content": "Conversation about memory consolidation", "importance": 0.9},
]

consolidated_memory = [
    {"concept": "Machine Learning", "strength": 0.85, "connections": ["Neural Networks", "Deep Learning"]},
    {"concept": "Memory Systems", "strength": 0.92, "connections": ["Episodic", "Semantic", "Procedural"]},
]

schema_memory = [
    {"schema": "Conversation Pattern", "structure": {"greeting": True, "topic_exploration": True, "conclusion": True}},
    {"schema": "Learning Session", "structure": {"introduction": True, "examples": True, "practice": True}},
]

current_process = {
    "active": False,
    "stage": "idle",
    "progress": 0,
    "details": "Awaiting sleep cycle trigger"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    user_message = data.get('message', '')
    
    # Add user message to history
    conversation_history.append({
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })
    
    # Generate a sample AI response
    ai_response = f"This is a sample response to: '{user_message}'. In production, this would be connected to your actual AI agent."
    
    conversation_history.append({
        'role': 'assistant',
        'content': ai_response,
        'timestamp': datetime.now().isoformat()
    })
    
    return jsonify({
        'success': True,
        'response': ai_response,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/memory/episodic', methods=['GET'])
def get_episodic_memory():
    """Get episodic memory data"""
    return jsonify(episodic_memory)

@app.route('/api/memory/consolidated', methods=['GET'])
def get_consolidated_memory():
    """Get consolidated memory data"""
    return jsonify(consolidated_memory)

@app.route('/api/memory/schema', methods=['GET'])
def get_schema_memory():
    """Get schema memory data"""
    return jsonify(schema_memory)

@app.route('/api/sleep/status', methods=['GET'])
def get_sleep_status():
    """Get current sleep cycle status"""
    return jsonify(current_process)

@app.route('/api/sleep/trigger', methods=['POST'])
def trigger_sleep():
    """Manually trigger sleep cycle (for demo purposes)"""
    global current_process
    current_process = {
        "active": True,
        "stage": "compression",
        "progress": 0,
        "details": "Beginning memory compression phase"
    }
    return jsonify({'success': True, 'message': 'Sleep cycle initiated'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
