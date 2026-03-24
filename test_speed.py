"""Quick speed test for benchmark components."""
import os
import time
from dotenv import load_dotenv

load_dotenv()

print(f"API Key set: {'GOOGLE_API_KEY' in os.environ}")

from evaluation.baselines import create_agent
from evaluation.personamem_benchmark import PersonaMemEvaluator

print('\n=== Testing Agent Speed ===')
print('Creating vanilla agent...')
start = time.time()
agent = create_agent('vanilla')
print(f'Created in {time.time()-start:.2f}s')

print('\nRunning one interaction...')
start = time.time()
response = agent.interact('Hello, how are you?', persona='Test user', use_memory=False)
print(f'Interaction took {time.time()-start:.2f}s')
print(f'Response: {response[:100]}...')

print('\n=== Testing Evaluator Speed ===')
print('Creating evaluator...')
start = time.time()
evaluator = PersonaMemEvaluator()
print(f'Created in {time.time()-start:.2f}s')

print('\nRunning QA evaluation...')
start = time.time()
result = evaluator.evaluate_long_horizon_qa(
    response="The sky is blue.",
    correct_answer="blue",
    incorrect_answers=["red", "green"]
)
print(f'Evaluation took {time.time()-start:.2f}s')
print(f'Result: {result}')
