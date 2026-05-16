ABSTRACT:

Large language Models (LLMs) and LLM agents are excellent at in context reasoning but show a degradation in consistent memory across sessions and over multimodal interactions. So, there are limitations in the LLM’s ability to build on prior interactions and abstract general knowledge from experience. An analogous process of biological memory is that during sleep, replay of episodes and transfer of systematic layouts is done by the hippocampus to the neocortex. This is done via oscillating activity and spindle dependent information processing. We draw directly on this mechanism to design a sleep inspired memory consolidation framework for LLM agents.

This framework is implemented as a three tier memory architecture accompanied by a five phase sleep cycle covering replay. We evaluate it against four baselines across four datasets including text and image. The sleep cycle enabled framework achieves higher answer utility, multi session continuity, and a delayed recall gain on the tested datasets. These results suggest that structured offline consolidation modeled after the human sleep can immensely improve LLM and agentic memory.

COMMANDS TO EXECUTE:

cd Sleep-Inspired-Memory

source .venv/bin/activate

python personamem_preprocessing.py

python benchmark_runner.py --split benchmark --num_samples 100 --methods all

python personachat_preprocessing.py

python personachat_runner.py --split validation --num_samples 200 --methods all

python locomo_preprocessing.py

python locomo_runner.py--split benchmark --num_samples 200 --methods all

python okvqa_preprocessing.py

python okvqa_runner.py --split benchmark --num_samples 50 --methods sleep

python okvqa_postprocessing.py --output_dir okvqa_results
