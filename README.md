cd /home/spongyshaman/Documents/Sleep-Inspired-Memory

source /home/spongyshaman/Documents/Sleep-Inspired-Memory/.venv/bin/activate

python personamem_preprocessing.py

python benchmark_runner.py --split benchmark --num_samples 100 --methods all

python personachat_preprocessing.py

python personachat_runner.py --split validation --num_samples 200 --methods all

python locomo_preprocessing.py

python locomo_runner.py--split benchmark --num_samples 200 --methods all

python okvqa_preprocessing.py

python okvqa_runner.py --split benchmark --num_samples 50 --methods sleep

python okvqa_postprocessing.py --output_dir okvqa_results