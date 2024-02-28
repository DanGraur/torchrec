import time

from torch.utils.data import DataLoader
import torch.utils.data.datapipes as dp

from torchrec.datasets.criteo import criteo_terabyte

WORKER_COUNT=94
BATCH_SIZE = 32
MAX_BATCH_COUNT = 1000
FILES = ['/home/dan/projects/db-preproc/data/train.tsv'] * 13

# Set up the input pipeline
df = criteo_terabyte(FILES)
df = dp.iter.Batcher(df, BATCH_SIZE)
df = dp.iter.Collator(df)
dataset = DataLoader(df, num_workers=WORKER_COUNT)

batch_iterator = iter(dataset)

s = time.time()
for batch_idx, batch in enumerate(batch_iterator):
  if batch_idx >= MAX_BATCH_COUNT:
    break
  if batch_idx % 1000 == 0:
    print(f"Created batch #{batch_idx}")
s = time.time() - s

print(f"Completed the experiment; {MAX_BATCH_COUNT * BATCH_SIZE / s} "
      f"[instances/s] = ({MAX_BATCH_COUNT} * {BATCH_SIZE}) / {s}")
