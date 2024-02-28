import time

import torch.utils.data.datapipes as dp

from torchrec.datasets.criteo import criteo_terabyte

BATCH_SIZE = 32
MAX_BATCH_COUNT = 1000
FILES = ['/home/dan/projects/db-preproc/data'] * 13

# Set up the input pipeline
df = criteo_terabyte(FILES)
df = dp.iter.Batcher(df, BATCH_SIZE)
df = dp.iter.Collator(df)

batch_iterator = iter(df)

s = time.time()
for batch_idx, batch in enumerate(batch_iterator):
  if batch_idx >= MAX_BATCH_COUNT:
    break
  if batch_idx % 1000 == 0:
    print(f"Created batch #{batch_idx}")
s = time.time() - s

print(f"Completed the experiment; {MAX_BATCH_COUNT * BATCH_SIZE / s} "
      f"[instances/s] = {MAX_BATCH_COUNT} * {BATCH_SIZE}) / {s}")
