import time

from torch.utils.data import DataLoader
import torch.utils.data.datapipes as dp

from torchrec.datasets.criteo import criteo_terabyte

WORKER_COUNT = 94
BATCH_SIZE = 32
EXPERIMENT_TIME = 120  # should be in seconds
FILES = ['/home/dan/projects/db-preproc/data/train.tsv'] * 13

# Set up the input pipeline
df = criteo_terabyte(FILES)
df = dp.iter.Batcher(df, BATCH_SIZE)
df = dp.iter.Collator(df)
dataset = DataLoader(df, num_workers=WORKER_COUNT)

batch_iterator = iter(dataset)

s = time.time()
batch_idx = 0
while time.time() - s >= EXPERIMENT_TIME:
  batch = next(batch_iterator)
  if batch_idx % 1000 == 0:
    print(f"At batch #{batch_idx}")
  batch_idx += 1
s = time.time() - s

# for batch_idx, batch in enumerate(batch_iterator):
#   if time.time() - s >= EXPERIMENT_TIME:
#     break
#   if batch_idx % 1000 == 0:
#     print(f"At batch #{batch_idx}")
# s = time.time() - s

print(f"Completed the experiment; {batch_idx * BATCH_SIZE / s} "
      f"[instances/s] = ({batch_idx} * {BATCH_SIZE}) / {s}")
