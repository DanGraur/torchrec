from absl import app
from absl import flags

import time

from torch.utils.data import DataLoader
import torch.utils.data.datapipes as dp

from torchrec.datasets.criteo import criteo_terabyte

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 512, 'The batch size.')
flags.DEFINE_integer('experiment_time', 120, 'The total experiment time.')
flags.DEFINE_integer('worker_count', 32, 'The total number of parallel workers.')

def main(argv):
  del argv
  FILES = ['/home/dan/projects/db-preproc/data/train.tsv'] * FLAGS.worker_count

  df = criteo_terabyte(FILES)
  df = dp.iter.Batcher(df, FLAGS.batch_size)
  df = dp.iter.Collator(df)
  dataset = DataLoader(df, num_workers=FLAGS.worker_count)

  batch_iterator = iter(dataset)

  batch_idx = 0
  s = time.time()
  while time.time() - s <= FLAGS.experiment_time:
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

  print(f"Instance Throughput {batch_idx * FLAGS.batch_size / s} "
        f"[instances/s] = ({batch_idx} * {FLAGS.batch_size}) / {s}")


if __name__ == '__main__':
  app.run(main)
