#!/usr/bin/env bash
# Sequential experiment queue for one GPU. Usage: run_overnight.sh <gpu> <queue>
#
# Each job runs to completion before the next starts, so two of these (one per
# card) can run unattended. Progress is appended to /tmp/overnight/progress.log
# so a partial queue is still readable if something dies.
set -u

GPU=$1
QUEUE=$2
OUT=/tmp/overnight
mkdir -p "$OUT"
PROGRESS="$OUT/progress.log"

DATA=/home/tom/Developments/Raman/oracle/data/processed
GLASGOW=$DATA/glasgow/glasgow_16_trimmed_smoothed.zarr
MIXED=$DATA/glasgow/glasgow_mixed_1_16.zarr
FISH=$DATA/fish/fish_6frame.zarr
SMALL="8,16,32,64"

run () {                       # run <name> <args...>
  local name=$1; shift
  local log="$OUT/${name}.log"
  echo "$(date +%H:%M:%S)  gpu$GPU  START  $name" >> "$PROGRESS"
  CUDA_VISIBLE_DEVICES=$GPU python -u -m lumos.train "$@" > "$log" 2>&1
  local status=$?
  local ckpt
  ckpt=$(ls -dt "$PWD"/checkpoints/*/ 2>/dev/null | head -1)
  echo "$(date +%H:%M:%S)  gpu$GPU  DONE   $name  exit=$status  $(basename "${ckpt:-none}")" >> "$PROGRESS"
}

case $QUEUE in
# ---------------------------------------------------------------- seeds and size
a)
  # The headline configuration rests on a single run. Three more seeds decide
  # whether 0.940 at T=16 and 0.927 at T=1 is real.
  run seed1_small_uniform  --data "$GLASGOW" --n_times_train 16 --conv_channels "$SMALL" --seed 1
  run seed2_small_uniform  --data "$GLASGOW" --n_times_train 16 --conv_channels "$SMALL" --seed 2
  run seed3_small_uniform  --data "$GLASGOW" --n_times_train 16 --conv_channels "$SMALL" --seed 3
  # Does the size floor move once sampling is uniform?
  run size_4_8_16_32       --data "$GLASGOW" --n_times_train 16 --conv_channels "4,8,16,32"
  run size_16_32_64_128    --data "$GLASGOW" --n_times_train 16 --conv_channels "16,32,64,128"
  # Fixed window at the new size, as the sampling control.
  run small_fixedT         --data "$GLASGOW" --n_times_train 16 --conv_channels "$SMALL" --t_sampling ""
  # Held-out estimate of the headline.
  run small_uniform_induct --data "$GLASGOW" --n_times_train 16 --conv_channels "$SMALL" --no-transductive
  ;;
# ------------------------------------------------- fish and the semi-supervised question
b)
  # Fish has never been run with the new defaults. Uniform cost it 0.054 at 1000
  # epochs with the large encoder; 2000 epochs may recover it as it did on glasgow.
  run fish_t3_new          --data "$FISH" --n_times_train 3 --conv_channels "$SMALL"
  run fish_t6_new          --data "$FISH" --n_times_train 6 --conv_channels "$SMALL"
  run fish_t3_fixedT       --data "$FISH" --n_times_train 3 --conv_channels "$SMALL" --t_sampling ""
  run fish_t6_fixedT       --data "$FISH" --n_times_train 6 --conv_channels "$SMALL" --t_sampling ""
  run fish_t3_induct       --data "$FISH" --n_times_train 3 --conv_channels "$SMALL" --no-transductive
  # Does adding unlabelled single frames help? Compare against the inductive
  # 186-cell run already in flight.
  run mixed_seed1          --data "$MIXED" --n_times_train 16 --conv_channels "$SMALL" --seed 1
  run mixed_seed2          --data "$MIXED" --n_times_train 16 --conv_channels "$SMALL" --seed 2
  run mixed_induct         --data "$MIXED" --n_times_train 16 --conv_channels "$SMALL" --no-transductive
  ;;
esac

echo "$(date +%H:%M:%S)  gpu$GPU  QUEUE $QUEUE COMPLETE" >> "$PROGRESS"
