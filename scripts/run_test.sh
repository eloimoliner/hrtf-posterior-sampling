#!/bin/bash
# Runs the proposed method on every sample of a test set (e.g. test_v3p1_600), in parallel.
# Settings come from downstream_tasks/hrtf_estimation.yaml (T=101, zeta=1.25, as in the paper's final run).
# Usage: scripts/run_test.sh <test_set_dir> [n_workers] [threads_per_worker]
# REVERSE=1 processes samples in reverse order (to run a second instance, e.g. on CPU, alongside).
# GPUS="0 1 2 3" spreads samples round-robin over those GPUs; GPUS="" runs on CPU.
# Resumable: samples whose estimate already exists are skipped.
set -u
cd "$(dirname "$0")/.."

main_path=$1                # e.g. /scratch/.../Binaural_BUDDy/test_v3p1_600
n_workers=${2:-12}
threads=${3:-8}
python=${PYTHON:-uv run --frozen python}
GPUS=${GPUS-0}
read -ra gpu_list <<< "$GPUS"
n_gpus=${#gpu_list[@]}

run_name=hrtf_estimates
conf=hrtf_estimation.yaml
mkdir -p $main_path/$run_name/logs $main_path/hrtf

run_one() {
    idx=$1
    basename=$2
    if [ $n_gpus -gt 0 ]; then
        gpu_list=($GPUS)
        export CUDA_VISIBLE_DEVICES=${gpu_list[$((idx % n_gpus))]}
        force_cpu=False
    else
        force_cpu=True
    fi
    out=$main_path/$run_name/$basename
    [ -f $out.npy ] && return 0
    if [ ! -f $main_path/brir/${basename}_receiver_0.wav ]; then
        echo "SKIP $basename (no brir)"; return 0
    fi
    OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads $python -m downstream_tasks.hrtf_estimation --config-name=$conf \
        gpu=0 device.force_cpu=$force_cpu \
        hydra.run.dir=$main_path/$run_name/logs/$basename \
        kwargs.speech_file=$main_path/speech_48k/${basename}_receiver_0.flac \
        kwargs.brir_file=$main_path/brir/${basename}_receiver_0.wav \
        kwargs.hrir_file=$main_path/hrir/$basename.np \
        kwargs.doa_file=$main_path/doa/$basename.np \
        kwargs.anthropometry_file=$main_path/anthropometry/$basename.np \
        kwargs.save_hrtf_file=$out \
        kwargs.save_gt_hrtf_file=$main_path/hrtf/$basename \
        kwargs.save_parameters_file=$out.pth \
        > $main_path/$run_name/logs/$basename.txt 2>&1 \
        && echo "done $basename" || echo "FAILED $basename"
}
export -f run_one
export main_path run_name conf threads python GPUS n_gpus

ls $main_path/hrir | sed 's/\.np$//' | sort ${REVERSE:+-r} | awk '{print NR-1, $0}' | xargs -P $n_workers -L1 bash -c 'run_one "$0" "$1"'
