"""Run train.py and restart only after a reported CUDA OOM."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--name', required=True)
    parser.add_argument('--restore_ckpt')
    parser.add_argument('--pretrain_ckpt')
    parser.add_argument('--batch_size', type=int)
    args, training_args = parser.parse_known_args()
    run = Path('checkpoints') / args.name
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_name = re.compile(rf'{re.escape(run.name)}_e(\d+)\.pth')
    delay = max(1., float(os.environ.get('OOM_RETRY_SECONDS', '120')))
    min_free = int(os.environ.get('OOM_MIN_FREE_MIB', '0'))
    adaptive = os.environ.get('OOM_ADAPTIVE_BATCH') == '1'
    free_for_four = int(os.environ.get('OOM_BATCH4_FREE_MIB', '31000'))
    free_for_two = int(os.environ.get('OOM_BATCH2_FREE_MIB', '21500'))
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]

    def log(message):
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S %z")}] {message}', flush=True)

    def wait_for_gpu(prefer_two):
        while min_free or adaptive:
            query = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                             '--format=csv,noheader,nounits'], text=True)
            free = int(query.strip())
            if adaptive:
                if free >= free_for_four and not prefer_two:
                    return 4
                if free >= free_for_two:
                    return 2
                log(f'GPU {gpu}: {free} MiB free, batch 2 requires {free_for_two} MiB. Checking in {delay:g}s')
                time.sleep(delay)
                continue
            if free >= min_free:
                break
            log(f'GPU {gpu}: {free} MiB free, waiting for {min_free} MiB ({delay:g}s)')
            time.sleep(delay)
        return args.batch_size

    child = None

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
            child.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (run / '.training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'A training supervisor is already running for {run}')
        if args.restore_ckpt and args.pretrain_ckpt:
            parser.error('Pass only one of --restore_ckpt and --pretrain_ckpt')
        checkpoint = args.restore_ckpt or args.pretrain_ckpt
        checkpoint_flag = '--pretrain_ckpt' if args.pretrain_ckpt else '--restore_ckpt'
        if checkpoint is None and any(checkpoint_name.fullmatch(p.name) for p in run.iterdir()):
            raise SystemExit(f'{run} has checkpoints. Specify --restore_ckpt or use a new run name.')
        prefer_two = False
        while True:
            batch = wait_for_gpu(prefer_two)
            command = [sys.executable, '-u', 'train.py', '--name', args.name, *training_args]
            if batch is not None:
                command.extend(['--batch_size', str(batch)])
            if checkpoint:
                command.extend([checkpoint_flag, checkpoint])
            log(f'Starting {args.name} from {checkpoint or "initialization"}, batch size {batch}')
            child = subprocess.Popen(command, env={**os.environ, 'STEREO_OOM_SUPERVISED': '1',
                                                   'STEREO_ADAPTIVE_BATCH': '1' if adaptive else '0'},
                                     start_new_session=True)
            status = child.wait()
            if status != 75:
                return status if status >= 0 else 128 - status
            prefer_two = adaptive and batch == 4
            saved = [(int(match[1]), p) for p in run.iterdir()
                     if (match := checkpoint_name.fullmatch(p.name)) and p.is_file()]
            if saved:
                checkpoint = str(max(saved)[1])
                checkpoint_flag = '--restore_ckpt'
            log(f'CUDA OOM. Retrying from {checkpoint or "initialization"} in {delay:g}s')
            time.sleep(delay)


if __name__ == '__main__':
    sys.exit(main())
