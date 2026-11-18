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
    args, training_args = parser.parse_known_args()
    run = Path('checkpoints') / args.name
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_name = re.compile(rf'{re.escape(run.name)}_e(\d+)\.pth')
    delay = max(1., float(os.environ.get('OOM_RETRY_SECONDS', '120')))
    min_free = int(os.environ.get('OOM_MIN_FREE_MIB', '0'))
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]

    def log(message):
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S %z")}] {message}', flush=True)

    def wait_for_gpu():
        while min_free:
            query = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                             '--format=csv,noheader,nounits'], text=True)
            free = int(query.strip())
            if free >= min_free:
                break
            log(f'GPU {gpu}: {free} MiB free, waiting for {min_free} MiB ({delay:g}s)')
            time.sleep(delay)

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
        while True:
            wait_for_gpu()
            command = [sys.executable, '-u', 'train.py', '--name', args.name, *training_args]
            if checkpoint:
                command.extend([checkpoint_flag, checkpoint])
            log(f'Starting {args.name} from {checkpoint or "initialization"}')
            child = subprocess.Popen(command, env={**os.environ, 'STEREO_OOM_SUPERVISED': '1'},
                                     start_new_session=True)
            status = child.wait()
            if status != 75:
                return status if status >= 0 else 128 - status
            saved = [(int(match[1]), p) for p in run.iterdir()
                     if (match := checkpoint_name.fullmatch(p.name)) and p.is_file()]
            if saved:
                checkpoint = str(max(saved)[1])
                checkpoint_flag = '--restore_ckpt'
            log(f'CUDA OOM. Retrying from {checkpoint or "initialization"} in {delay:g}s')
            time.sleep(delay)


if __name__ == '__main__':
    sys.exit(main())
