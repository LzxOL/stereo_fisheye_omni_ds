"""Linux/container service: stop one identified training process after a saved epoch.

Watch CLOSE_WRITE rather than file existence: torch.save writes checkpoints in
place. The learning-rate schedule and training source are left unchanged.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import struct
import time


def process_identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    return text[text.rfind(')') + 2:].split()[19]  # Linux starttime, field 22


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    identity = process_identity(args.pid)
    command = Path(f'/proc/{args.pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    target = args.checkpoint_dir / f'{args.prefix}_e{args.epochs - 1}.pth'
    args.status.parent.mkdir(parents=True, exist_ok=True)

    def status(state, **extra):
        record = dict(state=state, time=time.time(), pid=args.pid, start_ticks=identity,
                      command=command, target_epochs=args.epochs, checkpoint=str(target), **extra)
        temporary = args.status.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(args.status)

    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(args.checkpoint_dir), 0x8 | 0x80) < 0:
        raise OSError(ctypes.get_errno(), 'Cannot watch checkpoint writes')
    status('watching')
    # Register the watch before checking existing files, avoiding a startup race.
    existing = target.exists()
    try:
        while True:
            try:
                if process_identity(args.pid) != identity:
                    status('process_identity_changed')
                    return
            except FileNotFoundError:
                status('training_exited_before_target')
                return
            ready = existing
            existing = False
            if select.select([fd], [], [], 5)[0]:
                events = os.read(fd, 65536)
                offset = 0
                while offset < len(events):
                    _, mask, _, length = struct.unpack_from('iIII', events, offset)
                    name = events[offset + 16:offset + 16 + length].split(b'\0', 1)[0].decode()
                    ready |= name == target.name and bool(mask & (0x8 | 0x80))
                    offset += 16 + length
            if not ready:
                status('watching')
                continue
            # Stop just this PID while checking the completed CPU-readable checkpoint.
            os.kill(args.pid, signal.SIGSTOP)
            stopped = True
            try:
                import torch
                checkpoint = torch.load(target, map_location='cpu', weights_only=False)
                required = {'epoch', 'net_state_dict', 'optimizer', 'scheduler'}
                if not required.issubset(checkpoint) or checkpoint['epoch'] != args.epochs - 1:
                    raise ValueError('Checkpoint epoch or required state does not match')
                details = dict(completed_epochs=checkpoint['epoch'] + 1,
                               global_step=checkpoint.get('global_step'),
                               checkpoint_bytes=target.stat().st_size)
                del checkpoint
                status('stopping_after_checkpoint', **details)
                os.kill(args.pid, signal.SIGTERM)
                os.kill(args.pid, signal.SIGCONT)
                stopped = False
                status('stop_signal_sent', **details)
                for _ in range(100):
                    proc = Path(f'/proc/{args.pid}/stat')
                    exited = not proc.exists()
                    if not exited:
                        stat = proc.read_text()
                        exited = stat[stat.rfind(')') + 2:].split()[0] == 'Z'
                    if exited:
                        status('stopped_at_target', **details)
                        return
                    time.sleep(.1)
                status('termination_not_confirmed', **details)
                return
            except Exception as error:
                status('checkpoint_not_ready', error=repr(error))
            finally:
                if stopped:
                    os.kill(args.pid, signal.SIGCONT)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
