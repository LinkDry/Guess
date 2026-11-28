"""Run the support-six model, exact elimination and Newton probe under one budget."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from time import perf_counter, sleep


MEMORY_MIB = 3520
SECONDS = 1790


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def worker(args):
    if sys.stdin.readline() != 'start\n':
        return 2
    from build_support5_lmi import model_inputs
    from reduce_support5_lmi import reduce_lmi
    from solve_support5_barrier import Pencil, find_margin

    deadline = float(os.environ['ETF_SUPPORT6_DEADLINE'])
    model_dir = args.output
    phase_args = SimpleNamespace(face=args.prior/'model.json', chart=args.prior/'affine_chart.json.gz',
                                 output=model_dir, stage='support6', memory_mib=args.memory_mib,
                                 prior_source='support5', deadline=deadline)
    if args.resume:
        if not (model_dir/'model.json').exists() or not all((model_dir/f'psd_map_{k}.npz').exists() for k in range(8)):
            raise ValueError('the guarded model phase must have completed')
        print('Reusing the guarded support-six assembly', flush=True)
    else:
        print('Generating exact sparse support-six model', flush=True)
        begin = perf_counter()
        model_inputs(phase_args)
        print(f'Model phase {perf_counter()-begin:.2f} s', flush=True)
    begin = perf_counter()
    reduce_lmi(phase_args)
    print(f'Elimination, rank and export phase {perf_counter()-begin:.2f} s', flush=True)
    if not (model_dir/'pencil/manifest.json').exists():
        return 0
    if deadline-perf_counter() < 40:
        write_report(args.output/'solve_gate.json', {'status': 'insufficient_remaining_time'})
        return 0
    phase_output = model_dir/'barrier'
    phase_output.mkdir()
    thread_environment = {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}
    write_report(args.output/'solve_gate.json', {'status': 'started', 'remaining_seconds': deadline-perf_counter(),
                                                 'thread_environment': thread_environment})
    with Pencil(model_dir/'pencil', phase_output/'pencil_scratch') as pencil:
        find_margin(pencil, phase_output, deadline-12, 16)
    return 0


def supervise(args):
    import psutil
    from solve_general_4_9 import THREADS, windows_job

    args.output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    env = dict(os.environ, PYTHONUNBUFFERED='1', **{name: str(args.threads) for name in THREADS})
    if os.name == 'nt' and sys.prefix != sys.base_prefix:
        command[0] = sys._base_executable
        env['__PYVENV_LAUNCHER__'] = sys.executable
    started = perf_counter()
    elapsed = 0.
    if args.resume:
        first = json.loads((args.output/'probe_guard.json').read_text(encoding='utf-8'))
        if (not first['windows_job_commit_limit_enforced'] or first['kill_reason'] is not None
                or first['exit_code'] == 0 or first['memory_limit_bytes'] != args.memory_mib*2**20):
            raise ValueError('resume requires a prior non-budget failure with the same hard cap')
        elapsed = first['seconds']
    deadline = started+args.seconds-elapsed
    env['ETF_SUPPORT6_DEADLINE'] = str(deadline)
    limit = args.memory_mib*2**20
    peak, reason, release = 0, None, None
    prefix = 'resumed_probe' if args.resume else 'probe'
    with (args.output/f'{prefix}.log').open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, text=True)
        try:
            release = windows_job(child,limit)
            process = psutil.Process(child.pid)
            child.stdin.write('start\n')
            child.stdin.close()
            while child.poll() is None:
                try:
                    peak = max(peak,process.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak >= limit or perf_counter() >= deadline:
                    reason = 'rss_limit' if peak >= limit else 'wall_limit'
                    break
                sleep(max(0.,min(0.05,deadline-perf_counter())))
        finally:
            if release is not None:
                release()
            if child.poll() is None:
                child.kill()
            child.wait()
    record = {'exit_code': child.returncode, 'kill_reason': reason, 'seconds': perf_counter()-started,
              'peak_sampled_rss_bytes': peak, 'wall_limit_seconds': args.seconds, 'memory_limit_bytes': limit,
              'execution_seconds_including_first_phase': perf_counter()-started+elapsed,
              'windows_job_commit_limit_enforced': release is not None, 'command': command}
    write_report(args.output/f'{prefix}_guard.json', record)
    print(json.dumps(record, indent=2), flush=True)
    return int(reason is not None or child.returncode != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=SECONDS)
    parser.add_argument('--memory-mib', type=int, default=MEMORY_MIB)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Resume after a guarded non-budget failure')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.seconds <= SECONDS or not 0 < args.memory_mib <= MEMORY_MIB or not 1 <= args.threads <= 8:
        parser.error('expected at most 1790 s, 3520 MiB, and eight BLAS threads')
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
