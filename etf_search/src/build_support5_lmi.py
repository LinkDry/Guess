"""Build the five-vertex facial LMI without storing the original moment matrices."""
import argparse
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
from itertools import combinations_with_replacement
import json
from math import gcd, lcm
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

from general_4_9_model import (N, NUMBER_OF_VARIABLES, apply_action, basis_blocks,
                              edge_action, transposition)
from level3_4_9_model import ExtendedMoments, cubic_basis, odd
from level3_face_model import BASE_FOCAL_COUNT, kernel_directions


def monomials(stage):
    return tuple(w for d in range(3) for w in combinations_with_replacement(range(NUMBER_OF_VARIABLES), d)) + tuple(cubic_basis(stage))


def tail_orbits(words, fixed):
    """One monomial per orbit under permutations fixing the first `fixed` vertices."""
    actions = [transposition(i, i+1) for i in range(fixed, N-1)]
    seen = set()
    for word in words:
        if word in seen:
            continue
        yield word
        seen.add(word)
        queue = [word]
        for current in queue:
            for action in actions:
                image, _ = apply_action(current, action)
                if image not in seen:
                    seen.add(image)
                    queue.append(image)


def cross_products(words, equations):
    """Yield (equation, multiplier, test word) spanning all structural face rows.

    A permutation sends the focal vertices of a kernel generator to their
    standard labels. For the equations used here, the tail group fixes the
    polynomial. Canonicalize the multiplier under this tail group. The
    smaller group fixing the multiplier's entire support then moves the
    test word, up to a sign, into one of the enumerated orbits.
    """
    from general_4_9_model import VARIABLES

    @lru_cache(maxsize=N)
    def representatives(fixed):
        return tuple(tail_orbits(words, fixed))

    word_set = set(words)
    low = tuple(w for w in words if len(w) <= 2)
    for name, fixed in BASE_FOCAL_COUNT.items():
        equation = equations[name]
        degree = max(map(len, equation))
        equation_parity = {odd(w) for w in equation}
        if len(equation_parity) != 1:
            raise ValueError(f'mixed conjugation parity for {name}')
        for i in range(fixed, N-1):
            action = transposition(i, i+1)
            transformed = {}
            for w, (a, b) in equation.items():
                image, sign = apply_action(w, action)
                transformed[image] = a*sign, b*sign
            if equation != transformed:
                raise ValueError(f'equation {name} not tail invariant')
        for multiplier in tail_orbits(tuple(w for w in low if len(w) <= 3-degree), fixed):
            support = set(range(fixed))
            for variable in multiplier:
                _, u, v = VARIABLES[variable]
                support.update((u, v))
            permutation = tuple(sorted(support)) + tuple(i for i in range(N) if i not in support)
            inverse = edge_action(tuple(permutation.index(i) for i in range(N)))
            multiplier, _ = apply_action(multiplier, inverse)
            shifted = [(tuple(sorted(w+multiplier)), c) for w,c in equation.items()]
            if any(w not in word_set for w, _ in shifted):
                continue
            parity = next(iter(equation_parity)) ^ odd(multiplier)
            for test in representatives(len(support)):
                if odd(test) == parity:
                    yield name, multiplier, test, shifted


def rational_row(shifted, test, orbits):
    """Generate a primitive integer row after the Round 7 beta rescaling."""
    result = defaultdict(int)
    tag = None
    for word, (a,b) in shifted:
        product = tuple(sorted(word+test))
        signed_index = orbits.orbit(product)
        if not signed_index:
            continue
        parity = len(product) % 2
        current = parity ^ bool(b)
        if (a and b) or (tag is not None and tag != current):
            raise ValueError('mixed beta parity in face equation')
        tag = current
        coeff = 5*a if a else (2 if parity else 5)*b
        result[abs(signed_index)-1] += coeff if signed_index>0 else -coeff
    ordered = tuple((i,c) for i,c in sorted(result.items()) if c)
    if not ordered:
        return ()
    divisor = gcd(*(c for _,c in ordered))
    if ordered[0][1]<0:
        divisor = -divisor
    return tuple((i,c//divisor) for i,c in ordered)


def psd_stencils(block, keep, orbits):
    """Yield the unweighted moment forms in SCS lower-column packing."""
    basis = block['basis']
    for col, j in enumerate(keep):
        for row in range(col,len(keep)):
            left = basis[keep[row]]
            representative, sign = left[0]
            weights = defaultdict(int)
            for word, weight in basis[j]:
                index = orbits.orbit(tuple(sorted(representative+word)))
                if index:
                    weights[abs(index)-1] += weight if index>0 else -weight
            scale = len(left)*sign
            yield tuple((i,scale*w) for i,w in sorted(weights.items()) if w)


def csr(rows, width):
    from array import array
    import numpy as np
    from scipy.sparse import csr_matrix

    index, values, bounds = array('i'), array('q'), array('i',[0])
    for row in rows:
        for j, c in row:
            index.append(j)
            values.append(c)
        bounds.append(len(index))
    return csr_matrix((np.frombuffer(values,dtype=np.int64),np.frombuffer(index,dtype=np.int32),
                       np.frombuffer(bounds,dtype=np.int32)),shape=(len(bounds)-1,width))


def model_inputs(args):
    import numpy as np
    from scipy.sparse import save_npz

    started = perf_counter()
    with gzip.open(args.face,'rt',encoding='utf-8') as stream:
        model = json.load(stream)
    old = tuple(map(tuple,model['moments']))
    del model
    from general_4_9_model import base_equations
    equations = base_equations()
    orbits = ExtendedMoments.from_representatives(old)
    words = monomials(args.stage)
    seen = {}
    generated = 0
    for name,multiplier,test,shifted in cross_products(words,equations):
        generated += 1
        row = rational_row(shifted,test,orbits)
        if row and row not in seen:
            seen[row] = [name,multiplier,test]
    print(f'{args.stage}: {generated} face row proposals, {len(seen)} unique nonzero rows, {len(orbits.representatives)} moments so far',flush=True)
    blocks = [{'character': key,'basis': basis} for key,basis in basis_blocks(words,split=True,fixed_vertices=int(args.stage[-1]))]
    directions, pivots = kernel_directions({'stage':args.stage,'base_equations':equations},blocks,high_first=True)
    congruences = []
    sizes=[]
    for block,pivot in zip(blocks,pivots,strict=True):
        keep = [i for i in range(len(block['basis'])) if i not in pivot]
        sizes.append(len(keep))
        weights = 1/np.sqrt([sum(v*v for _,v in block['basis'][k]) for k in keep])
        congruences.append(np.concatenate([w*weights[j:] for j,w in enumerate(weights)]))
    # All stencils may introduce additional moments. Assign the final width after
    # generating each sparse block. The shape is widened with zero columns.
    for i,(block,pivot,scale) in enumerate(zip(blocks,pivots,congruences,strict=True)):
        keep = [j for j in range(len(block['basis'])) if j not in pivot]
        def rows():
            yield from psd_stencils(block,keep,orbits)
        sparse = csr(rows(), len(old)+15000)
        sparse.resize(sparse.shape[0],len(orbits.representatives))
        save_npz(args.output/f'psd_map_{i}.npz',sparse,compressed=True)
        np.save(args.output/f'congruence_{i}.npy',scale,allow_pickle=False)
        print(f'PSD block {i}: {len(keep)} by {len(keep)}, {sparse.nnz} terms, {len(orbits.representatives)} moments so far',flush=True)
    width = len(orbits.representatives)
    save_npz(args.output/'cross_rows.npz',csr(seen,width),compressed=True)
    report = {'stage':args.stage,'block_sizes':sizes,'old_moments':len(old),'moments':orbits.representatives,
              'cross_row_proposals':generated,'unique_cross_rows':len(seen),'build_seconds':perf_counter()-started,
              'kernel_dimensions':[len(v) for v in directions]}
    (args.output/'model.json').write_text(json.dumps(report,separators=(',',':'))+'\n',encoding='utf-8')
    with gzip.open(args.output/'cross_provenance.json.gz','wt',encoding='utf-8') as stream:
        json.dump(list(seen.values()),stream,separators=(',',':'))
    del seen, blocks, directions, pivots, orbits
    ExtendedMoments._canonical_id.cache_clear()
    from level3_4_9_model import high_degree_canonical
    high_degree_canonical.cache_clear()
    print(f'Sparse model ready in {perf_counter()-started:.2f}s',flush=True)


def supervise(args):
    import psutil
    from solve_general_4_9 import THREADS, windows_job

    args.output.mkdir(parents=True, exist_ok=True)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    env = dict(os.environ, PYTHONUNBUFFERED='1', **{k: str(args.threads) for k in THREADS})
    if os.name == 'nt' and sys.prefix != sys.base_prefix:
        # Use the real interpreter so the worker itself joins the Job before
        # doing any model construction, bypassing the venv redirector.
        command[0] = sys._base_executable
        env['__PYVENV_LAUNCHER__'] = sys.executable
    start=perf_counter()
    cap=args.memory_mib*2**20
    peak=0
    reason=None
    with (args.output/f'{args.phase}_run.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command, env=env, stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
        try:
            release=windows_job(child,cap)
        except BaseException:
            child.kill()
            child.wait()
            raise
        try:
            monitor=psutil.Process(child.pid)
            child.stdin.write('start\n')
            child.stdin.close()
            while child.poll() is None:
                try:
                    rss=sum(p.memory_info().rss for p in [monitor,*monitor.children(recursive=True)])
                    peak=max(peak,rss)
                except psutil.NoSuchProcess:
                    pass
                if peak>=cap or perf_counter()-start>=args.seconds:
                    reason='memory_limit' if peak>=cap else 'wall_limit'
                    break
                sleep(0.05)
        finally:
            release()
            child.wait()
    record={'phase':args.phase,'exit_code':child.returncode,'kill_reason':reason,'wall_seconds':perf_counter()-start,
            'peak_sampled_rss_bytes':peak,'memory_limit_bytes':cap,'wall_limit_seconds':args.seconds,
            'windows_job_commit_limit_enforced':True,'command':command}
    (args.output/f'{args.phase}_guard.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(record,flush=True)
    return 1 if reason else child.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face',type=Path,required=True)
    parser.add_argument('--chart',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['support4','support5'],default='support5')
    parser.add_argument('--phase',choices=['model','reduce'],default='model')
    parser.add_argument('--seconds',type=float,default=900)
    parser.add_argument('--memory-mib',type=int,default=4000)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 0<args.memory_mib<4096 or not 0<args.seconds<=1800 or not 1<=args.threads<=8:
        parser.error('expected <4096 MiB, <=1800 s and at most eight threads')
    if not args.worker:
        return supervise(args)
    if sys.stdin.readline()!='start\n':
        return 2
    if args.phase=='model':
        model_inputs(args)
    else:
        from reduce_support5_lmi import reduce_lmi
        reduce_lmi(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
