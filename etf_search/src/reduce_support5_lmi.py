"""Exact second-stage equality and PSD-image reduction of the five-point model."""
from fractions import Fraction
import gzip
import hashlib
import json
from math import lcm
from time import perf_counter

from flint import fmpq_mat
import numpy as np
from scipy.linalg import qr
from scipy.sparse import load_npz
from sympy import nextprime


ZERO = Fraction(0)


def decode(rows, width):
    return [[{slot: Fraction(num, den) for slot, num in terms}.get(j, ZERO) for j in range(width)]
            for den, terms in rows]


def encode(rows):
    answer = []
    for row in rows:
        denominator = lcm(*(value.denominator for value in row))
        answer.append([denominator, [[j, int(value*denominator)] for j, value in enumerate(row) if value]])
    return answer


def multiply_selected(matrix, chart, row_indices):
    return [[sum((int(c)*chart[j][k] for j,c in zip(row.indices,row.data,strict=True)), ZERO)
             for k in range(len(chart[0]))] for row in (matrix.getrow(int(i)) for i in row_indices)]


def choose_rows(array, threshold=2e-9):
    _, triangle, order = qr(array.T, mode='economic', pivoting=True, check_finite=False)
    diagonal = abs(triangle.diagonal())
    rank = int(np.count_nonzero(diagonal > threshold*max(1.,max(diagonal,default=0.))))
    return order[:rank]


def exact_null_check(matrix, directions):
    """Verify all sparse products exactly, by bounded integer arithmetic or CRT.

    Each rational column is cleared separately. If a dot product can exceed
    int64, reduce the integer coordinates modulo enough distinct small primes
    that their product exceeds the absolute bound on every integer dot product.
    """
    if not directions:
        return 0
    row_bound = int(np.max(np.asarray(abs(matrix).sum(axis=1)), initial=0))
    columns = []
    maximum = 0
    for direction in directions:
        denominator = lcm(*(x.denominator for x in direction))
        values = [int(denominator*x) for x in direction]
        maximum = max(maximum, row_bound*max(map(abs,values),default=0))
        columns.append(values)
    ordinary = maximum < 2**63
    if ordinary:
        moduli = [None]
    else:
        assert row_bound*2**26 < 2**63
        moduli, product, prime = [], 1, 2**25
        while product <= maximum:
            prime = int(nextprime(prime))
            if prime >= 2**26:
                raise ArithmeticError('ran out of bounded CRT moduli')
            moduli.append(prime)
            product *= prime
    for modulus in moduli:
        for begin in range(0, len(columns), 16):
            batch = columns[begin:begin+16]
            if modulus:
                batch = [[v % modulus for v in col] for col in batch]
            dots = matrix @ np.ascontiguousarray(np.asarray(batch,dtype=np.int64).T)
            if np.any(dots if modulus is None else dots % modulus):
                raise ArithmeticError(f'exact sparse equality check failed around column {begin}')
    return len(moduli) if not ordinary else 0


def solve_equalities(matrix, prior, original_free):
    """Solve a nonsingular additional pivot system and check all new rows."""
    approximations = matrix @ np.asarray(prior,dtype=float)
    selected = choose_rows(approximations[:,1:])
    _, _, column_order = qr(approximations[selected,1:],mode='economic',pivoting=True,check_finite=False)
    rank = len(selected)
    pivots = sorted(map(int,column_order[:rank]))
    free = sorted(set(range(len(original_free)))-set(pivots))
    exact = multiply_selected(matrix,prior,selected)
    if rank:
        left = fmpq_mat(rank,rank,[str(row[j+1]) for row in exact for j in pivots])
        rhs = fmpq_mat(rank,len(free)+1,[str(-row[j]) for row in exact for j in [0,*[k+1 for k in free]]])
        solution = left.solve(rhs,algorithm='dixon')
        table = {j: [Fraction(str(solution[i,k])) for k in range(len(free)+1)] for i,j in enumerate(pivots)}
    else:
        table = {}
    for k,j in enumerate(free,1):
        table[j] = [Fraction(int(k==i)) for i in range(len(free)+1)]
    supports = [[(k-1, value) for k,value in enumerate(row) if k and value] for row in prior]
    print(f'{rank} new pivots and {len(free)} free variables. Composing exact charts.',flush=True)
    rows = []
    for i,support in enumerate(supports):
        composed = [prior[i][0]] + [ZERO]*len(free)
        for j,v in support:
            for k,w in enumerate(table[j]):
                if w:
                    composed[k] += v*w
        rows.append(composed)
    moduli = exact_null_check(matrix,list(zip(*rows,strict=True)))
    return rows,[original_free[j] for j in free], {'additional_rank':rank,'selected_cross_rows':selected.tolist(),
              'eliminated_moment_coordinates':[original_free[j] for j in pivots], 'cross_check_crt_moduli':moduli}


def rank_psd(maps, rows, parity):
    """Calculate exact PSD direction rank using a nonsingular minor and exact kernels."""
    approximation = np.asarray(rows,dtype=float)[:,1:]
    previous = np.zeros((0,approximation.shape[1]))
    witnesses = []
    for index,matrix in enumerate(maps):
        # Moment entry homogeneous parity lets us remove a nonzero row scale.
        for j in range(matrix.shape[0]):
            ids = matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]]
            if len(ids) and np.any(parity[ids] != parity[ids[0]]):
                raise ValueError('mixed beta parity in PSD entry')
        for begin in range(0,matrix.shape[0],8192):
            values = matrix[begin:begin+8192] @ approximation
            candidate = np.vstack((previous,values))
            selected = choose_rows(candidate)
            witnesses = [witnesses[j] if j<len(previous) else (index,int(begin+j-len(previous))) for j in selected]
            previous = candidate[selected]
        print(f'PSD numeric image rank after block {index}: {len(previous)}',flush=True)
    rank = len(previous)
    if not rank:
        raise ValueError('expected a nonconstant pencil')
    _, _, order = qr(previous,mode='economic',pivoting=True,check_finite=False)
    selected = sorted(map(int,order[:rank]))
    rejected = sorted(set(range(approximation.shape[1]))-set(selected))
    exact = [multiply_selected(maps[index],rows,[i])[0][1:] for index,i in witnesses]
    minor = fmpq_mat(rank,rank,[str(row[j]) for row in exact for j in selected])
    rest = fmpq_mat(rank,len(rejected),[str(row[j]) for row in exact for j in rejected])
    coefficients = minor.solve(rest,algorithm='dixon')
    columns = list(zip(*rows,strict=True))[1:]
    relations = []
    checks = []
    for i,j in enumerate(rejected):
        factors = [Fraction(str(coefficients[k,i])) for k in range(rank)]
        nonzero = [(s,v) for s,v in zip(selected,factors,strict=True) if v]
        relations.append([j,[[s,str(v)] for s,v in nonzero]])
        checks.append([value - sum((v*columns[s][k] for s,v in nonzero if columns[s][k]),ZERO)
                       for k,value in enumerate(columns[j])])
    moduli = [exact_null_check(matrix,checks) for matrix in maps]
    return selected,{'rank':rank,'retained_direction_columns':selected,'witness_packed_rows':witnesses,
                     'discarded_as_retained_combinations':relations,'psd_check_crt_moduli':moduli}


def reduce_lmi(args):
    start=perf_counter()
    out=args.output
    model=json.loads((out/'model.json').read_text(encoding='utf-8'))
    raw_chart=gzip.decompress(args.chart.read_bytes())
    old_chart=json.loads(raw_chart)
    if hashlib.sha256(gzip.decompress(args.face.read_bytes())).hexdigest() != old_chart['model_sha256']:
        raise ValueError('Round 7 chart hash mismatch')
    old_size=model['old_moments']
    moments=model['moments']
    parity=np.asarray([len(word)%2 for word in moments])
    if old_size!=len(old_chart['rows']) or parity[:old_size].tolist()!=old_chart['parity']:
        raise ValueError('Round 7 chart coordinate mismatch')
    new_moments=len(moments)-old_size
    old_free=old_chart['free_columns']
    prior=decode(old_chart['rows'],1+len(old_free))
    if new_moments:
        prior=[row+[ZERO]*new_moments for row in prior]
        prior += [[Fraction(int(j==1+len(old_free)+i)) for j in range(len(prior[0]))] for i in range(new_moments)]
    equality=load_npz(out/'cross_rows.npz').tocsr()
    rows,free,elimination=solve_equalities(equality,prior,old_free+list(range(old_size,len(moments))))
    with gzip.open(out/'affine_chart.json.gz','wt',encoding='utf-8') as stream:
        json.dump({'schema':'support5-affine-chart-v1', 'source_chart_sha256':hashlib.sha256(raw_chart).hexdigest(),
                   'model_sha256':hashlib.sha256((out/'model.json').read_bytes()).hexdigest(),
                   'parity':parity.tolist(),'rows':encode(rows),'free_columns':free,**elimination},stream,separators=(',',':'))
    sizes=model['block_sizes']
    maps=[]
    for j in range(len(sizes)):
        matrix=load_npz(out/f'psd_map_{j}.npz').tocsr()
        matrix.resize(matrix.shape[0],len(moments))
        maps.append(matrix)
    keep,rank=rank_psd(maps,rows,parity)
    (out/'pencil_rank_certificate.json').write_text(json.dumps(rank,separators=(',',':'))+'\n',encoding='utf-8')
    rank_count=len(keep)
    # Estimate four dense copies of the largest block stencil, plus compact work.
    estimated=4*max(sizes)**2*(rank_count+1)*8+(rank_count+1)**2*8+256*2**20
    report={'stage':model['stage'],'moments':len(moments),'additional_equality_rank':elimination['additional_rank'],
            'equality_rank':len(moments)-len(free),'free_dimension':len(free),'psd_image_rank':rank_count,
            'block_sizes':sizes,'packed_psd_rows':sum(size*(size+1)//2 for size in sizes),
            'estimated_blockwise_bytes':estimated,'reduction_seconds':perf_counter()-start,
            'budget_gate_passed':estimated<args.memory_mib*2**20}
    print(report,flush=True)
    (out/'reduction_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not report['budget_gate_passed']:
        return
    packed=out/'pencil'
    packed.mkdir(exist_ok=True)
    coordinates=np.asarray(rows,dtype=float)[:,[0,*[j+1 for j in keep]]]
    coordinates[parity==1] *= np.sqrt(10)/5
    for k,(size,matrix) in enumerate(zip(sizes,maps,strict=True)):
        block=matrix@coordinates
        c,r=np.triu_indices(size)
        block[c!=r] *= np.sqrt(2)
        scale=np.load(out/f'congruence_{k}.npy',allow_pickle=False)
        np.savez_compressed(packed/f'block_{k}.npz', A=block[:,1:],b=block[:,0],scale=scale)
    (packed/'manifest.json').write_text(json.dumps({**report,'free_dimension':rank_count,
                   'retained_direction_columns':keep,'full_chart_dimension':len(free)},indent=2)+'\n',encoding='utf-8')
