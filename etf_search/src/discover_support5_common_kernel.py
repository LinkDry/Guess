"""Recover exact common kernel vectors exposed by the five-point Newton run."""
import argparse
from fractions import Fraction
from math import lcm

from flint import fmpq_mat
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.sparse import load_npz, csr_matrix

from build_support5_lmi import monomials
from general_4_9_model import base_equations, basis_blocks
from level3_face_model import kernel_directions
from reduce_support5_lmi import decode, exact_null_check
from solve_support5_barrier import PackedBlock


def discover(root, solution, output):
    with gzip.open(root/'affine_chart.json.gz','rt',encoding='utf-8') as stream:
        chart = json.load(stream)
    model = json.loads((root/'model.json').read_text(encoding='utf-8'))
    if model['stage'] != 'support5':
        raise ValueError('expected a five-point model')
    chart_rows = decode(chart['rows'],len(chart['free_columns'])+1)
    columns = list(zip(*chart_rows,strict=True))
    parity = np.asarray(chart['parity'])
    blocks = [{'character':char,'basis':basis}
              for char,basis in basis_blocks(monomials('support5'),split=True,fixed_vertices=5)]
    _, original_pivots = kernel_directions({'stage':'support5','base_equations':base_equations()},blocks,high_first=True)
    with np.load(solution,allow_pickle=False) as data:
        point = data['x']
    rank = json.loads((root/'pencil_rank_certificate.json').read_text(encoding='utf-8'))
    values = [Fraction(1)] + [Fraction(0)]*len(chart['free_columns'])
    for slot, value in zip(rank['retained_direction_columns'],point,strict=True):
        values[slot+1]=Fraction(round(float(value)*10**14),10**14)
    moment = [sum(a*b for a,b in zip(row,values,strict=True)) for row in chart_rows]
    common = lcm(*(v.denominator for v in moment))
    integers = [int(v*common) for v in moment]
    report = []
    for k, (block, pivots, size) in enumerate(zip(blocks,original_pivots,model['block_sizes'],strict=True)):
        basis_parity = np.asarray([len(poly[0][0])%2 for j,poly in enumerate(block['basis']) if j not in pivots])
        with np.load(root/'pencil'/f'block_{k}.npz',allow_pickle=False) as data:
            packed = data['b'] + data['A'] @ point
        helper = PackedBlock(size)
        scale = (np.sqrt(10)/5)**basis_parity
        numerical = helper.unpack(packed)/scale[None,:]/scale[:,None]
        spectrum, basis = np.linalg.eigh(numerical)
        if spectrum[0] < -1e-8:
            raise ArithmeticError(f'block {k} is numerically indefinite')
        nullity = int(np.count_nonzero(spectrum < 1e-8))
        if not nullity:
            report.append({'size':size,'basis_parity':basis_parity.tolist(),'pivot_indices':[],'vectors':[]})
            continue
        kernel = basis[:,:nullity]
        _, _, ordering = qr(kernel.T,pivoting=True,mode='economic')
        pivot_indices = sorted(map(int,ordering[:nullity]))
        matrix = load_npz(root/f'psd_map_{k}.npz').tocsr()
        matrix.resize(matrix.shape[0],len(parity))
        # Conjugating by diag(beta^-parity) rationalizes the pencil.
        # The extra factor of two clears beta^-2=5/2 in the odd/odd part.
        doubled = np.where(basis_parity[helper.row] & basis_parity[helper.column],5,2)
        entries = [[0]*size for _ in range(size)]
        for i,(r,c,factor) in enumerate(zip(helper.row,helper.column,doubled,strict=True)):
            a,b = matrix.indptr[i:i+2]
            entry = sum(int(v)*integers[j] for j,v in zip(matrix.indices[a:b],matrix.data[a:b],strict=True))*int(factor)
            entries[r][c] = entries[c][r] = entry
        keep = sorted(set(range(size)) - set(pivot_indices))
        restricted = fmpq_mat(len(keep),len(keep),[entries[i][j] for i in keep for j in keep])
        rhs = fmpq_mat(len(keep),nullity,[-entries[i][j] for i in keep for j in pivot_indices])
        solution = restricted.solve(rhs,algorithm='dixon')
        reconstructed = [[Fraction(0)]*size for _ in range(nullity)]
        for k_vector, pivot in enumerate(pivot_indices):
            reconstructed[k_vector][pivot] = Fraction(1)
            for k_row, j in enumerate(keep):
                reconstructed[k_vector][j] = Fraction(str(solution[k_row,k_vector]))
        stencils = [[{} for _ in range(size)] for _ in reconstructed]
        for position,(row,col,factor) in enumerate(zip(helper.row,helper.column,doubled,strict=True)):
            for j,vector in enumerate(reconstructed):
                for dest,src in ((row,col), (col,row)) if row != col else ((row,col),):
                    if not vector[src]:
                        continue
                    start,stop=matrix.indptr[position:position+2]
                    q=stencils[j][dest]
                    for i,c in zip(matrix.indices[start:stop],matrix.data[start:stop],strict=True):
                        if parity[i] != (basis_parity[row] ^ basis_parity[col]):
                            raise ValueError('basis parity disagrees with moment parity')
                        q[i]=q.get(i,Fraction(0))+int(c)*int(factor)*vector[src]
        col_ids, val, offsets = [], [], [0]
        for rows in stencils:
            for row in rows:
                den=lcm(*(x.denominator for x in row.values()))
                for i,c in sorted(row.items()):
                    if c:
                        col_ids.append(i)
                        val.append(int(c*den))
                offsets.append(len(val))
        contraction=csr_matrix((np.asarray(val,dtype=np.int64),col_ids,offsets),shape=(nullity*size,len(parity)))
        exact_null_check(contraction,columns)
        print(f'block {k}: verified {nullity} common kernel vectors against {len(columns)} affine coefficients',flush=True)
        report.append({'size':size,'basis_parity':basis_parity.tolist(),'pivot_indices':pivot_indices,
                       'vectors':[[[i,str(c)] for i,c in enumerate(vector) if c] for vector in reconstructed],
                       'smallest_nonzero_eigenvalue_approx':float(spectrum[nullity])})
    output.write_text(json.dumps({'schema':'support5-common-kernel-v1','blocks':report},indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--solution',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    discover(args.root,args.solution,args.output)


if __name__=='__main__':
    main()
