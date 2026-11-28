"""Report numerical rank and consistency of the equality part of a saved SDP."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import csc_matrix


def inspect(path):
    with np.load(path, allow_pickle=False) as data:
        cone = csc_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
        count = int(data['zero_cone'])
        scales = np.linalg.norm(cone[:count, :].toarray(), axis=1)
        normalized = cone[:count, :].toarray() / scales[:, None]
        right = data['b'][:count] / scales
    solution, _, rank, singular = lstsq(normalized, right, cond=1e-10, lapack_driver='gelsd')
    return {'equality_shape': list(normalized.shape), 'relative_rank_cutoff': 1e-10,
            'numeric_rank': int(rank), 'nullity': normalized.shape[1] - int(rank),
            'smallest_retained_singular': float(singular[rank-1]),
            'largest_discarded_singular': float(singular[rank]) if rank < len(singular) else None,
            'normalized_equality_residual_inf': float(np.max(abs(normalized @ solution-right)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    result = {run.name: inspect(run / 'solver_conic_data.npz') for run in args.runs}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
