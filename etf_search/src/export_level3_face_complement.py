"""Export and check the exact orthogonal complement of an archived face."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from level3_face_model import independent_insert, multiply_by_five, orthogonal_basis
from verify_sdp_level3 import canonical_psd_bases


def export_complements(manifest):
    blocks = []
    full_bases = canonical_psd_bases(manifest['stage'])
    for block in manifest['face_reduction']['blocks']:
        full_basis = full_bases[tuple(block['character'])]
        directions = [{int(i): tuple(pair) for i, pair in item['coordinates'].items()}
                      for item in block['directions']]
        echelon = {}
        for direction in directions:
            if not independent_insert(direction, echelon):
                raise ValueError('Dependent kernel direction in face manifest')
        vectors = orthogonal_basis({'basis': full_basis}, echelon)
        norms = [len(poly) for poly in full_basis]
        by_coordinate = defaultdict(list)
        for row, direction in enumerate(directions):
            for i, coeff in direction.items():
                by_coordinate[i].append((row, coeff))
        free = set(range(len(norms))) - echelon.keys()
        used = set()
        for vector in vectors:
            free_coordinates = vector.keys() & free
            if len(free_coordinates) != 1 or free_coordinates <= used:
                raise ValueError('Complement vectors do not have independent free coordinates')
            used.update(free_coordinates)
            products = defaultdict(lambda: [0, 0])
            for i, coeff in vector.items():
                for row, other in by_coordinate[i]:
                    a, b = multiply_by_five(coeff, other)
                    products[row][0] += a*norms[i]
                    products[row][1] += b*norms[i]
            if any(a or b for a, b in products.values()):
                raise ValueError('Complement vector is not orthogonal to the kernel')
        if used != free:
            raise ValueError('Incomplete orthogonal complement')
        blocks.append({'character': block['character'], 'original_size': len(norms),
                       'kernel_size': len(directions), 'vectors': vectors})
    return {'stage': manifest['stage'], 'coefficient_basis': ['1', 'sqrt(10)/5'],
            'coordinates': 'original character orbit-sum basis', 'orthogonality_checked_exactly': True,
            'blocks': blocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    model = json.loads(args.manifest.read_text(encoding='utf-8'))
    result = export_complements(model)
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    print(result['stage'], [len(block['vectors']) for block in result['blocks']])


if __name__ == '__main__':
    main()
