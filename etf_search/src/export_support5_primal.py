"""Lift a five-point Newton candidate through the exact moment chart."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def export_primal(root, solution, output, denominator):
    with gzip.open(root/'affine_chart.json.gz','rt',encoding='utf-8') as stream:
        chart = json.load(stream)
    if chart['schema'] != 'support5-affine-chart-v1':
        raise ValueError('expected a five-point affine chart')
    rank = json.loads((root/'pencil_rank_certificate.json').read_text(encoding='utf-8'))
    retained = rank['retained_direction_columns']
    width = len(chart['free_columns'])
    if len(set(retained)) != len(retained) or any(not 0 <= j < width for j in retained):
        raise ValueError('invalid retained direction coordinates')
    with np.load(solution,allow_pickle=False) as archive:
        point = archive['x']
    if point.shape != (len(retained),) or not np.isfinite(point).all() or denominator <= 0:
        raise ValueError('expected a finite primal point and positive denominator')
    slots = [Fraction(1)] + [Fraction(0)]*width
    for j,value in zip(retained,point,strict=True):
        slots[j+1] = Fraction(round(float(value)*denominator),denominator)
    moments = []
    for (scale,terms),parity in zip(chart['rows'],chart['parity'],strict=True):
        value = sum((slots[j]*numerator for j,numerator in terms),Fraction(0))/scale
        moments.append(['0',str(value)] if parity else [str(value),'0'])
    certificate = {'schema':'anchored-exact-primal-v1', 'coefficient_basis':['1','sqrt(10)/5'],
                   'moments':moments,'source_chart_free_rounding_denominator':denominator,
                   'source_solution_sha256':hashlib.sha256(solution.read_bytes()).hexdigest()}
    output.write_text(json.dumps(certificate,separators=(',',':'))+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--solution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--denominator',type=int,default=10**14)
    args = parser.parse_args()
    export_primal(args.root,args.solution,args.output,args.denominator)


if __name__ == '__main__':
    main()
