"""Tests for the streaming common-margin barrier solver."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import solve_support5_barrier as barrier


def save_pencil(path, blocks, sizes, start=0):
    path.mkdir()
    (path/'manifest.json').write_text(json.dumps({
        'free_dimension': len(blocks[0][0][0]), 'block_sizes': sizes,
    }), encoding='utf-8')
    for index, (a, b, scale) in enumerate(blocks, start):
        np.savez_compressed(path/f'block_{index}.npz', A=a, b=b, scale=scale)


class StreamingBarrierTests(unittest.TestCase):
    def run_solver(self, source, output, *extra, check=True):
        completed = subprocess.run(
            [sys.executable, '-B', str(SRC/'solve_support5_barrier.py'), '--input',
             str(source), '--output', str(output), '--seconds', '20', *extra],
            capture_output=True, text=True, timeout=30,
        )
        if check:
            log = output/'solver_run.log'
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr +
                             (log.read_text(encoding='utf-8') if log.exists() else ''))
        return completed

    def test_batched_block_derivatives_match_direct_formulas(self):
        rng = np.random.default_rng(44)
        size, rank = 4, 5
        a = rng.normal(size=(10, rank))
        d = np.array([1., 1.5, 2., 2.5])
        vectors = [a[:, j] for j in range(rank)]
        lower = np.tril(rng.normal(scale=0.2, size=(size, size))) + np.eye(size)
        block = barrier.PackedBlock(size)
        scale = block.pack(np.outer(d, d))
        scale[block.off_diagonal] /= np.sqrt(2)
        r = np.triu(rng.normal(scale=.1, size=(rank, rank))) + np.eye(rank)
        expected = [d[:, None] * block.unpack(v) * d for v in vectors]
        transformed = np.linalg.solve(r.T, np.array(expected).reshape(rank, -1))
        stencils = [*transformed.reshape(rank, size, size), -np.eye(size)]
        sinv = np.linalg.inv(lower @ lower.T)
        gradient = -np.array([np.trace(sinv @ item) for item in stencils])
        hessian = np.array([[np.trace(sinv @ u @ sinv @ v) for u in stencils]
                            for v in stencils])
        for batch in [1, 2, rank+1]:
            with self.subTest(batch=batch):
                g, h = barrier.block_derivatives(block, a, scale, lower, r, batch)
                np.testing.assert_allclose(g, gradient, atol=2e-12)
                np.testing.assert_allclose(h, hessian, atol=2e-12)

    def test_packed_column_order_matches_scs(self):
        s = np.sqrt(2)
        block = barrier.PackedBlock(3)
        np.testing.assert_allclose(block.unpack([1, 2*s, 3*s, 4, 5*s, 6]),
                                   [[1, 2, 3], [2, 4, 5], [3, 5, 6]])

    def test_condition_pencil_computes_least_squares_center(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_pencil(root/'pencil', [([[1], [-1], [3]], [-4, 0, 8], [1, 1, 1])], [2])
            with barrier.Pencil(root/'pencil', root/'scratch') as pencil:
                center, triangle = barrier.condition_pencil(pencil)
                self.assertAlmostEqual(center[0], -20/11)
                self.assertAlmostEqual(abs(triangle[0, 0]), np.sqrt(11))
            self.assertFalse((root/'scratch').exists())

    def test_guarded_feasible_positive_sign_and_infeasible_dual(self):
        for constant, expected in [(4., 'primal_candidate'), (-2., 'dual_candidate')]:
            with self.subTest(constant=constant), TemporaryDirectory() as tmp:
                root = Path(tmp)
                pencil, output = root/'pencil', root/'out'
                # First block [[x, 0.25*x], [0.25*x, 1]], second scalar constant - x.
                blocks = [([[1.], [.25*np.sqrt(2)], [0.]], [0., 0., 1.], [4., 6., 9.]),
                          ([[-1.]], [constant], [0.5])]
                save_pencil(pencil, blocks, [2, 1], start=1)
                self.run_solver(pencil, output, '--batch-directions', '1')
                summary = json.loads((output/'solver_summary.json').read_text())
                guard = json.loads((output/'solver_guard.json').read_text())
                self.assertEqual(summary['status'], expected)
                self.assertTrue(guard['hard_memory_limit_enforced'])
                if os.name == 'nt':
                    self.assertTrue(guard['windows_job_commit_limit_enforced'])
                self.assertLess(guard['memory_limit_bytes'], 4*1024**3)
                self.assertLessEqual(guard['wall_limit_seconds'], 1800)
                self.assertIsNone(guard['kill_reason'])
                self.assertFalse((output/'pencil_scratch').exists())
                with np.load(output/'solver_primal_dual.npz') as vectors:
                    x, y, slack = (vectors[key] for key in ['x', 'y', 's'])
                    np.testing.assert_array_equal(vectors['block_sizes'], [2, 1])
                a = np.concatenate([np.asarray(block[0]) for block in blocks])
                b = np.concatenate([block[1] for block in blocks])
                np.testing.assert_allclose(slack, b + a @ x)
                self.assertAlmostEqual(b @ y, summary['dual_b_dot_y'], places=12)
                self.assertAlmostEqual(np.max(np.abs(a.T @ y)),
                                       summary['dual_stationarity_inf_unscaled'], places=12)
                self.assertGreaterEqual(min(summary['dual_block_min_eigenvalues_unscaled']), -1e-12)
                if constant > 0:
                    self.assertGreater(min(summary['primal_block_min_eigenvalues_unscaled']), 0)
                else:
                    self.assertLess(b @ y, -0.1)
                    self.assertLess(np.max(np.abs(a.T @ y)), 1e-4 * abs(b @ y))

    def test_feasible_warm_start_remains_in_original_coordinates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_pencil(root/'pencil', [([[1.]], [1.], [1.]), ([[-1.]], [1.], [1.])], [1, 1])
            np.savez(root/'warm.npz', x=[0.375])
            self.run_solver(root/'pencil', root/'out', '--warm-start', str(root/'warm.npz'))
            summary = json.loads((root/'out'/'solver_summary.json').read_text())
            self.assertEqual(summary['termination'], 'initial_point')
            self.assertEqual(summary['newton_steps'], 0)
            with np.load(root/'out'/'solver_primal_dual.npz') as vectors:
                np.testing.assert_array_equal(vectors['x'], [0.375])
                np.testing.assert_array_equal(vectors['s'], [1.375, 0.625])

    @unittest.skipUnless(os.name == 'nt', 'Windows Job wall deadline')
    def test_guard_kills_a_worker_that_outlives_deadline(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_worker = root/'slow_worker.py'
            fake_worker.write_text('import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n')
            args = barrier.parse_args(['--input', str(root/'missing'), '--output', str(root/'out'),
                                       '--seconds', '0.5'])
            with patch.object(barrier, '__file__', str(fake_worker)):
                self.assertNotEqual(barrier.supervise(args), 0)
            guard = json.loads((root/'out'/'solver_guard.json').read_text())
            self.assertEqual(guard['kill_reason'], 'wall_limit')
            self.assertTrue(guard['windows_job_commit_limit_enforced'])
            self.assertLess(guard['wall_seconds'], 3)

    def test_rejects_invalid_scaling_and_budget(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_pencil(root/'pencil', [([[1], [2], [3]], [0, 0, 0], [4, 8, 9])], [2])
            process = self.run_solver(root/'pencil', root/'out', check=False)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('not a diagonal PSD congruence', (root/'out'/'solver_run.log').read_text())
            for options in [('--memory-mib', '4096'), ('--seconds', '1801')]:
                process = self.run_solver(root/'pencil', root/'unused', *options, check=False)
                self.assertEqual(process.returncode, 2)
            self.assertFalse((root/'unused').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows Job assignment')
    def test_guard_fails_closed_without_windows_job(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'out'
            with patch.object(barrier, 'windows_job', side_effect=OSError('test failure')):
                args = barrier.parse_args(['--input', str(output/'missing'), '--output', str(output)])
                self.assertNotEqual(barrier.supervise(args), 0)
            guard = json.loads((output/'solver_guard.json').read_text())
            self.assertFalse(guard['windows_job_commit_limit_enforced'])
            self.assertEqual(guard['kill_reason'], 'guard_error')
            self.assertFalse((output/'solver_primal_dual.npz').exists())


if __name__ == '__main__':
    unittest.main()
