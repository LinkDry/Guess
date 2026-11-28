"""Smoke tests for the guarded Round 7 PSD-only solver and its output contract."""
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.sparse import csc_matrix


RUNNER = Path(__file__).resolve().parents[1] / 'src' / 'solve_level3_lmi_4_9.py'


def save_conic(path, a, b, q, sizes, **extra):
    matrix = csc_matrix(a, dtype=float)
    np.savez(path, A_data=matrix.data, A_indices=matrix.indices, A_indptr=matrix.indptr,
             A_shape=matrix.shape, b=b, q=q, block_sizes=sizes, **extra)


class LmiSolverTests(unittest.TestCase):
    def run_backend(self, backend, source, output, *extra, check=True):
        completed = subprocess.run(
            [sys.executable, '-B', str(RUNNER), '--backend', backend, '--input', str(source),
             '--output', str(output), *extra], capture_output=True, text=True, timeout=40)
        if check:
            log = output / 'solver_run.log'
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr +
                             (log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''))
        return completed

    def test_both_solve_in_same_order_and_save_raw_vectors(self):
        # SCS order: (00, 10, 20, 11, 21, 22) then a scalar x + 2.
        # The objective maximizes x over a 3x3 block with |x| <= sqrt(2).
        for backend in ('clarabel', 'scs'):
            if find_spec(backend) is None:
                continue
            with self.subTest(backend=backend), TemporaryDirectory() as tmp:
                root = Path(tmp)
                b = np.array([1., 0., 0., 2., 0., 4., 2.])
                a = np.array([[0.], [-np.sqrt(2)], [0.], [0.], [0.], [0.], [-1.]])
                source, output = root/'conic.npz', root/'out'
                save_conic(source, a, b, [-1.], [3, 1],
                           psd_congruence=[4., 6., 10., 9., 15., 25., 4.])
                self.run_backend(backend, source, output, '--psd-congruence')
                with np.load(output/'solver_primal_dual.npz', allow_pickle=False) as vectors:
                    self.assertAlmostEqual(vectors['x'][0], np.sqrt(2), places=6)
                    np.testing.assert_allclose(a @ vectors['x'] + vectors['s'], b, atol=1e-7)
                    np.testing.assert_allclose(a.T @ vectors['y'], [1.], atol=1e-7)
                    order = [0, 1, 3, 2, 4, 5, 6] if backend == 'clarabel' else list(range(7))
                    self.assertEqual(vectors['s_backend'][5], vectors['s'][5] * 25.)
                    np.testing.assert_array_equal(vectors['backend_order'], order)
                summary = json.loads((output/'solver_summary.json').read_text())
                self.assertEqual(summary['numerical_status'], 'feasible')
                self.assertEqual(summary['psd_blocks'], [3, 1])
                self.assertTrue(summary['diagnostics_complete'])
                guard = json.loads((output/'solver_guard.json').read_text())
                self.assertEqual(guard['wall_limit_seconds'], 900)
                self.assertEqual(guard['memory_limit_bytes'], 4 * 1024**3)
                self.assertTrue(guard['hard_memory_limit_enforced'])
                first_log_line = (output/'solver_run.log').read_text().splitlines()[0]
                self.assertIn(f"Python worker PID {guard['worker_pid']},", first_log_line)
                if backend == 'scs':
                    self.assertEqual(summary['settings']['max_iters'], 20_000)
                    self.assertEqual(summary['settings']['eps_abs'], 1e-9)
                    self.assertEqual(summary['settings']['eps_rel'], 1e-9)

    def test_infeasible_scalar_psd_blocks_preserve_native_ray_and_nan_iterate(self):
        if find_spec('scs') is None:
            self.skipTest('SCS unavailable')
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/'conic.npz', root/'out'
            save_conic(source, [[1.], [-1.]], [0., -1.], [0.], [1, 1])
            self.run_backend('scs', source, output)
            with np.load(output/'solver_primal_dual.npz', allow_pickle=False) as vectors:
                self.assertTrue(np.isnan(vectors['x']).all())
                np.testing.assert_allclose(vectors['y'], [1., 1.], atol=1e-8)
                np.testing.assert_array_equal(vectors['y'], vectors['y_backend'])
            summary = json.loads((output/'solver_summary.json').read_text())
            self.assertEqual(summary['numerical_status'], 'infeasible')
            self.assertAlmostEqual(summary['diagnostics']['dual_infeasibility_ray']['b_dot_y'], -1.)

    def test_watchdog_stops_worker_at_short_deadline(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/'conic.npz', root/'out'
            save_conic(source, [[-1.]], [1.], [0.], [1])
            completed = self.run_backend('scs', source, output, '--seconds', '0.005', check=False)
            self.assertNotEqual(completed.returncode, 0)
            guard = json.loads((output/'solver_guard.json').read_text())
            self.assertEqual(guard['kill_reason'], 'wall_limit')
            self.assertEqual(guard['wall_limit_seconds'], 0.005)
            self.assertFalse((output/'solver_primal_dual.npz').exists())

    def test_rejects_invalid_psd_row_count_before_solver_setup(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/'conic.npz', root/'out'
            save_conic(source, [[1.], [-1.]], [0., -1.], [0.], [3])
            completed = self.run_backend('clarabel', source, output, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('PSD row count', (output/'solver_run.log').read_text())
            self.assertFalse((output/'solver_primal_dual.npz').exists())
            self.assertTrue((output/'solver_guard.json').exists())

    def test_rejects_scaling_that_is_not_a_congruence(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/'conic.npz', root/'out'
            save_conic(source, [[1.], [0.], [0.]], [1., 0., 2.], [0.], [2],
                       psd_congruence=[4., 3., 9.])
            completed = self.run_backend('clarabel', source, output, '--psd-congruence', check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('diagonal PSD congruence', (output/'solver_run.log').read_text())


if __name__ == '__main__':
    unittest.main()
