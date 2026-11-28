import Mathlib

namespace ETF49.FixedOne

abbrev Vec3 (K : Type*) := Fin 3 → K
abbrev Vec4 (K : Type*) := Fin 4 → K
abbrev Mat34 (K : Type*) := Fin 3 → Fin 4 → K

section Coordinates

variable {K : Type*} [Field K] [CharZero K]

def dot3 (u v : Vec3 K) : K := u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def dot4 (u v : Vec4 K) : K := u 0 * v 0 + u 1 * v 1 + u 2 * v 2 + u 3 * v 3

def norm3 (v : Vec3 K) : K := dot3 v v

def norm4 (v : Vec4 K) : K := dot4 v v

def mulVec (A : Mat34 K) (h : Vec4 K) : Vec3 K := fun i => dot4 (A i) h

def tmulVec (A : Mat34 K) (z : Vec3 K) : Vec4 K := fun j => dot3 (fun i => A i j) z

private def δ {n : Nat} (i j : Fin n) : K := if i = j then 1 else 0

/-- T = XXᵀ + YYᵀ − (9/4)I. -/
def tight (X Y : Mat34 K) (i j : Fin 3) : K :=
  dot4 (X i) (X j) + dot4 (Y i) (Y j) - (9 / 4) * δ i j

/-- E = WᵀW + (5/16)YᵀY − (27/32)I. -/
def bivector (Y W : Mat34 K) (i j : Fin 4) : K :=
  dot3 (fun k => W k i) (fun k => W k j) +
    (5 / 16) * dot3 (fun k => Y k i) (fun k => Y k j) - (27 / 32) * δ i j

def quad3 (A : Fin 3 → Fin 3 → K) (z : Vec3 K) : K :=
  dot3 z (fun i => dot3 (A i) z)

def quad4 (A : Fin 4 → Fin 4 → K) (h : Vec4 K) : K :=
  dot4 h (fun i => dot4 (A i) h)

def residual (h q : Vec4 K) : Vec4 K := fun j => (9 / 4) * h j - q j

private theorem transpose_pairing (A : Mat34 K) (h : Vec4 K) (z : Vec3 K) :
    dot4 h (tmulVec A z) = dot3 (mulVec A h) z := by
  simp only [tmulVec, mulVec, dot3, dot4]
  ring

private theorem expand_residual (h q : Vec4 K) :
    norm4 (residual h q) = (9 / 4 : K)^2 * norm4 h -
      2 * (9 / 4) * dot4 h q + norm4 q := by
  simp only [norm4, residual, dot4]
  ring

set_option maxHeartbeats 1000000 in
private theorem expand_tight (X Y : Mat34 K) (z : Vec3 K) :
    quad3 (tight X Y) z =
      norm4 (tmulVec X z) + norm4 (tmulVec Y z) - (9 / 4 : K) * norm3 z := by
  simp (config := { decide := true }) [quad3, tight, tmulVec, norm3, norm4, dot3, dot4, δ] <;> ring

set_option maxHeartbeats 1000000 in
private theorem expand_bivector (Y W : Mat34 K) (h : Vec4 K) :
    quad4 (bivector Y W) h =
      norm3 (mulVec W h) + (5 / 16 : K) * norm3 (mulVec Y h) - (27 / 32) * norm4 h := by
  simp (config := { decide := true }) [quad4, bivector, mulVec, norm3, norm4, dot3, dot4, δ] <;> ring

/-- The two length-four sums contribute the eight rank-one SOS blocks. -/
def certificateRhs (X Y W : Mat34 K) (h : Vec4 K) : K :=
  let z := mulVec Y h
  (80 / 81) * (norm4 (residual h (tmulVec Y z)) + norm4 (tmulVec X z)) +
    (norm4 h - 1) + (64 / 9) * (quad4 (bivector Y W) h - norm3 (mulVec W h)) -
    (80 / 81) * quad3 (tight X Y) z

/-- The archived SOS polynomial identity with exact rational constants. The
identity holds for arbitrary W and hence for columns Wⱼ = xⱼ × yⱼ. -/
theorem certificate_identity (X Y W : Mat34 K) (h : Vec4 K) :
    (-1 : K) = certificateRhs X Y W h := by
  dsimp only [certificateRhs]
  rw [expand_residual, expand_bivector, expand_tight, transpose_pairing]
  ring

end Coordinates

private theorem norm4_nonneg (v : Vec4 ℝ) : 0 ≤ norm4 v := by
  simpa only [norm4, dot4, pow_two] using
    (add_nonneg (add_nonneg (add_nonneg (sq_nonneg (v 0)) (sq_nonneg (v 1)))
      (sq_nonneg (v 2))) (sq_nonneg (v 3)))

/-- The lifted real system for one fixed line has no solution. Its geometric
application supplies T, E, and a normalized kernel vector of W. -/
theorem no_real_solution (X Y W : Mat34 ℝ) (h : Vec4 ℝ)
    (hT : ∀ i j, tight X Y i j = 0) (hE : ∀ i j, bivector Y W i j = 0)
    (hW : ∀ i, mulVec W h i = 0) (hh : norm4 h = 1) : False := by
  have hTzero : quad3 (tight X Y) (mulVec Y h) = 0 := by simp [quad3, dot3, hT]
  have hEzero : quad4 (bivector Y W) h = 0 := by simp [quad4, dot4, hE]
  have hWzero : norm3 (mulVec W h) = 0 := by simp [norm3, dot3, hW]
  have hc := certificate_identity X Y W h
  dsimp only [certificateRhs] at hc
  rw [hTzero, hEzero, hWzero, hh] at hc
  have hNonneg : 0 ≤ (80 / 81 : ℝ) *
      (norm4 (residual h (tmulVec Y (mulVec Y h))) + norm4 (tmulVec X (mulVec Y h))) :=
    mul_nonneg (by norm_num) (add_nonneg (norm4_nonneg _) (norm4_nonneg _))
  linarith

end ETF49.FixedOne
