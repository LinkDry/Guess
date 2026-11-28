import Mathlib

noncomputable section

namespace ETF49

section Signature

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- ETF signature obtained from a Gram matrix by subtracting its unit diagonal
and dividing by the nonzero coherence. -/
def signature (μ : ℂ) (G : Matrix n n ℂ) : Matrix n n ℂ :=
  μ⁻¹ • (G - 1)

/-- The coefficient β in S² = 8I + βS. -/
def signatureCoeff (μ : ℂ) : ℂ := (1 / 4) / μ

theorem gram_eq_shifted_signature (G : Matrix n n ℂ) (μ : ℂ) (hμ : μ ≠ 0) :
    G = 1 + μ • signature μ G := by
  rw [signature, smul_smul, mul_inv_cancel₀ hμ, one_smul]
  abel

private theorem square_shift (S : Matrix n n ℂ) (μ : ℂ) :
    (1 + μ • S) * (1 + μ • S) =
      1 + (2 * μ) • S + (μ * μ) • (S * S) := by
  simp only [Matrix.add_mul, Matrix.mul_add, Matrix.one_mul, Matrix.mul_one, Matrix.smul_mul, Matrix.mul_smul,
    smul_smul, two_mul, add_smul] <;> abel

private theorem shifted_tight_residual (S : Matrix n n ℂ) (μ : ℂ)
    (hμ : μ ≠ 0) (hSq : μ * μ = (5 : ℂ) / 32) :
    (1 + μ • S) * (1 + μ • S) - (9 / 4 : ℂ) • (1 + μ • S) =
      (μ * μ) • (S * S - (8 : ℂ) • (1 : Matrix n n ℂ) - signatureCoeff μ • S) := by
  have hCoeff : μ * μ * signatureCoeff μ = (1 / 4 : ℂ) * μ := by
    dsimp only [signatureCoeff]
    field_simp [hμ] <;> ring
  rw [square_shift]
  apply Matrix.ext
  intro i j
  simp only [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  linear_combination ((8 : ℂ) * (1 : Matrix n n ℂ) i j) * hSq + S i j * hCoeff

private theorem matrix_smul_eq_zero_iff (A : Matrix n n ℂ) (c : ℂ) (hc : c ≠ 0) :
    c • A = 0 ↔ A = 0 := by
  constructor
  · intro hz
    apply Matrix.ext
    intro i j
    have hij := congrArg (fun B : Matrix n n ℂ => B i j) hz
    simp only [Matrix.smul_apply, smul_eq_mul, Matrix.zero_apply] at hij ⊢
    exact (mul_eq_zero.mp hij).resolve_left hc
  · rintro rfl
    exact smul_zero c

/-- For the ETF parameters, Gram tightness and the quadratic signature equation
are equivalent. The dimension-free statement isolates the algebraic step. -/
theorem tight_iff_signature (G : Matrix n n ℂ) (μ : ℂ)
    (hSq : μ * μ = (5 : ℂ) / 32) :
    G * G = (9 / 4 : ℂ) • G ↔
      signature μ G * signature μ G =
        (8 : ℂ) • (1 : Matrix n n ℂ) + signatureCoeff μ • signature μ G := by
  have hμ : μ ≠ 0 := by
    intro hz
    norm_num [hz] at hSq
  let S := signature μ G
  change G * G = (9 / 4 : ℂ) • G ↔
    S * S = (8 : ℂ) • (1 : Matrix n n ℂ) + signatureCoeff μ • S
  have hG : G = 1 + μ • S := gram_eq_shifted_signature G μ hμ
  have hResidual : G * G - (9 / 4 : ℂ) • G =
      (μ * μ) • (S * S - (8 : ℂ) • (1 : Matrix n n ℂ) - signatureCoeff μ • S) :=
    (congrArg (fun A : Matrix n n ℂ => A * A - (9 / 4 : ℂ) • A) hG).trans
      (shifted_tight_residual S μ hμ hSq)
  calc
    G * G = (9 / 4 : ℂ) • G ↔ G * G - (9 / 4 : ℂ) • G = 0 := sub_eq_zero.symm
    _ ↔ (μ * μ) • (S * S - (8 : ℂ) • (1 : Matrix n n ℂ) - signatureCoeff μ • S) = 0 := by
      rw [hResidual]
    _ ↔ S * S - (8 : ℂ) • (1 : Matrix n n ℂ) - signatureCoeff μ • S = 0 :=
      matrix_smul_eq_zero_iff _ _ (mul_ne_zero hμ hμ)
    _ ↔ S * S = (8 : ℂ) • (1 : Matrix n n ℂ) + signatureCoeff μ • S := by
      rw [← sub_add_eq_sub_sub, sub_eq_zero]

/-- Synthesis-matrix tightness implies tightness of its Gram matrix. -/
theorem gram_tight_of_frame {m : Type*} [Fintype m] [DecidableEq m]
    (F : Matrix m n ℂ)
    (hF : F * F.conjTranspose = (9 / 4 : ℂ) • (1 : Matrix m m ℂ)) :
    (F.conjTranspose * F) * (F.conjTranspose * F) =
      (9 / 4 : ℂ) • (F.conjTranspose * F) := by
  calc
    (F.conjTranspose * F) * (F.conjTranspose * F) =
        F.conjTranspose * ((F * F.conjTranspose) * F) := by
      rw [Matrix.mul_assoc (F.conjTranspose) F (F.conjTranspose * F),
        Matrix.mul_assoc F (F.conjTranspose) F]
    _ = _ := by rw [hF, Matrix.smul_mul, Matrix.one_mul, Matrix.mul_smul]

end Signature
end ETF49

end
