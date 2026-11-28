import ETF49

open ETF49

example (X Y W : FixedOne.Mat34 ℚ) (h : FixedOne.Vec4 ℚ) :
    (-1 : ℚ) = FixedOne.certificateRhs X Y W h :=
  FixedOne.certificate_identity X Y W h

#print axioms ETF49.tight_iff_signature
#print axioms ETF49.gram_tight_of_frame
#print axioms ETF49.FixedOne.certificate_identity
#print axioms ETF49.FixedOne.no_real_solution
