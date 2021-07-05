package tensor

func (t *Tensor)MustAddWithAlpha(other *Tensor, scala float64, del bool) *Tensor {
	if scala == 1.0 {
		return t.MustAdd(other, del)
	}
	s := FloatScalar(scala)
	defer s.MustDrop()
	otherAlpha := other.MustMul1(s, false)
	defer otherAlpha.MustDrop()
	return t.MustAdd(otherAlpha, del)
}

func (t *Tensor)MustAddWithAlpha_(other *Tensor, scala float64) {
	if scala == 1.0 {
		t.MustData(false).MustAdd_(other)
		return
	}
	s := FloatScalar(scala)
	defer s.MustDrop()
	otherAlpha := other.MustMul1(s, false)
	defer otherAlpha.MustDrop()
	t.MustAdd_(otherAlpha)
}