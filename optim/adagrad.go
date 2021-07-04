package optim

import (
	ts "github.com/sugarme/gotch/tensor"
)

type Adagrad struct {
	*BaseOptimizer
	*AdagradConfig
}


// Length of State is 1, sum
func (a *Adagrad) Step()  {
	once.Do(func() {
		eps = ts.FloatScalar(a.Eps)
	})
	for name, t := range a.TrainedTensors {
		grad := t.Tensor.MustGrad(false)
		defer grad.MustDrop()
		clr := ts.FloatScalar(-(1 + float64(a.StepCount - 1) * a.LrDecay) / a.LR)
		sum := a.States[name][0]
		sum.MustAddcmul_(grad, grad)
		std := sum.MustSqrt(false)
		std.MustAdd1_(eps)
		std.MustMul1_(clr)
		t.Tensor.MustAddcdiv_(grad, std)
	}
}
