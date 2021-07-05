package optim

import (
	ts "github.com/zonghaowang/gotch/tensor"
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
	a.BaseOptimizer.Step(a.AdagradConfig)
	for name, t := range a.TrainedTensors {
		grad := t.Tensor.MustGrad(false)
		defer grad.MustDrop()
		clr := ts.FloatScalar(-(1 + float64(a.StepCount - 1) * a.LrDecay) / a.LR)
		sum := a.States[name][0]
		sum.MustAddcmul_(grad, grad)
		std := sum.MustSqrt(false).MustAdd1(eps, true).MustMul1(clr, true)
		t.Tensor.MustData(false).MustAddcdiv_(grad, std)
		std.MustDrop()
		clr.MustDrop()
	}
}
