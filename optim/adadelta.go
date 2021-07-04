package optim

import ts "github.com/sugarme/gotch/tensor"

// Length of State is 2, Index0 is square_avg, Index1 is acc_delta
type Adadelta struct {
	*BaseOptimizer
	*AdadeltaConfig
}


func (a *Adadelta) Step() {
	once.Do(func() {
		eps = ts.FloatScalar(a.Eps)
		rho = ts.FloatScalar(a.Rho)
		rhoMB1 = ts.FloatScalar(1 - a.Rho)

	})
	for name, t := range a.TrainedTensors {
		grad := t.Tensor.MustGrad(false)
		defer grad.MustDrop()
		squareAvg:= a.States[name][0]
		accDelta := a.States[name][1]
		if a.Wd != 0.0 {
			grad.MustAddWithAlpha_(t.Tensor, a.Wd)
		}
		squareAvg.MustMul1_(rho)
		gradScope := grad.MustMul1(rhoMB1, false)
		defer gradScope.MustDrop()
		squareAvg.MustAddcmul_(grad, gradScope)
		std := squareAvg.MustAdd1(eps, false)
		std.MustSqrt_()
		delta := accDelta.MustAdd1(eps, false)
		delta.MustSqrt_()
		delta.MustDiv_(std)
		delta.MustMul_(grad)
		t.Tensor.MustAddWithAlpha_(delta, -a.LR)
		accDelta.MustMul1_(rho)
		deltaScope := delta.MustMul1(rhoMB1, false)
		defer deltaScope.MustDrop()
		accDelta.MustAddcmul_(delta, deltaScope)
	}
}

