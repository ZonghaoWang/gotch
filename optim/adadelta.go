package optim

import ts "github.com/zonghaowang/gotch/tensor"

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
	a.BaseOptimizer.Step(a.AdadeltaConfig)
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
		squareAvg.MustAddcmul_(grad, gradScope)
		std := squareAvg.MustAdd1(eps, false).MustSqrt(true)
		delta := accDelta.MustAdd1(eps, false).MustSqrt(true).MustDiv(std, true).MustMul(grad, true)
		t.Tensor.MustData(false).MustAddWithAlpha_(delta, -a.LR)
		accDelta.MustMul1_(rho)
		deltaScope := delta.MustMul1(rhoMB1, false)
		accDelta.MustAddcmul_(delta, deltaScope)
		deltaScope.MustDrop()
		gradScope.MustDrop()
		delta.MustDrop()
		std.MustDrop()
	}
}

