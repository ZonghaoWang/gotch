package optim

import ts "github.com/zonghaowang/gotch/tensor"

type RMSProp struct {
	*BaseOptimizer
	*RMSPropConfig
}

// Length of State is 1, square_avg
// If mon > 0 then +1, momentum_buffer
// if centered then +1, grad_avg
func (r *RMSProp) Step() {
	once.Do(func() {
		eps = ts.FloatScalar(r.Eps)
		alpha = ts.FloatScalar(r.Alpha)
		alphaBM1 = ts.FloatScalar(1 - r.Alpha)
		momentum = ts.FloatScalar(r.Momentum)
		lr = ts.FloatScalar(r.LR)
		lrNeg = ts.FloatScalar(-r.LR)
	})
	r.BaseOptimizer.Step()
	for name, t := range r.TrainedTensors {
		grad := t.Tensor.MustGrad(false)
		defer grad.MustDrop()
		squareAvg := r.States[name][0]
		var momentumBuffer, gradAvg *ts.Tensor
		if r.Momentum > 0 {
			momentumBuffer = r.States[name][1]
		}
		if r.Centered {
			gradAvg = r.States[name][len(r.States[name])-1]
		}
		if r.Wd != 0 {
			grad.MustAddWithAlpha_(t.Tensor, r.Wd)
		}
		squareAvg.MustMul1_(alpha)
		gradScope := grad.MustMul1(alphaBM1, false)
		squareAvg.MustAddcmul_(grad, gradScope)
		var avg *ts.Tensor
		if r.Centered {
			gradAvg.MustMul1_(alpha)
			gradAvg.MustAddWithAlpha_(grad, 1 - r.Alpha)
			gradAvgPow := gradAvg.MustMul(gradAvg, false)
			avg = squareAvg.MustSub(gradAvgPow, false)
		} else {
			avg = squareAvg.MustSqrt(false)
			avg.MustAdd1_(eps)
		}
		if r.Momentum > 0 {
			momentumBuffer.MustMul1_(momentum)
			momentumBuffer.MustAddcdiv_(grad, avg)
			t.Tensor.MustAddWithAlpha_(momentumBuffer, -r.LR)
		} else {
			gradStep := grad.MustMul1(lrNeg, false)
			t.Tensor.MustAddcdiv_(gradStep, avg)
			gradStep.MustDrop()
		}
	}
}


