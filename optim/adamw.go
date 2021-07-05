package optim

import (
	ts "github.com/zonghaowang/gotch/tensor"
	"math"
)

type AdamW struct {
	*BaseOptimizer
	*AdamWConfig
}

var paramDecay *ts.Scalar

func (a *AdamW) Step() {
	once.Do(func() {
		beta1 = ts.FloatScalar(a.Beta1)
		beta2 = ts.FloatScalar(a.Beta2)
		beta2MB1 = ts.FloatScalar(1 - a.Beta2)
		eps = ts.FloatScalar(a.Eps)
		paramDecay = ts.FloatScalar(1 - a.LR * a.Wd)
	})
	a.BaseOptimizer.Step()
	for name, t := range a.TrainedTensors {
		grad := t.Tensor.MustGrad(false)
		defer grad.MustDrop()
		t.Tensor.MustMul1_(paramDecay)

		expAvg := a.States[name][0]
		expAvgSq := a.States[name][1]
		if a.Wd != 0.0 {
			// Weight Decay
			grad.MustAddWithAlpha_(t.Tensor, a.Wd)
		}
		stepSize := ts.FloatScalar(-a.LR / (1 - math.Pow(a.Beta1, float64(a.StepCount))))
		biasCorrection2 := ts.FloatScalar(math.Sqrt(1 - math.Pow(a.Beta2, float64(a.StepCount))))
		defer stepSize.MustDrop()
		defer biasCorrection2.MustDrop()

		expAvg.MustMul1_(beta1)
		expAvg.MustAddWithAlpha_(grad, 1 - a.Beta1)
		gradScope := grad.MustMul1(beta2MB1, false)
		defer gradScope.MustDrop()
		expAvgSq.MustMul1_(beta2)
		expAvgSq.MustAddcmul_(grad, gradScope)
		var denom *ts.Tensor
		if a.Amsgrad {
			maxExpAvgSq := a.States[name][2]
			maxExpAvgSq.MustMaximumOut(maxExpAvgSq, expAvgSq, false)
			denom = maxExpAvgSq.MustSqrt(false).MustDiv1(biasCorrection2, true)
			denom.MustAdd1_(eps)
		} else {
			denom = expAvgSq.MustSqrt(false).MustDiv1(biasCorrection2, true)
			denom.MustAdd1_(eps)
		}
		expAvgScope := expAvg.MustMul1(stepSize, false)
		t.Tensor.MustAddcdiv_(expAvgScope, denom)
		expAvgScope.MustDrop()
		denom.MustDrop()
	}
}

