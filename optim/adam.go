package optim

import (
	ts "github.com/sugarme/gotch/tensor"
	"math"
)

type Adam struct {
	*BaseOptimizer
	*AdamConfig
}

// Length of State is 2, Index0 is exp_avg, Index1 is max_exp_avgsq
// If Amsgrad set, then 3
func (a *Adam) Step() {
	for name, t := range a.TrainedTensors {
		grad := t.Tensor.MustGrad(false)
		defer grad.MustDrop()
		expAvg := a.States[name][0]
		expAvgSq := a.States[name][1]
		if a.Wd != 0.0 {
			// Weight Decay
			grad.MustAddWithAlpha_(t.Tensor, a.Wd)
		}
		stepSize := ts.FloatScalar(-a.LR / (1 - math.Pow(a.Beta1, float64(a.StepCount))))
		biasCorrection2 := ts.FloatScalar(math.Sqrt(1 - math.Pow(a.Beta2, float64(a.StepCount))))
		beta1 := ts.FloatScalar(a.Beta1)
		beta2 := ts.FloatScalar(a.Beta2)
		beta2MB1 := ts.FloatScalar(1 - a.Beta2)
		eps := ts.FloatScalar(a.Eps)

		defer stepSize.MustDrop()
		defer biasCorrection2.MustDrop()
		defer beta1.MustDrop()
		defer beta2.MustDrop()
		defer beta2MB1.MustDrop()
		defer eps.MustDrop()

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
		defer expAvgScope.MustDrop()
		t.Tensor.MustAddcdiv_(expAvgScope, denom)
	}
}

