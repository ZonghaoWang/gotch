package optim

import (
	ts "github.com/zonghaowang/gotch/tensor"
)

type SGD struct {
	*BaseOptimizer
	*SGDConfig
}

// Length of State is 1.
func (s *SGD) Step() {
	s.BaseOptimizer.Step(s.SGDConfig)
	for name, t := range s.TrainedTensors {
		grad := t.Tensor.MustGrad(false)
		defer grad.MustDrop()
		if s.Wd != 0.0 {
			// Weight Decay
			grad.MustAddWithAlpha_(t.Tensor, s.Wd)
		}
		if s.Momentum != 0 {
			mon := ts.FloatScalar(s.Momentum)
			defer mon.MustDrop()
			s.States[name][0].MustMul1_(mon)
			s.States[name][0].MustAddWithAlpha_(grad, 1 - s.Dampening)
		}
		if s.Nesterov {
			grad.MustAddWithAlpha_(s.States[name][0], s.Momentum)
		} else if s.Momentum != 0 {
			grad.Copy_(s.States[name][0])
		}
		t.Tensor.MustData(false).MustAddWithAlpha_(grad, -s.LR)
	}
}


