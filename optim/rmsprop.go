package optim

import (
	ts "github.com/sugarme/gotch/tensor"
)

type RMSProp struct {
	*BaseOptimizer
	*RMSPropConfig
}

func (R RMSProp) StateDict() map[string][]*ts.Tensor {
	panic("implement me")
}

func (R RMSProp) Step() {
	panic("implement me")
}

func (R RMSProp) ZeroGrad() {
	panic("implement me")
}

