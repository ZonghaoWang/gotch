package optim

import (
	ts "github.com/sugarme/gotch/tensor"
)

type AdamW struct {
	*BaseOptimizer
	*AdamWConfig
}

func (a AdamW) StateDict() map[string][]*ts.Tensor {
	panic("implement me")
}

func (a AdamW) Step() {
	panic("implement me")
}

func (a AdamW) ZeroGrad() {
	panic("implement me")
}

