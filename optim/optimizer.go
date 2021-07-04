package optim

// Optimizers from pytorch

import (
	"github.com/sugarme/gotch/nn"
	"log"
	"sync"

	ts "github.com/sugarme/gotch/tensor"
)

var (
	once sync.Once
	eps, rho,rhoMB1 *ts.Scalar
)

// Optimizer is a struct object to run gradient descent.
type Optimizer interface {
	StateDict() map[string][]*ts.Tensor
	Step()
	ZeroGrad()
}

type BaseOptimizer struct {
	VS *nn.VarStore
	States map[string][]*ts.Tensor
	TrainedTensors map[string]nn.Var
	LR float64
	StepCount int64
}

func (b *BaseOptimizer) StateDict() map[string][]*ts.Tensor {
	return b.States
}

func ParseNamedTrainedTensors(vs *nn.VarStore) map[string]nn.Var {
	rst := make(map[string]nn.Var, len(vs.Vars.TrainableVariables))
	for _, trainedVar := range vs.Vars.TrainableVariables {
		dataPtr, err := trainedVar.Tensor.DataPtr()
		if err != nil {
			panic(err)
		}
		for name, namedTensor := range vs.Vars.NamedVariables {
			if dp, err := namedTensor.DataPtr(); err != nil {
				panic(err)
			} else if dp == dataPtr {
				rst[name] = trainedVar
			}
		}
	}
	return rst
}

// OptimizerConfig defines Optimizer configurations. These configs can be used to build optimizer.
type OptimizerConfig interface {
	Build(vs *nn.VarStore, stateDict map[string][]*ts.Tensor, lr float64) Optimizer
	GetAssTensorNum() uint
}

func (c *SGDConfig) Build(vs *nn.VarStore, stateDict map[string][]*ts.Tensor, lr float64) Optimizer {
	return &SGD{BaseOptimizer: &BaseOptimizer{VS: vs, LR: lr, States: stateDict, TrainedTensors: ParseNamedTrainedTensors(vs)}, SGDConfig: c}
}

func (c *SGDConfig) GetAssTensorNum() uint {
	return 0
}

func (c *AdamConfig) Build(vs *nn.VarStore, stateDict map[string][]*ts.Tensor,  lr float64) Optimizer {
	return &Adam{BaseOptimizer: &BaseOptimizer{VS: vs, LR: lr, States: stateDict, TrainedTensors: ParseNamedTrainedTensors(vs)}, AdamConfig: c}
}

func (c *AdamConfig) GetAssTensorNum() uint {
	return 2
}

func (c *AdamWConfig) Build(vs *nn.VarStore, stateDict map[string][]*ts.Tensor,  lr float64) Optimizer {
	return &AdamW{BaseOptimizer: &BaseOptimizer{VS: vs, LR: lr, States: stateDict, TrainedTensors: ParseNamedTrainedTensors(vs)}, AdamWConfig: c}
}

func (c *AdamWConfig) GetAssTensorNum() uint {
	return 2
}

func (c *RMSPropConfig) Build(vs *nn.VarStore, stateDict map[string][]*ts.Tensor,  lr float64) Optimizer {
	return &RMSProp{BaseOptimizer: &BaseOptimizer{VS: vs, LR: lr, States: stateDict, TrainedTensors: ParseNamedTrainedTensors(vs)}, RMSPropConfig: c}
}

func (c *RMSPropConfig) GetAssTensorNum() uint {
	return 2
}

// SGDConfig holds parameters for building the SGD (Stochastic Gradient Descent) optimizer.
type SGDConfig struct {
	Momentum  float64
	Dampening float64
	Wd        float64
	Nesterov  bool
}

// DefaultSGDConfig creates SGDConfig with default values.
func DefaultSGDConfig() *SGDConfig {
	return &SGDConfig{
		Momentum:  0.0,
		Dampening: 0.0,
		Wd:        0.0,
		Nesterov:  false,
	}
}

// NewSGD creates the configuration for a SGD optimizer with specified values
func NewSGDConfig(momentum, dampening, wd float64, nesterov bool) *SGDConfig {
	return &SGDConfig{
		Momentum:  momentum,
		Dampening: dampening,
		Wd:        wd,
		Nesterov:  nesterov,
	}
}
// Adam optimizer:
// ===============

type AdamConfig struct {
	Beta1 float64
	Beta2 float64
	Wd    float64
	Eps   float64
	Amsgrad bool
}

// DefaultAdamConfig creates AdamConfig with default values
func DefaultAdamConfig() *AdamConfig {
	return &AdamConfig{
		Beta1: 0.9,
		Beta2: 0.999,
		Wd:    0.0,
		Eps:   1e-7,
		Amsgrad: false,
	}
}

// NewAdamConfig creates AdamConfig with specified values
func NewAdamConfig(beta1, beta2, wd float64, eps float64, amsgrad bool) *AdamConfig {
	return &AdamConfig{
		Beta1: beta1,
		Beta2: beta2,
		Wd:    wd,
		Eps:   eps,
		Amsgrad: amsgrad,
	}
}

// AdamW optimizer:
// ===============

type AdamWConfig struct {
	Beta1 float64
	Beta2 float64
	Wd    float64
}

// DefaultAdamConfig creates AdamConfig with default values
func DefaultAdamWConfig() *AdamConfig {
	return &AdamConfig{
		Beta1: 0.9,
		Beta2: 0.999,
		Wd:    0.0,
	}
}

// NewAdamConfig creates AdamConfig with specified values
func NewAdamWConfig(beta1, beta2, wd float64) *AdamWConfig {
	return &AdamWConfig{
		Beta1: beta1,
		Beta2: beta2,
		Wd:    wd,
	}
}

// RMSProp optimizer:
// ===============

type RMSPropConfig struct {
	Alpha    float64
	Eps      float64
	Wd       float64
	Momentum float64
	Centered bool
}

// DefaultAdamConfig creates AdamConfig with default values
func DefaultRMSPropConfig() *RMSPropConfig {
	return &RMSPropConfig{
		Alpha:    0.99,
		Eps:      1e-8,
		Wd:       0.0,
		Momentum: 0.0,
		Centered: false,
	}
}

// NewRMSPropConfig creates RMSPropConfig with specified values
func NewRMSPropConfig(alpha, eps, wd, momentum float64, centered bool) *RMSPropConfig {
	return &RMSPropConfig{
		Alpha:    alpha,
		Eps:      eps,
		Wd:       wd,
		Momentum: momentum,
		Centered: centered,
	}
}

type AdagradConfig struct {
	LrDecay    float64
	Eps      float64
	Wd       float64
}

// AdagradConfig creates AdagradConfig with specified values
func NewAdagradConfig(alpha, eps, wd float64) *AdagradConfig {
	return &AdagradConfig{
		LrDecay:    alpha,
		Eps:      eps,
		Wd:       wd,
	}
}

func DefaultAdagradConfig() *AdagradConfig {
	return &AdagradConfig{
		LrDecay:    0.99,
		Eps:      1e-8,
		Wd:       0.0,
	}
}



type AdadeltaConfig struct {
	Rho      float64
	Eps      float64
	Wd       float64
}

// AdagradConfig creates AdagradConfig with specified values
func NewDefaultAdadeltaConfig(rho, eps, wd float64) *AdadeltaConfig {
	return &AdadeltaConfig{
		Rho:      rho,
		Eps:      eps,
		Wd:       wd,
	}
}

func DefaultAdadeltaConfig() *AdadeltaConfig {
	return &AdadeltaConfig{
		Rho:      0.9,
		Eps:      1e-7,
		Wd:       0.0,
	}
}



// ZeroGrad zeroes the gradient for the tensors tracked by this optimizer.
func (opt *BaseOptimizer) ZeroGrad() {
	for _, t := range opt.VS.Vars.TrainableVariables {
		t.Tensor.ZeroGrad()
	}
}

// Step performs an optimization step, updating the tracked tensors based on their gradients.
func (opt *BaseOptimizer) Step() {
	opt.StepCount++
}

// ResetStepCount set step count to zero.
func (opt *BaseOptimizer) ResetStepCount() {
	opt.StepCount = 0
}

/// TODO. Clips gradient L2 norm over all trainable parameters.
//
// The norm is computed over all gradients together, as if they were
// concatenated into a single vector.
func (opt BaseOptimizer) ClipGradNorm(max float64) {
	// TODO.
	log.Fatalf("Not implemented yet!")
}