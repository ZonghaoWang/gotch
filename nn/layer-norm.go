package nn

// A layer-normalization layer.
import (
	ts "github.com/zonghaowang/gotch/tensor"
)

// Layer-normalization config.
type LayerNormConfig struct {
	CudnnEnable       bool
	Eps               float64
	ElementwiseAffine bool
	WsInit            Init
	BsInit            Init
}

func DefaultLayerNormConfig() *LayerNormConfig {
	return &LayerNormConfig{
		CudnnEnable:       true,
		Eps:               1e-5,
		ElementwiseAffine: true,
		WsInit:            NewConstInit(1.0),
		BsInit:            NewConstInit(0.0),
	}
}

// A layer-normalization layer.
type LayerNorm struct {
	Config          *LayerNormConfig
	Ws              *ts.Tensor // optional
	Bs              *ts.Tensor // optional
	NormalizedShape []int64
}

func NewLayerNorm(vs *Path, normalizedShape []int64, config *LayerNormConfig) *LayerNorm {

	var (
		ws *ts.Tensor
		bs *ts.Tensor
	)
	if config.ElementwiseAffine {
		ws = vs.NewVar("weight", normalizedShape, config.WsInit)
		bs = vs.NewVar("bias", normalizedShape, config.BsInit)
	}

	return &LayerNorm{config, ws, bs, normalizedShape}
}

// Implement Module interface for LayerNorm:
// =========================================

func (ln *LayerNorm) Forward(xs *ts.Tensor) (retVal *ts.Tensor) {

	return ts.MustLayerNorm(xs, ln.NormalizedShape, ln.Ws, ln.Bs, ln.Config.Eps, ln.Config.CudnnEnable)
}
