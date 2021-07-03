package main

import (
	"fmt"
	// "log"

	"github.com/zonghaowang/gotch/tensor"
)

func main() {
	x := tensor.TensorFrom([]float64{2.0})
	x = x.MustSetRequiresGrad(true, false)
	x.ZeroGrad()

	xy := tensor.TensorFrom([]float64{2.0})
	xz := tensor.TensorFrom([]float64{3.0})

	y := x.MustMul(xy, false)
	z := x.MustMul(xz, false)

	y.Backward()
	xgrad := x.MustGrad(false)
	xgrad.Print() // [2.0]
	z.Backward()
	xgrad = x.MustGrad(false)
	xgrad.Print() // [5.0] due to accumulated 2.0 + 3.0

	isGradEnabled := tensor.MustGradSetEnabled(false)
	fmt.Printf("Previous GradMode enabled state: %v\n", isGradEnabled)
	isGradEnabled = tensor.MustGradSetEnabled(true)
	fmt.Printf("Previous GradMode enabled state: %v\n", isGradEnabled)

}
