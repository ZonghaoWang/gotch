package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/gotch/vision"
)

const (
	ImageDim int64  = 784
	Label    int64  = 10
	MnistDir string = "../../data/mnist"

	epochs    = 100
	batchSize = 256
)

func runLinear() {
	var ds vision.Dataset
	ds = vision.LoadMNISTDir(MnistDir)

	// fmt.Printf("Train image size: %v\n", ds.TrainImages.MustSize())
	// fmt.Printf("Train label size: %v\n", ds.TrainLabels.MustSize())
	// fmt.Printf("Test image size: %v\n", ds.TestImages.MustSize())
	// fmt.Printf("Test label size: %v\n", ds.TestLabels.MustSize())

	device := (gotch.CPU).CInt()
	dtype := (gotch.Float).CInt()

	ws := ts.MustZeros([]int64{ImageDim, Label}, dtype, device).MustSetRequiresGrad(true)
	bs := ts.MustZeros([]int64{Label}, dtype, device).MustSetRequiresGrad(true)

	for epoch := 0; epoch < epochs; epoch++ {

		var loss ts.Tensor
		trainIter := ds.TrainIter(batchSize)
		trainIter.Shuffle().ToDevice(gotch.CPU)
		// item a pair of images and labels as 2 tensors
		for {
			batch, ok := trainIter.Next()
			if !ok {
				break
			}

			logits := batch.Images.MustMm(ws).MustAdd(bs)
			loss = logits.MustLogSoftmax(-1, dtype).MustNllLoss(batch.Labels)
			ws.ZeroGrad()
			bs.ZeroGrad()
			loss.Backward()

			ts.NoGrad(func() {
				ws.MustAdd_(ws.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
				bs.MustAdd_(bs.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
			})
		}

		/*
		 *       logits := ds.TrainImages.MustMm(ws).MustAdd(bs)
		 *       loss := logits.MustLogSoftmax(-1, dtype).MustNllLoss(ds.TrainLabels)
		 *
		 *       ws.ZeroGrad()
		 *       bs.ZeroGrad()
		 *       loss.Backward()
		 *
		 *       ts.NoGrad(func() {
		 *         ws.MustAdd_(ws.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
		 *         bs.MustAdd_(bs.MustGrad().MustMul1(ts.FloatScalar(-1.0)))
		 *       })
		 *       loss.Print()
		 *  */

		// bs.MustGrad().Print()

		testLogits := ds.TestImages.MustMm(ws).MustAdd(bs)
		testAccuracy := testLogits.MustArgmax(-1, false).MustEq1(ds.TestLabels).MustTotype(gotch.Float).MustMean(gotch.Float.CInt()).MustView([]int64{-1}).MustFloat64Value([]int64{0})

		fmt.Printf("Epoch: %v - Test accuracy: %v\n", epoch, testAccuracy*100)

		// fmt.Printf("Epoch: %v - Train loss: %v - Test accuracy: %v\n", epoch, loss.MustView([]int64{-1}).MustFloat64Value([]int64{0}), testAccuracy*100)
	}
}
