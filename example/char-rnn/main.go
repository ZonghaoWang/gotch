package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

const (
	LearningRate float64 = 0.01
	HiddenSize   int64   = 256
	SeqLen       int64   = 180
	BatchSize    int64   = 256
	Epochs       int     = 100
	SamplingLen  int64   = 1024
)

func sample(data ts.TextData, lstm nn.LSTM, linear nn.Linear, device gotch.Device) (retVal string) {

	labels := data.Labels()
	state := lstm.ZeroState(1)
	lastLabel := int64(0)
	var result string

	for i := 0; i < int(SamplingLen); i++ {

		input := ts.MustZeros([]int64{1, labels}, gotch.Float, device)
		input.MustNarrow(1, lastLabel, 1, false).MustFill_(ts.FloatScalar(1.0))
		state = lstm.Step(input, state)

		forwardTs := linear.Forward(state.(nn.LSTMState).H())
		squeeze1Ts := forwardTs.MustSqueeze1(0, false)
		softmaxTs := squeeze1Ts.MustSoftmax(-1, gotch.Float, false)
		sampledY := softmaxTs.MustMultinomial(1, false, false)

		lastLabel = sampledY.Int64Values()[0]

		result += fmt.Sprintf("%v", lastLabel)
	}

	return result
}

func main() {
	cuda := gotch.NewCuda()
	device := cuda.CudaIfAvailable()

	vs := nn.NewVarStore(device)
	data, err := ts.NewTextData("../../data/char-rnn/input.txt")
	if err != nil {
		log.Fatal(err)
	}

	labels := data.Labels()
	fmt.Printf("Dataset loaded, %v labels\n", labels)

	lstm := nn.NewLSTM(vs.Root(), labels, HiddenSize, nn.DefaultRNNConfig())
	linear := nn.NewLinear(vs.Root(), HiddenSize, labels, nn.DefaultLinearConfig())
	optConfig := nn.DefaultAdamConfig()
	opt, err := optConfig.Build(vs, LearningRate)
	if err != nil {
		log.Fatal(err)
	}

	for epoch := 1; epoch <= Epochs; epoch++ {
		sumLoss := 0.0
		cntLoss := 0.0

		dataIter := data.IterShuffle(SeqLen+1, BatchSize)

		for {
			batchTs, ok := dataIter.Next()
			if !ok {
				break
			}

			testTs := ts.MustOfSlice([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8})
			testVal := testTs.Onehot(14).Float64Values()
			fmt.Printf("testVal: %v\n", testVal)

			fmt.Printf("batchTs shape: %v\n", batchTs.MustSize())

			batchNarrow := batchTs.MustNarrow(1, 0, SeqLen, false)
			fmt.Printf("batchNarrow shape: %v\n", batchNarrow.MustSize())
			xsOnehotTmp := batchNarrow.Onehot(labels)
			xsOnehot := xsOnehotTmp.MustTo(device, true) // shape: [256, 180, 65]
			ys := batchTs.MustNarrow(1, 1, SeqLen, false).MustTotype(gotch.Int64, false)

			// NOTE. WARNING occurred here...
			// Warning: RNN module weights are not part of single contiguous chunk of memory.
			// This means they need to be compacted at every call, possibly greatly increasing memory usage.
			// To compact weights again call flatten_parameters().
			// See: https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/21
			lstmOut, _ := lstm.Seq(xsOnehot)

			panic("reached")

			logits := linear.Forward(lstmOut)

			lossView := logits.MustView([]int64{BatchSize * SeqLen, labels}, false)

			loss := lossView.CrossEntropyForLogits(ys.MustTo(device, false)).MustView([]int64{BatchSize * SeqLen}, false)

			opt.BackwardStepClip(loss, 0.5)
			sumLoss += loss.Float64Values()[0]
			cntLoss += 1.0
		}

		fmt.Printf("Epoch %v - Loss: %v", epoch, sumLoss/cntLoss)
		fmt.Printf("Sample: %v", sample(data, lstm, linear, device))

	}

}
