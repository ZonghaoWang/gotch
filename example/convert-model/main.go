package main

import (
	"fmt"
	"log"

	ts "github.com/zonghaowang/gotch/tensor"
)

func main() {
	// NOTE. Python script to save model to .npz can be found at https://github.com/zonghaowang/pytorch-pretrained/bert/bert-base-uncased-to-npz.py
	filepath := "../../data/convert-model/bert/model.npz"

	namedTensors, err := ts.ReadNpz(filepath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Num of named tensor: %v\n", len(namedTensors))
	outputFile := "/home/zonghaowang/projects/transformer/data/bert/model.gt"
	err = ts.SaveMultiNew(namedTensors, outputFile)
	if err != nil {
		log.Fatal(err)
	}
}
