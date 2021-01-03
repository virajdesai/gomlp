package main

import (
	"NeuralNet/mlp"
	"NeuralNet/mnist"
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// get data
	training, test, err := mnist.Load("data")
	if err != nil {
		fmt.Println("error loading data")
	}

	// normalize image byte data
	var trainData, trainLabels,testData, testLabels []*mat.Dense

	for _, image := range training.Images {
		im := make([]float64, 28*28)
		for i, b := range image {
			im[i] = float64(b) / 255
		}
		trainData = append(trainData, mat.NewDense(28*28, 1, im))
	}

	for _, label := range training.Labels {
		l := make([]float64, 10)
		l[label] = 1
		trainLabels = append(trainLabels, mat.NewDense(10, 1, l))
	}

	for _, image := range test.Images {
		im := make([]float64, 28*28)
		for i, b := range image {
			im[i] = float64(b) / 255
		}
		testData = append(testData, mat.NewDense(28*28, 1, im))
	}

	for _, label := range test.Labels {
		l := make([]float64, 10)
		l[label] = 1
		testLabels = append(testLabels, mat.NewDense(10, 1, l))
	}

	net := mlp.New(784, 20, 20, 10)
	net.Train(
		mlp.DataSet{Data: trainData[:30000], Labels: trainLabels[:30000]},
		mlp.DataSet{Data: trainData[59000:], Labels: trainLabels[59000:]},
		10,
		10,
		3,
	)
	fmt.Printf("\nAccuracy: ")
	a := net.Evaluate(mlp.DataSet{testData, testLabels}) * 100
	fmt.Printf(" %v%%\n", a)
}

