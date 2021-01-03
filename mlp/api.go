package mlp

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

type DataSet struct {
	Data   []*mat.Dense
	Labels []*mat.Dense
}

func New(sizes ...int) *net {
	if len(sizes) < 2 {
		panic("network must have at least two layers")
	}

	var weights = make([]*mat.Dense, len(sizes) - 1)
	var biases = make([]*mat.Dense, len(sizes) - 1)

	for i := 0; i < len(sizes) - 1; i++ {
		// creates a matrix of random values for each layer such that w[j][k] is the
		// weight for the connection between the kth neuron in the previous layer and the
		// jth in the next layer
		rand.Seed(time.Now().UnixNano())
		w := make([]float64, sizes[i + 1] * sizes[i])
		for j := range w {
			w[j] = rand.NormFloat64()
		}
		weights[i] = mat.NewDense(sizes[i + 1], sizes[i], w)

		// creates a vector of random values for each layer (no biases for input layer)
		b := make([]float64, sizes[i + 1])
		for j := range b {
			b[j] = rand.NormFloat64()
		}
		biases[i] = mat.NewDense(sizes[i + 1], 1, b)
	}

	return &net{sizes, biases, weights}
}

func (n *net) Train(data, validation DataSet, batchSize, epochs int, learningRate float64) {
	for i := 0; i < epochs; i++ {
		fmt.Printf("Epoch %v", i + 1)
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(
			len(data.Data),
			func(i, j int) {
				data.Data[i], data.Data[j] = data.Data[j], data.Data[i]
				data.Labels[i], data.Labels[j] = data.Labels[j], data.Labels[i]
			},
		)

		for j := 0; j < len(data.Data); j += batchSize {
			n.gradientDescent(
				data.Data[j : j + batchSize],
				data.Labels[j : j + batchSize],
				learningRate,
			)
		}

		if len(validation.Data) > 0 {
			fmt.Print(": ")
			n.Evaluate(validation)
		}
		fmt.Println()
	}
}

func (n *net) Evaluate(data DataSet) float64 {
	correct := 0
	for i := 0; i < len(data.Data); i++ {
		output := n.Predict(data.Data[i])
		r, _ := output.Dims()

		maxP := output.At(0, 0)
		maxPInd := 0
		maxE := data.Labels[i].At(0, 0)
		maxEInd := 0

		for j := 0; j < r; j++ {
			p := mat.Row(nil, j, output)
			e := mat.Row(nil, j, data.Labels[i])
			if p[0] > maxP {
				maxP = p[0]
				maxPInd = j
			}
			if e[0] > maxE {
				maxE = e[0]
				maxEInd = j
			}
		}
		if maxEInd == maxPInd {
			correct++
		}
	}

	fmt.Printf("%v / %v", correct, len(data.Data))
	return float64(correct) / float64(len(data.Data))
}

func (n *net) Predict(input *mat.Dense) mat.Matrix {
	a, _ := n.forward(input)
	return a[len(a) - 1]
}

