package mlp

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestNetwork_Forward11(t *testing.T) {
	// initialize two neuron network
	var w, b []*mat.Dense
	var input, w0, b0 float64 = 7, 4, -25
	b = append(b, mat.NewDense(1, 1, []float64{b0}))
	w = append(w, mat.NewDense(1, 1, []float64{w0}))
	net := net{[]int{1, 1},  b,  w}

	// feed forward through network
	activations, _ := net.forward(mat.NewDense(1, 1, []float64{input}))
	result := activations[len(activations) - 1].At(0, 0)

	expected := sigmoid(input * w0 + b0)
	if result != expected {
		t.Errorf("Two node network failed, expected %v, got %v", expected, result)
	}
}

func TestNetwork_Forward32(t *testing.T) {
	// initialize two layer network with layers 3 and 2
	var w, b []*mat.Dense

	var w00, w10, w01, w11, w02, w12 float64 = 1, 2, 3, 4, 5, 6
	wL0 := []float64{w00, w01, w02,
				     w10, w11, w12}

	var b0, b1 float64 = -5, -3
	bL0 := []float64{b0, b1}

	b = append(b, mat.NewDense(2, 1, bL0))
	w = append(w, mat.NewDense(2, 3, wL0))

	net := net{[]int{3, 2},  b,  w}

	// feed forward inputs x[i]
	var x0, x1, x2 float64 = 2, 3, 4
	activations, _ := net.forward(mat.NewDense(3, 1, []float64{x0, x1, x2}))

	var result = activations[len(activations)-1]

	a := []float64{sigmoid(w00*x0 + w01*x1 + w02*x2 + b0), sigmoid(w10*x0 + w11*x1 + w12*x2 + b1)}
	if result.At(0, 0) != a[0] && result.At(1,0) != a[1] {
		t.Errorf("Two layer network failed")
	}
}

func TestNetwork_Cost(t *testing.T) {
	// initialize two neuron network
	net := New(5, 10,5, 2)
	x := mat.NewDense(5, 1, []float64{0.9,0.3,0.56,0.93,0.23})
	y := mat.NewDense(2, 1, []float64{1, 0})

	net.backward(x, y)
}