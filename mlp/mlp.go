package mlp

import (
	"gonum.org/v1/gonum/mat"
)

type net struct {
	layers  []int
	biases  []*mat.Dense
	weights []*mat.Dense
}

// Runs input through network and returns slice of activations and weighted sums
func (n *net) forward(x *mat.Dense) (a, z []*mat.Dense) {
	a = make([]*mat.Dense, len(n.layers))
	z = make([]*mat.Dense, len(n.layers)-1)
	a[0] = x

	for i := 0; i < len(n.layers)-1; i++ {
		// aL and zL are activations and weighted sums for a particular layer respectively
		var zL, aL = new(mat.Dense), new(mat.Dense)

		// z[l] = w[l] * a[l-1] + b[l]
		zL.Mul(n.weights[i], x)
		zL.Apply(
			func(_, c int, v float64) float64 {
				return v + n.biases[i].At(c, 0)
			},
			zL,
		)

		// a[l] = sigmoid(z[l])
		aL.Apply(
			func(_, _ int, v float64) float64 {
				return sigmoid(v)
			},
			zL,
		)

		a[i+1] = aL
		z[i] = zL
		x = aL
	}

	return
}

// Calculates gradients to minimize cost through the backpropagation technique
func (n *net) backward(x, y *mat.Dense) (dW, dB []*mat.Dense) {
	a, z := n.forward(x)

	// To calculate gradient, find dC/dw and dC/db for all weights and biases
	// Cost: C = (1/2)(a[l] - y)^2
	// Activation: a[l] = sigmoid(z[l])
	// z[l] = w[l] * a[l-1] + b[l]

	// dC/dz[l] = dC/da[l] * da[l]/dz[l]
	// dC/dw[l] = dC/dz[l] * dz[l]/dw[l]
	// dC/db[l] = dC/dz[l] * dz[l]/db[l]

	// dC/da[l] = a[l] - y
	dC := new(mat.Dense)
	dC.Sub(a[len(a)-1], y)

	// sp = da[l]/dz[l] = sigmoidPrime(z[l])
	sp := new(mat.Dense)
	sp.Apply(
		func(_, _ int, v float64) float64 {
			return sigmoidPrime(v)
		},
		z[len(z)-1],
	)

	// delta = dC/dz[l] = dC/da[l] * da/dz[l]
	delta := new(mat.Dense)
	delta.MulElem(dC, sp)

	dB = make([]*mat.Dense, len(n.biases))
	dW = make([]*mat.Dense, len(n.weights))

	// dz[l]/db[l] = 1, so dC/db[l] = delta[l] * 1
	dB[len(dB)-1] = delta

	// dz[l]/dw[l] = a[l-1], so dC/dw[l] = delta[l] * a[l-1]
	dW[len(dW)-1] = new(mat.Dense)
	dW[len(dW)-1].Mul(delta, a[len(a)-2].T())

	// calculate partial derivatives of a previous layer from partials of next
	for i := len(n.layers) - 3; i >= 0; i-- {
		// dC/dz[l] can be written in terns of dC/dz[l+1]
		// dC/dz[l] = dC/dz[l+1] * dz[l+1]/da[l] * da[l]/dz[l]

		// da[l]/dz[l] = sigmoidPrime(z[l])
		sp = new(mat.Dense)
		sp.Apply(
			func(_, _ int, v float64) float64 {
				return sigmoidPrime(v)
			},
			z[i],
		)

		// delta[l] = delta[l+1] * w[l+1] * sp
		newDelta := new(mat.Dense)
		newDelta.Mul(n.weights[i+1].T(), delta)
		newDelta.MulElem(newDelta, sp)
		delta = newDelta

		// dz[l]/db[l] = 1, so dC/db[l] = delta[l] * 1
		dB[i] = delta

		// dz[l]/dw[l] = a[l-1], so dC/dw[l] = delta[l] * a[l-1]
		// a includes input layer so a[l-1] corresponds with a[i]
		dW[i] = new(mat.Dense)
		dW[i].Mul(delta, a[i].T())
	}

	return
}

func (n *net) gradientDescent(x, y []*mat.Dense, alpha float64) {
	// sum dC/dw and dC/db for all input and expected
	dW, dB := n.backward(x[0], y[0])
	for i := 1; i < len(x); i++ {
		nW, nB := n.backward(x[i], y[i])
		for j, w := range dW {
			w.Add(w, nW[j])
		}
		for j, b := range dB {
			b.Add(b, nB[j])
		}
	}

	// average and apply learning rate to dC/dw and dC/db
	for _, w := range dW {
		w.Apply(
			func(_, _ int, v float64) float64 {
				return v / float64(len(x)) * alpha
			},
			w,
		)
	}
	for _, b := range dB {
		b.Apply(
			func(_, _ int, v float64) float64 {
				return v / float64(len(x)) * alpha
			},
			b,
		)
	}

	// Adjust weights and biases accordingly
	for i := 0; i < len(n.weights); i++ {
		n.weights[i].Sub(n.weights[i], dW[i])
	}
	for i := 0; i < len(n.biases); i++ {
		n.biases[i].Sub(n.biases[i], dB[i])
	}
}
