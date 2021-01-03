package mlp

import "math"

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func relu(x float64) float64 {
	return math.Max(0, x)
}

func reluPrime(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func tanh(x float64) float64 {
	return 2.0 / (1.0 + math.Exp(-2.0 * x)) - 1
}

func tanhPrime(x float64) float64 {
	return 1 - math.Pow(tanh(x), 2)
}