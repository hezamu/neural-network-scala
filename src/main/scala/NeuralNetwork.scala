import MatrixOps._

import scala.util.Random

case class NeuralNetwork(weights: Seq[Seq[Seq[Double]]], biases: Seq[Seq[Double]]) {
  lazy val layers: Seq[(Seq[Seq[Double]], Seq[Double])] = weights zip biases
}

object NeuralNetwork {
  // Geometry is an array of neurons in each layer.
  def initNetwork(geometry: Seq[Int]): NeuralNetwork = {
    // Weights initialize to (-0.1,0.1)
    val weights = (1 until geometry.length) map { layer =>
      (0 until geometry(layer)) map { _ =>
        (0 until geometry(layer - 1)) map { _ => Random.nextDouble / 5 - 0.1 }
      }
    }

    // Biases initialize to (-3.0,3.0)
    val biases = geometry.tail map { neurons =>
      (0 until neurons) map { _ => Random.nextDouble() * 8 - 4 }
    }

    NeuralNetwork(weights, biases)
  }

  // Stochastic gradient descent over a training set of inputs and expected outputs
  def train(network: NeuralNetwork, data: Seq[(Seq[Double], Seq[Double])], epochs: Int, batchSize: Int, eta: Double): NeuralNetwork = {
    var result = network

    0 until epochs foreach { _ =>
      Random.shuffle(data).grouped(batchSize) foreach { batch =>
        result = SGD(result, batch, eta)
      }
    }

    result
  }

  // Basic MSE cost function. io is an array of (input, expected outcome) tuples.
  def cost(network: NeuralNetwork, io: Seq[(Seq[Double], Seq[Double])]): Double =
    (distances(network)(io) map { d => d * d }).sum / 2 * io.size

  private def distances(network: NeuralNetwork)(io: Seq[(Seq[Double], Seq[Double])]): Seq[Double] =
    io map { case (input, expected) => distance(activations(network)(input).last, expected) }

  private def values(network: NeuralNetwork)(input: Seq[Double]): Seq[Seq[Double]] = {
    var values = Seq(input) // "Result" of first layer are the inputs

    network.layers foreach { case (weights, biases) =>
      values :+= weights zip biases map { case (w, b) => dot(values.last, w) + b }
    }

    values
  }

  private def activations(values: Seq[Seq[Double]]): Seq[Seq[Double]] = values map sigmoids

  private def activations(network: NeuralNetwork)(input: Seq[Double]): Seq[Seq[Double]] =
    activations(values(network)(input))

  private def SGD(network: NeuralNetwork, miniBatch: Seq[(Seq[Double], Seq[Double])], eta: Double = 3.0) = {
    val gradients = miniBatch map (gradient(network) _).tupled

    val weightSum = gradients map { _._1 } reduceLeft sums
    val adjustedWeightSum = weightSum map mulm(eta / miniBatch.size)
    val updatedWeights = network.weights zip adjustedWeightSum map (subs _).tupled

    val biasSum = gradients map { _._2 } reduceLeft sum
    val adjustedBiasSum = biasSum map mulv(eta / miniBatch.size)
    val updatedBiases = subs(network.biases, adjustedBiasSum)

    NeuralNetwork(updatedWeights, updatedBiases)
  }

  private def gradient(network: NeuralNetwork)(input: Seq[Double], expected: Seq[Double]) = {
    val vs: Seq[Seq[Double]] = values(network)(input)
    val as: Seq[Seq[Double]] = activations(vs)

    // Backpropagation, output layer first.
    var delta: Seq[Double] = mul(sub(as.last, expected), sigmoidPrimes(vs.last))
    var biasGradient: Seq[Seq[Double]] = Seq(delta)
    var weightGradient: Seq[Seq[Seq[Double]]] = Seq(dots(delta, as.dropRight(1).last))

    // Compute errors for the rest of the layers, back to front
    (network.layers.size-1 to 1 by -1) foreach { l =>
      delta = mul(dotm(transpose(network.weights(l)), delta), sigmoidPrimes(vs(l-1)))
      biasGradient :+= delta
      weightGradient :+= dots(delta, as(l-1))
    }

    (weightGradient.reverse, biasGradient.reverse)
  }
}