import scala.util.Random

object Main {
  def main(args: Array[String]): Unit = {
    // Input is a tuple of 2 numbers, one close to 1.0 and one to 0.0. Output
    // activation indicates which one was near 1.0.
    def makeData(n: Int) = 0 until n map { _ =>
      val r = (Seq(Random.nextDouble / 100, 0.9 + Random.nextDouble / 10), Seq(0.0, 1.0))
      if(Random.nextBoolean) (r._1.reverse, r._2.reverse) else r
    }

    // 2 input neurons, 2 hidden neurons and 2 output neurons
    val untrained = NeuralNetwork.initNetwork(Seq(2, 2, 2))

    val trainingData = makeData(5000)

    print("Training...")
    val trained = NeuralNetwork.train(untrained, trainingData, 10, 100, 3.0)

    val testData = makeData(10)

    val untrainedCost = NeuralNetwork.cost(untrained, testData)
    val trainedCost = NeuralNetwork.cost(trained, testData)
    val improvement = 100*(untrainedCost - trainedCost) / untrainedCost

    println(f" done. Cost improvement $improvement%.2f %%")
  }
}
