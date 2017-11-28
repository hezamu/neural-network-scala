// Some linear algebra operations. Should really use a library for these...
object MatrixOps {
  def distance(v1: Seq[Double], v2: Seq[Double]): Double = Math.sqrt(sub(v1, v2) map { Math.pow(_, 2)} sum)

  def mul(v1: Seq[Double], v2: Seq[Double]): Seq[Double] = (v1 zip v2) map { case (n1, n2) => n1 * n2 }

  def sub(v1: Seq[Double], v2: Seq[Double]): Seq[Double] = (v1 zip v2) map { case (n1, n2) => n1 - n2 }

  def mulv(n: Double)(m: Seq[Double]): Seq[Double] = m map { _ * n }

  def mulm(n: Double)(m: Seq[Seq[Double]]): Seq[Seq[Double]] = m map mulv(n)

  def subs(m1: Seq[Seq[Double]], m2: Seq[Seq[Double]]): Seq[Seq[Double]] = (m1 zip m2) map (sub _).tupled

  def add(v1: Seq[Double], v2: Seq[Double]): Seq[Double] = (v1 zip v2) map { case (n1, n2) => n1 + n2 }

  def sum(m1: Seq[Seq[Double]], m2: Seq[Seq[Double]]) = m1 zip m2 map (add _).tupled

  def sums(m1: Seq[Seq[Seq[Double]]], m2: Seq[Seq[Seq[Double]]]): Seq[Seq[Seq[Double]]] = m1 zip m2 map (sum _).tupled

  def dot(v1: Seq[Double], v2: Seq[Double]): Double = mul(v1, v2) sum

  def dots(v1: Seq[Double], v2: Seq[Double]): Seq[Seq[Double]] = v1 map { v1v => v2 map { _ * v1v } }

  def dotm(m: Seq[Seq[Double]], v: Seq[Double]): Seq[Double] = m map { mv => dot(mv, v) }

  def sigmoid(z: Double): Double = 1.0 / (1.0 + Math.exp(-1.0 * z))

  def sigmoids(zs: Seq[Double]): Seq[Double] = zs map sigmoid

  def sigmoidPrime(z: Double): Double = sigmoid(z) * (1 - sigmoid(z))

  def sigmoidPrimes(zs: Seq[Double]): Seq[Double] = zs map sigmoidPrime

  def transpose(m: Seq[Seq[Double]]): Seq[Seq[Double]] = 0 until m(0).size map { x =>
    0 until m.size map { y => m(y)(x) }
  }
}
