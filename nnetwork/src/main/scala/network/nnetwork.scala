package network

import breeze.linalg.DenseMatrix
import breeze.numerics._
import scala.annotation.tailrec

trait Network {
  def feedforward:DenseMatrix[Double]
  def cost:DenseMatrix[Double]
  def activation:Double
  def activation_derivative:Double
}


class NNetwork(val layers: List[Int]) extends Network  {

  // initialize layers in network
  var weights = (layers zip layers.tail) map (x => DenseMatrix.rand(x._1,x._2))
  var biases  = ((layers.tail) map  (x => DenseMatrix.rand(1,x))
//  var y = DenseMatrix.zeros[Double](5,1)

  def activation(z:Double): Double = {
    (1.0/(1.0+ exp(-1.0*z)))
  }

  def activation_derivative(z:Double): Double = {
    activation(z)*(1.0-activation(z))
  }

  def cost(y_hat: DenseMatrix[Double], y:DenseMatrix[Double]): DenseMatrix[Double] = {
    y.t-y_hat
  }

  @tailrec
  def feedforward(xs: DenseMatrix[Double]): DenseMatrix[Double] = {
    ((weights zip biases).foldLeft(xs)((s,x) => s*x._1 + x._2)) map (activation)
  }

  def get_states(xs: DenseMatrix[Double]): List[DenseMatrix[Double]] = {
    var states: List[DenseMatrix[Double]] = List()
    states = states:+xs
    ((weights zip biases).foldLeft(xs)((s,x) =>{
      states = states :+ (s*x._1+x._2)
      s*x._1+x._2
    }))
    states
  }
  @tailrec
  def backpropogation(y_out:DenseMatrix[Double] , y:DenseMatrix[Double],xs: DenseMatrix[Double]): List[DenseMatrix[Double]] = {
    var wval: List[DenseMatrix[Double]] = List()
    var bval: List[DenseMatrix[Double]] = List()
    var states = get_states(xs).reverse
    var DELTA = cost(y_out,y)

    weights.foldRight(DELTA)((w,d)=> {

      wval = wval :+ (w*d.reshape(w.cols,1))
      w*d.reshape(w.cols,1)

    })
    DELTA = DELTA*:* (states(0) map (activation_derivative))
    var states_activation_prime = states.tail.map(x => x map ( y => activation_derivative(y)))
    var delta_weights = wval map (x => x map activation_derivative)
    var layer_activation = states.tail.map(x => x map ( y => activation(y)))
    delta_weights = ((delta_weights zip layer_activation) map (x => x._1*x._2.t))
    delta_weights
  }

  def adjust_weights(delta_weights: List[DenseMatrix[Double]]):Unit = {
    for( j <- (0 until delta_weights.length))
    {
      weights.updated(j,delta_weights(j))
    }
  }
  def train(xs:List[(DenseMatrix[Double],DenseMatrix[Double])]): Unit = {
    for (z <- xs){
      var network_output = feedforward((z._1))
      var delta_cost = cost(network_output,z._2)
      var delta_weights = backpropogation(network_output,z._2,z._1)
      adjust_weights((delta_weights))
    }
  }
}
