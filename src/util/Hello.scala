package util

object Hello {
  def main(args: Array[String]) {

    println("Hello world!");
    var a = List(1, 2, 3, 4, 5);
    println(a);
    var b = a.reduce(_+_)
    println(b);
  }
}