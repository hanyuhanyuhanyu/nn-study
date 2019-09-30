inp = (-9..9).to_a.map{|a| a.*(0.1).round(2)}.combination(3).to_a
puts "def test_data():"
print "    inp = "
p inp
out = inp.map{|arr|
  a, b, c = arr
  [
    (a + b + c).*(0.1).round(5),
    arr.count{|i| i > 0}.*(0.1).round(2),
  ]
}
print "    out = "
p out
puts '    return {"inp": inp, "out": out}'
