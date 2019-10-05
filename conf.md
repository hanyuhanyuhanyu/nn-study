# (any).__init__

キーワード引数が基本

普通の引数は本当に必須なのが明らかでかつ、ユーザが直接値を指定できるようなプログラムの表層で利用されるクラスが使うこと

# Layer.__init__

絶対にkwargsで初期値をもらうこと

# LayerFactory.createに渡すもの
{
    layers: {
        name string;
        setting dict
    }
}
