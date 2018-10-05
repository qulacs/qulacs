# How to install

## 環境について
下記の環境での動作を確認しています。

- gcc-7 @ Ubuntu
- gcc-7 @ MacOS 
- gcc-7 @ Windows-msys2-MINGW64
- gcc-7 @ Windows-cygwin64
- MSVC 15.0 Win64 @ Windows

## インストール

### 必要なファイルのインストール
#### Linux
gcc-7, g++-7, make, git, wget, cmake, python3が必要になります。
python3には、numpyをインストールしてください。

#### Mac OS
gcc-7, g++-7, make, git, wget, cmake, python3が必要になります。
Macのデフォルトのgccとg++はclangとなっているので、別途gccを導入してください。
python3には、numpyをインストールしてください。


#### Windows
msys2やcygwinを用いる場合gcc, g++, make, git, wget, cmake, python3をインストールしてください。python3には、numpyをインストールしてください。

Visual Studioを用いる場合、Visual Studio 2017以降と、git, cmake, python3をインストールしてください。Visual Studioとpython3の連携はAnaconda3での動作を確認しています。

### gccでのビルド

#### ライブラリのビルド
作業ディレクトリに移動し、以下のコマンドを実行してください。
```sh
git clone https://github.com/vqcsim/vqcsim.git
cd vqcsim
./get_eigen.sh
./get_gtest.sh
./get_pybind.sh
mkdir build
cd build
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ ..
```

gcc/g++が古かったり、MacOSでバックエンドがclangである場合は、上記のcmakeコマンドのgcc/g++をgcc-7,g++-7に置き換えてください。

並列化を無効にしたい場合はcmakeのフラグに"-D USE_OMP=no"を加えてください。

cmakeが完了したら、下記のコマンドでライブラリをビルドします。
```sh
make
```

#### ライブラリのテスト
下記のコマンドでgoogle testが実行されます。
```sh
make test
```

#### ライブラリのベンチマーク
下記のコマンドで簡易的なベンチマークが実行されます。
```sh
make bench
```

#### pythonライブラリの生成
下記のコマンドでpythonのライブラリが生成されます。
```sh
make python
```
エラーが生じてしまう場合はドキュメントのトラブルシューティングを参照してください。

#### pythonライブラリのテスト
下記のコマンドでpythonのライブラリのテストが実行されます。
```sh
make pythonpython
```


### Visual Studioでのビルド
#### プロジェクトファイルの作成
作業ディレクトリに移動し、以下のコマンドを実行してください。
```sh
git clone https://github.com/vqcsim/vqcsim.git
cd vqcsim
./get_eigen.sh
./get_gtest.sh
./get_pybind.sh
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -D CMAKE_BUILD_TYPE=Release ..
```
上記コマンドで、Project.slnが生成されるので、これを開いてください。

次に、上部ボックスでReleaseを選択し、メニューの「ビルド」からリビルドを選択します。



### 使い型

#### C++ライブラリの利用
例えばcppsimの用いるとき、
```
g++ -I ./vqcsim/include -I ./vqcsim/eigen_3_3_5 your_code.cpp ./vqcsim/lib/libcppsim.a
```
としてビルドできます。(Windowsの場合は拡張子は.aではなく.libです。)
関数のリストや実装例についてはドキュメントを参照してください。


### pythonライブラリの利用
pybind11を用いてcppsimがpython用のライブラリとして出力されています。

環境に応じて、拡張子が.dllか.pydのpycppsimライブラリが、binフォルダかlibフォルダに生成されます。
このdllをpythonスクリプトから参照できる位置に配置することで、以下のようにpythonから呼び出せます。
```python
from pycppsim import Hamiltonian, QuantumCircuit, QuantumState
from pycppsim.gate import Y,CNOT,merge

state = QuantumState(3)
state.set_Haar_random_state()

circuit = QuantumCircuit(3)
circuit.add_X_gate(0)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.append_gate(merged_gate)
circuit.add_RX_gate(1,0.5)
circuit.update_quantum_state(state)

ham = Hamiltonian(3)
ham.add_operator(2.0, "X 2 Y 1 Z 0")
ham.add_operator(-3.0, "Z 2")
value = ham.get_expectation_value(state)
print(value)
```
**pycppsimはpython3の関連ファイルでビルドされているので、pythonがpython3のエイリアスになっていない場合、必ずpython3からimportしてください。**

関数やクラスのリストは一部委嘱されていない関数を除き、cppsimと同じです。
実装例についてはドキュメントを参照してください。

