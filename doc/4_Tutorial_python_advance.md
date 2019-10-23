# Python Tutorial (Advanced)

量子情報の用語をある程度知っていたり、数値計算のための細かなチューニングをしたい人のための解説です。用語の詳細についてはM.A.Nielsenらによる教科書Quantum Computation and Quantum Information、または[量子情報科学入門](https://www.kyoritsu-pub.co.jp/bookdetail/9784320122994)などを参照してください。

## QuantumStateクラス
complex128の精度で\f$2^n\f$個の配列を管理するクラスインスタンスを生成します。状態ベクトルの移動や変換の他、状態に関する情報の計算や変形をサポートします。

### 生成と破棄
インスタンス生成時に必要なメモリが確保されます。メモリはpythonがインスタンスを破棄した段階で解放されますが、明示的に破棄したい場合は<code>del</code>で解放できます。<code>\_\_repr\_\_</code>関数のオーバーライドにより状態ベクトルのフォーマットされた表示を提供します。
```python
from qulacs import QuantumState
n = 2
state = QuantumState(n)
print(state)
del state
```

### 量子状態とnumpy arrayの相互変換
また、<code>get_vector</code>および<code>load</code>関数で量子状態とnumpy arrayの相互変換が可能です。
```python
from qulacs import QuantumState

state = QuantumState(3)
print(state)
vec = state.get_vector()
state.load(vec)
```

### 量子状態間の操作
量子状態は<code>copy</code>で新たに複製できます。また<code>load</code>関数に量子状態を与えることで、既存の量子状態に新たに領域を確保することなく別の量子状態の量子ベクトルをコピーすることが出来ます。これにより既に確保した領域を再利用できます。既に持っている量子状態と同じサイズの状態ベクトルを確保して、状態のコピーはしなくてよい場合は<code>allocate_buffer</code>関数を使えます。
```python
from qulacs import QuantumState

initial_state = QuantumState(3)
buffer = initial_state.allocate_buffer()
for ind in range(10):
	buffer.load(initial_state)
	# some computation and get results
```

### 量子状態の初期化
下記は量子状態を特定の状態に初期化する関数です。
```python
from qulacs import QuantumState

n = 5
state = QuantumState(n)
# |0>状態へ初期化
state.set_zero_state()
# 指定値の二進数表記の計算基底へ初期化
state.set_computational_basis(0b00101)
# 引数の値をシードとしてハール測度でランダムな純粋状態へ初期化
# 指定値が無い場合はtime関数がシードとして使われる。疑似乱数はxorshiftを利用。
state.set_Haar_random_state(0)
```

### 量子状態の検査
下記は量子状態を変えない操作の一覧です。
```python
from qulacs import QuantumState

n = 5
state = QuantumState(n)

# 量子ビットの数を得る。
state.get_qubit_count()

# 指定番目の量子ビットが0に測定される確率を得る
state.get_zero_probability(5)

# 任意の周辺確率を得る
# 引数は量子ビット数と同じ長さの配列
# 0,1,2を指定する。0,1はその添え字がその値で測定される確率、
# 2はそのビットを周辺化することを意味する。
# 例えば、3番目が0で、0番目が1と測定される確率の計算は下記
state.get_marginal_probability([1,2,2,0,2])

# Z基底で測定した時の確率分布のエントロピーを得る
state.get_entropy()

# squared norm (<a|a>)の取得
# Trace preservingでない操作が可能なため、状態のノルムが1とは限らない
state.get_squared_norm()

# 引数で与えた数の回数Z基底で全量子ビットを測定しサンプリングする。
# 得られるバイナリを整数値にしたもののリストを得る。
state.sampling(100)

# 状態ベクトルがCPU/GPUのどちらにあるかを文字列で取得する
state.get_device_name()
```

### 古典レジスタの操作

量子状態は可変長の整数配列を古典レジスタを持ちます。古典レジスタはInstrument操作の結果を書き込んだり、古典レジスタの結果を条件として実行するゲートを記述するのに用います。まだ書き込まれていない古典レジスタの値は0です。なお、古典レジスタは<code>copy,load</code>関数で量子状態を複製した際に同時に複製されます。

```python
from qulacs import QuantumState
state = QuantumState(3)
position = 0
# position番目のレジスタ値を得ます
state.get_classical_value(position)
# position番目にvalueを書き込みます
value = 20
state.get_classical_value(position, value)
```


### 量子状態の変形

下記の関数は量子状態を書き換える関数です。
```python
from qulacs import QuantumState
state = QuantumState(3)
buffer = QuantumState(3)

# 量子状態間の和(state <- state+buffer)
# stateにbufferの状態を足し重ね合わせ状態を作ります。 
# 操作後のノルムは一般に1ではありません。
state.add_state(buffer)

# 量子状態と複素数の積
# 引数の複素数を全要素に掛けます。
# 操作後のノルムは一般に1ではありません。
coef = 0.5 + 0.1j
state.multiply_coef(coef)

# 量子状態の正規化
# 引数として現在のsquared normを与える必要があります。
squared_norm = state.get_squared_norm()
state.normalize(squared_norm)
```


### 量子状態間の計算
量子状態間の内積は<code>inner_product</code>で得られます。
```python
from qulacs import QuantumState
from qulacs.state import inner_product

n = 5
state_bra = QuantumState(n)
state_ket = QuantumState(n)
state_bra.set_Haar_random_state()
state_ket.set_computational_basis(0)

# 内積値の計算
value = inner_product(state_bra, state_ket)
print(value)
```

### GPUを用いた計算
Qulacsをqulacs-gpuパッケージからインストールした場合、<code>QuantumStateGpu</code>クラスが使用できます。
```python
from qulacs import QuantumStateGpu
state = QuantumStateGpu(5)
```
使い方は<code>QuantumState</code>と同様ですが、二点留意点があります。
1. <code>get_vector</code>関数はGPU/CPU間のコピーを要するため長い時間がかかります。出来る限りこの関数の利用を回避して計算を行うべきです。
2. CPU/GPUの状態間の<code>inner_product</code>は計算できません。GPUとCPUの状態ベクトルの間で状態ベクトルの<code>load</code>を行うことは可能ですが、時間がかかるので避けるべきです。

## 量子ゲート

### 量子ゲートに共通の操作

#### 量子ゲートのゲート行列の取得
生成した量子ゲートのゲート行列を取得できます。control量子ビットなどはゲート行列に含まれません。特にゲート行列を持たない種類のゲート（例えばn-qubitのパウリ回転ゲート）などは取得に非常に大きなメモリと時間を要するので気を付けてください。
```python
import numpy as np
from qulacs import QuantumState
from qulacs.gate import X, RY, merge

n = 3
state = QuantumState(n)
state.set_zero_state()

index = 1
x_gate = X(index)
angle = np.pi / 4.0
ry_gate = RY(index, angle)
x_and_ry_gate = merge(x_gate, ry_gate)

# ゲート行列を取得
matrix = x_and_ry_gate.get_matrix()
print(matrix)
```

#### 量子ゲートの情報の取得
printに流し込むことで、量子ゲートの情報を取得できます。行列要素をあらわに持つゲート(一般ゲート)の場合のみ、ゲート行列も表示されます。
```python
from qulacs.gate import X, to_matrix_gate
gate = X(0)
print(gate)
print(to_matrix_gate(gate))
```



### 量子ゲートの種類
量子ゲートは特殊ゲートと一般ゲートの二種類にわかれます。なお、Qulacsではユニタリ演算子に限らず、InstrumentやCPTP-mapをはじめとする任意の量子状態を更新する操作をゲートと呼びます。

特殊ゲートは事前に指定されたゲート行列を持ち、量子ゲートに対し限定された変形しか行えないものを指します。例えば、パウリゲート、パウリでの回転ゲート、射影測定などが対応します。特殊ゲートの利点は、ゲートの特性が限定されているため量子状態の更新関数が一般ゲートに比べ効率的である点です。特殊ゲートの欠点は上記の理由からゲートに対する可能な操作が限定されている点です。

作用するゲート行列を露に持つゲートを一般ゲートと呼びます。一般ゲートの利点はゲート行列を好きに指定できるところですが、欠点は特殊ゲートに比べ更新が低速である点です。

任意の特殊ゲートは<code>qulacs.gate</code>モジュールの<code>to_matrix_gate</code>関数で一般ゲートに変形出来ます。一般ゲートを特殊ゲートに変形することはできません。

### 特殊ゲート
下記に特殊ゲートを列挙します。

#### 1量子ビットゲート
第一引数に対象ビットの添え字を取ります
```python
from qulacs.gate import Identity # 単位行列
from qulacs.gate import X, Y, Z	# パウリ
from qulacs.gate import H, S, Sdag, sqrtX, sqrtXdag, sqrtY, sqrtYdag # クリフォード
from qulacs.gate import T, Tdag # Tゲート
from qulacs.gate import P0, P1 # 0,1への射影 (規格化はされない)
gate = X(3)
```
<code>Identity</code>は量子状態を更新しませんが、量子回路に入れると1stepを消費するゲートとしてカウントされます。

#### 1量子ビット回転ゲート
第一引数に対象ビットの添え字を、第二引数に回転角を取ります。
```python
from qulacs.gate import RX, RY, RZ
```
回転操作の定義は\f$R_X(\theta) = \exp(i\frac{\theta}{2} X)\f$です。

#### IBMQの基底ゲート
IBMQのOpenQASMで定義されている、virtual-Z分解に基づくゲートです。
```python
from qulacs.gate import U1,U2,U3
```
定義はそれぞれ

- \f$U_1(\lambda) = R_Z(\lambda)\f$
- \f$U_2(\phi, \lambda) = R_Z(\phi+\frac{\pi}{2}) R_X(\frac{\pi}{2}) R_Z(\lambda-\frac{\pi}{2})\f$
- \f$U_3(\theta, \phi, \lambda) = R_Z(\phi+3\pi) R_X(\pi/2) R_Z(\theta+\pi) R_X(\pi/2) R_Z(\lambda)\f$

になります。U3は任意の1qubitユニタリ操作の自由度と一致します。

#### 2量子ビットゲート
第1,2引数に対象ビットの添え字を取ります。CNOTゲートは第一引数がcontrol qubitになります。残りのゲートは対称な操作です。
```python
from qulacs.gate import CNOT, CZ, SWAP
```

#### 多ビットパウリ操作
多ビットパウリ操作はターゲット量子ビットのリストとパウリ演算子のリストを引数としてゲートを定義します。n-qubitパウリ操作の更新測度は1-qubitパウリ操作の更新コストとオーダーが同じため、パウリのテンソル積はこの形式でゲートを定義した方が多くの場合得です。パウリ演算子の指定は1,2,3がそれぞれX,Y,Zに対応します。
```python
from qulacs.gate import Pauli
target_list = [0,3,5]
pauli_index = [1,3,1] # 1:X , 2:Y, 3:Z
gate = Pauli(target_list, pauli_index) # = X_0 Z_3 X_5
```

#### 多ビットパウリ回転
多ビットパウリ演算子の回転操作です。多ビットのパウリ回転は愚直にゲート行列を計算すると大きいものになりますが、この形で定義すると効率的に更新が可能です。
```python
from qulacs.gate import PauliRotation
target_list = [0,3,5]
pauli_index = [1,3,1] # 1:X , 2:Y, 3:Z
angle = 0.5
gate = PauliRotation(target_list, pauli_index, angle) # = exp(i angle/2 X_0 Z_3 X_5)
```

#### 可逆回路
\f$2^n\f$個の添え字に対する全単射関数を与えることで、基底間の置換操作を行います。ゲート行列が置換行列になっていることと同義です。
```python
from qulacs.gate import ReversibleBoolean
def upper(val, dim):
	return (val+1)%dim
target_list = [0,3,5]
gate = Reversibleboolean(target_qubit, func)
```
上記のコードは対象の量子ビットの部分空間でベクトルの要素を一つずつ下に下げます(一番下の要素は一番上に動きます)。

#### 状態反射
量子状態|a>を引数として定義される、(I-2|a><a|>)というゲートです。これは|a>という量子状態をもとに反射する操作に対応します。グローバー探索で登場するゲートです。このゲートが作用する相手の量子ビット数は、引数として与えた量子状態の量子ビット数と一致しなければいけません。
```python
from qulacs.gate import StateReflection
from qulacs import QuantumState
state = QuantumState(3)
def upper(val, dim):
	return (val+1)%dim
target_list = [0,3,5]
gate = StateReflection(state)
```

### 一般ゲート
ゲート行列をあらわに持つ量子ゲートです。

#### 密行列ゲート
密行列を元に定義されるゲートです。

```python
from qulacs.gate import DenseMatrix

# 1-qubit gateの場合
gate = DenseMatrix(0, [[0,1],[1,0]])
print(gate)

# 2-qubit gateの場合
gate = DenseMatrix([0,1], [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
print(gate)
```


#### 疎行列ゲート
疎行列を元に定義されるゲートです。要素が十分疎であるばあい、密行列より高速に更新が可能です。

### 一般ゲートに対する操作

#### コントロールビットの追加
ゲート同士について以下のような操作が可能です。
```python
import numpy as np
from qulacs import QuantumState
from qulacs.gate import to_matrix_gate, X
n = 3
state = QuantumState(n)
state.set_zero_state()

index = 0
x_gate = X(index)
x_mat_gate = to_matrix_gate(x_gate)

# 1st-qubitが0の場合だけゲートを作用
control_index = 1
control_with_value = 0
x_mat_gate.add_control_qubit(control_index, control_with_value)

x_mat_gate.update_quantum_state(state)
print(state.get_vector())
```

#### ゲートの積
続けて作用する量子ゲートを合成し、新たな単一の量子ゲートを生成できます。これにより量子状態へのアクセスを減らせます。
```python
import numpy as np
from qulacs import QuantumState
from qulacs.gate import X, RY, merge

n = 3
state = QuantumState(n)
state.set_zero_state()

index = 1
x_gate = X(index)
angle = np.pi / 4.0
ry_gate = RY(index, angle)

# ゲートを合成して新たなゲートを生成
# 第一引数が先に作用する
x_and_ry_gate = merge(x_gate, ry_gate)
x_and_ry_gate.update_quantum_state(state)
print(state.get_vector())
```

#### ゲートの和
複数のゲートの和を取り、新たなゲートを作ることが出来ます。例えばパウリ演算子\f$P\f$に対して\f$(I+P)/2\f$といった+1固有値空間への射影を作るときに便利です。
```python
import numpy as np
from qulacs import QuantumState
from qulacs.gate import P0,P1,add, merge, Identity, X, Z

gate00 = merge(P0(0),P0(1))
gate11 = merge(P1(0),P1(1))
# |00><00| + |11><11|
proj_00_or_11 = add(gate00, gate11)
print(proj_00_or_11)

gate_ii_zz = add(Identity(0), merge(Z(0),Z(1)))
gate_ii_xx = add(Identity(0), merge(X(0),X(1)))
proj_00_plus_11 = merge(gate_ii_zz, gate_ii_xx)
# ((|00>+|11>)(<00|+<11|))/2 = (II + ZZ)(II + XX)/4
proj_00_plus_11.multiply_scalar(0.25)
print(proj_00_plus_11)
```

#### ランダムユニタリ
ハール測度でランダムなユニタリ行列をサンプリングし、密行列ゲートを生成します。
```
```

#### 確率的作用
<code>Probabilistic</code>関数を用いて、複数のユニタリ操作と確率分布を与えて作成します。

```python
from qulacs.gate import Probabilistic, X, Y

distribution = [0.1, 0.2, 0.3]
gate_list = [X(0), Y(0), X(1)]
gate = Probabilistic(distribution, gate_list)
```
なお、確率的作用をするゲートとして、BitFlipNoise, DephasingNoise, IndependentXZNoise, DepolarizingNoiseゲートが定義されています。それぞれ、エラー確率を入れることでProbabilisticのインスタンスが生成されます。

#### CPTP写像
<code>CPTP</code>関数に完全性を満たすクラウス演算子のリストとして与えて作成します。

```python
from qulacs.gate import merge,CPTP, P0,P1

gate00 = merge(P0(0),P0(1))
gate01 = merge(P0(0),P1(1))
gate10 = merge(P1(0),P0(1))
gate11 = merge(P1(0),P1(1))

gate_list = [gate00, gate01, gate10, gate11]
gate = CPTP(gate_list)
```
なお、CPTP-mapとしてAmplitudeDampingNoiseゲートが定義されています。
#### Instrument
Instrumentは一般のCPTP-mapの操作に加え、ランダムに作用したクラウス演算子の添え字を取得する操作です。例えば、Z基底での測定は<code>P0</code>と<code>P1</code>からなるCPTP-mapを作用し、どちらが作用したかを知ることに相当します。
cppsimでは<code>Instrument</code>関数にCPTP-mapの情報と、作用したクラウス演算子の添え字を書きこむ古典レジスタのアドレスを指定することで実現します。

```python
from qulacs import QuantumState
from qulacs.gate import merge,Instrument, P0,P1

gate00 = merge(P0(0),P0(1))
gate01 = merge(P0(0),P1(1))
gate10 = merge(P1(0),P0(1))
gate11 = merge(P1(0),P1(1))

gate_list = [gate00, gate01, gate10, gate11]
classical_pos = 0
gate = Instrument(gate_list, classical_pos)

state = QuantumState(2)
state.set_Haar_random_state()

print(state)
gate.update_quantum_state(state)
result = state.get_classical_value(classical_pos)
print(state)
print(result)
```
なお、InstrumentとしてMeasurementゲートが定義されています。

#### Adaptive操作
古典レジスタに書き込まれた値を用いた条件に応じて操作を行うか決定します。
条件はpythonの関数として記述することができます。pythonの関数は<code>unsigned int</code>型のリストを引数として受け取り、<code>bool</code>型を返す関数でなくてはなりません。

```python
from qulacs.gate import Adaptive, X

def func(list):
    return list[0]==1
gate = Adaptive(X(0), func)

state = QuantumState(2)
state.set_Haar_random_state()

# func returns False, and gate is not applied
gate.set_classical_value(0,0)
gate.update_quantum_state(state)

# func returns True, and gate is applied
gate.set_classical_value(0,1)
gate.update_quantum_state(state)
```



## 量子回路

### 量子回路の構成
量子回路は量子ゲートの集合として表されます。
例えば以下のように量子回路を構成できます。

```python
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import Z
n = 5
state = QuantumState(n)
state.set_zero_state()

# 量子回路を定義
circuit = QuantumCircuit(n)

# 量子回路にhadamardゲートを追加
for i in range(n):
    circuit.add_H_gate(i)

# ゲートを生成し、それを追加することもできる。
for i in range(n):
    circuit.add_gate(Z(i))

# 量子回路を状態に作用
circuit.update_quantum_state(state)

print(state.get_vector())
```

なお、<code>add_gate</code>で追加された量子回路は量子回路の解放時に一緒に解放されます。従って、代入したゲートは再利用できません。引数として与えたゲートを再利用したい場合はgate.copyを用いて自身のコピーを作成するか、<code>add_gate_copy</code>関数を用いてください。

### 量子回路のdepthの計算と最適化

量子ゲートをまとめて一つの量子ゲートとすることで、量子ゲートの数を減らすことができ、数値計算の時間を短縮できることがあります。（もちろん、対象となる量子ビットの数が増える場合や、専用関数を持つ量子ゲートを合成して専用関数を持たない量子ゲートにしてしまった場合は、トータルで計算時間が減少するかは状況に依ります。）

下記のコードでは<code>optimize</code>関数を用いて、量子回路の量子ゲートをターゲットとなる量子ビットが3つになるまで貪欲法で合成を繰り返します。

```python
from qulacs import QuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer
n = 5
depth = 10
circuit = QuantumCircuit(n)
for d in range(depth):
    for i in range(n):
        circuit.add_H_gate(i)

# depthを計算(depth=10)
print(circuit.calculate_depth())

# 最適化
opt = QuantumCircuitOptimizer()
# 作成を許す最大の量子ゲートのサイズ
max_block_size = 1
opt.optimize(circuit, max_block_size)

# depthを計算(depth=1へ)
print(circuit.calculate_depth())
```

### 量子回路の情報デバッグ
量子回路をprintすると、量子回路に含まれるゲートの統計情報などが表示されます。

```python
from qulacs import QuantumCircuit
from qulacs.circuit import QuantumCircuitOptimizer
n = 5
depth = 10
circuit = QuantumCircuit(n)
for d in range(depth):
    for i in range(n):
        circuit.add_H_gate(i)
print(circuit)
```


## オブザーバブル
### オブザーバブルの生成
オブザーバブルは実係数を持つパウリ演算子の線形結合として表現されます。パウリ演算子は下記のように定義できます。
```python
from qulacs import Observable
n = 5
coef = 2.0
# 2.0 X_0 X_1 Y_2 Z_4というパウリ演算子を設定
Pauli_string = "X 0 X 1 Y 2 Z 4"
observable = Observable(n)
observable.add_operator(coef,Pauli_string)
```

一般の複素係数によるパウリ演算子の線形結合は行列の基底をなしますが、一般に実固有値を持つとは限らないためオブザーバブルではありません。これらはGeneralPauliOperatorとして定義されます。

### オブザーバブルの評価
状態に対してオブザーバブルの期待値を評価できます。
```python
from qulacs import Observable, QuantumState

n = 5
coef = 2.0
Pauli_string = "X 0 X 1 Y 2 Z 4"
observable = Observable(n)
observable.add_operator(coef,Pauli_string)

state = QuantumState(n)
state.set_Haar_random_state()
# 期待値の計算
value = observable.get_expectation_value(state)
print(value)
```

### 遷移振動子の計算
ハミルトニアンHに対し<a|H|b>は遷移振動子と呼ばれます。一般の演算子に対し、transition_amplitude関数で遷移振動子を計算することが可能です。



## 変分量子回路
量子回路をParametricQuantumCircuitクラスとして定義すると、通所のQuantumCircuitクラスの関数に加え、変分法を用いて量子回路を最適化するのに便利ないくつかの関数を利用することができます。

### 変分量子回路の利用例

一つの回転角を持つ量子ゲート(X-rot, Y-rot, Z-rot, multi_qubit_pauli_rotation)はパラメトリックな量子ゲートとして量子回路に追加することができます。パラメトリックなゲートとして追加された量子ゲートについては、量子回路の構成後にパラメトリックなゲート数を取り出したり、後から回転角を変更することができます。

```python
from qulacs import ParametricQuantumCircuit
from qulacs import QuantumState
import numpy as np

n = 5
depth = 10

# construct parametric quantum circuit with random rotation
circuit = ParametricQuantumCircuit(n)
for d in range(depth):
	for i in range(n):
		angle = np.random.rand()
		circuit.add_parametric_RX_gate(i,angle)
		angle = np.random.rand()
		circuit.add_parametric_RY_gate(i,angle)
		angle = np.random.rand()
		circuit.add_parametric_RZ_gate(i,angle)
	for i in range(d%2, n-1, 2):
		circuit.add_CNOT_gate(i,i+1)

# add multi-qubit Pauli rotation gate as parametric gate (X_0 Y_3 Y_1 X_4)
target = [0,3,1,4]
pauli_ids = [1,2,2,1]
angle = np.random.rand()
circuit.add_parametric_multi_Pauli_rotation_gate(target, pauli_ids, angle)

# get variable parameter count, and get current parameter
parameter_count = circuit.get_parameter_count()
param = [circuit.get_parameter(ind) for ind in range(parameter_count)]

# set 3rd parameter to 0
circuit.set_parameter(3, 0.)

# update quantum state
state = QuantumState(n)
circuit.update_quantum_state(state)

# output state and circuit info
print(state)
print(circuit)
```