# Python Tutorial (Advanced)

この章は量子情報の用語をある程度知っていたり、数値計算のための細かなチューニングをしたい人のための解説です。用語の詳細についてはM.A.Nielsenらによる教科書Quantum Computation and Quantum Information、または[量子情報科学入門](https://www.kyoritsu-pub.co.jp/bookdetail/9784320122994)などを参照してください。

## QuantumStateクラス
complex128の精度で\f$2^n\f$個の複素配列をCPU/GPU上に確保し管理するクラスです。状態ベクトルの移動や変換の他、状態に関する情報の計算や変形をサポートします。

### 生成と破棄
インスタンス生成時に必要なメモリが確保されます。メモリはpythonがインスタンスを破棄した段階で解放されますが、メモリ解放のために明示的に破棄したい場合は<code>del</code>で解放できます。<code>\_\_repr\_\_</code>関数のオーバーライドにより状態ベクトルのフォーマットされた表示を提供します。
```python
from qulacs import QuantumState
n = 2
state = QuantumState(n)
print(state)
del state
```
```
 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(1,0)
(0,0)
(0,0)
(0,0)
```
### 量子状態とnumpy array間の変換
また、<code>get_vector</code>および<code>load</code>関数で量子状態とnumpy arrayの相互変換が可能です。ノルムが保存しているかなどは原則チェックされません。
```python
from qulacs import QuantumState

state = QuantumState(2)
vec = state.get_vector()
print(vec)
state.load([0,1,2,3])
print(state.get_vector())
```
```
[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
[0.+0.j 1.+0.j 2.+0.j 3.+0.j]
```

### 量子状態間のコピー
量子状態は<code>copy</code>で自身と同じインスタンスを新たに生成できます。また<code>load</code>関数に量子状態を与えることで、既存の量子状態に新たに領域を確保することなく別の量子状態の量子ベクトルをコピーすることが出来ます。これにより既に確保した領域を再利用できます。既に持っている量子状態と同じサイズの状態ベクトルを確保して、状態のコピーはしなくてよい場合は<code>allocate_buffer</code>関数を使えます。
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

n = 3
state = QuantumState(n)
# |0>状態へ初期化
state.set_zero_state()
print(state.get_vector())

# 指定値の二進数表記の計算基底へ初期化
state.set_computational_basis(0b101)
print(state.get_vector())

# 引数の値をシードとしてハール測度でランダムな純粋状態へ初期化
# 指定値が無い場合はtime関数がシードとして使われる。疑似乱数はxorshiftを利用。
state.set_Haar_random_state(0)
print(state.get_vector())
```
```
[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
[0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j]
[ 0.02537775+0.26340418j  0.1826813 +0.04834094j -0.23865176-0.04447825j
  0.22641072-0.11045776j  0.07783625+0.24911921j  0.45253202+0.38419963j
  0.09236077-0.08936245j -0.5348115 -0.22094398j]
```

### 量子状態の検査
下記は量子状態を変えずに量子状態の情報を調べる関数の一覧です。
```python
from qulacs import QuantumState

n = 5
state = QuantumState(n)
state.set_Haar_random_state(0)

# 量子ビットの数を得る。
qubit_count = state.get_qubit_count()
print("qubit_count", qubit_count)

# 指定番目の量子ビットが0に測定される確率を得る
prob = state.get_zero_probability(1)
print("zero_prob_1", prob)

# 任意の周辺確率を得る
# 引数は量子ビット数と同じ長さの配列
# 0,1,2を指定する。0,1はその添え字がその値で測定される確率、
# 2はそのビットを周辺化することを意味する。
# 例えば、3番目が0で、0番目が1と測定される確率の計算は下記
prob = state.get_marginal_probability([1,2,2,0,2])
print("marginal_prob", prob)

# Z基底で測定した時の確率分布のエントロピーを得る
ent = state.get_entropy()
print("entropy", ent)

# squared norm (<a|a>)の取得
# Trace preservingでない操作が可能なため、状態のノルムが1とは限らない
sq_norm = state.get_squared_norm()
print("sqaured_norm", sq_norm)

# 引数で与えた数の回数Z基底で全量子ビットを測定しサンプリングする。
# 得られるバイナリを整数値にしたもののリストを得る。
samples = state.sampling(10)
print("sampling", samples)

# 状態ベクトルがCPU/GPUのどちらにあるかを文字列で取得する
dev_type = state.get_device_name()
print("device", dev_type)
```

```
qubit_count 5
zero_prob_1 0.5151171741097239
marginal_prob 0.40401228349316776
entropy 3.0788233113265715
sqaured_norm 1.0
sampling [23, 9, 8, 4, 25, 7, 7, 24, 20, 27]
device cpu
```

### 量子状態の変形

下記の関数は量子状態を書き換える関数です。
```python
from qulacs import QuantumState
state = QuantumState(2)
state.set_computational_basis(0)
buffer = QuantumState(2)
buffer.set_computational_basis(2)
print("state" , state.get_vector())
print("buffer", buffer.get_vector())

# 量子状態間の和(state <- state+buffer)
# stateにbufferの状態を足し重ね合わせ状態を作ります。 
# 操作後のノルムは一般に1ではありません。
state.add_state(buffer)
print("added", state.get_vector())

# 量子状態と複素数の積
# 引数の複素数を全要素に掛けます。
# 操作後のノルムは一般に1ではありません。
coef = 0.5 + 0.1j
state.multiply_coef(coef)
print("mul_coef", state.get_vector())

# 量子状態の正規化
# 引数として現在のsquared normを与える必要があります。
squared_norm = state.get_squared_norm()
print("sq_norm", squared_norm)
state.normalize(squared_norm)
print("normalized", state.get_vector())
print("sq_norm", state.get_squared_norm())
```
```
state [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
buffer [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
added [1.+0.j 0.+0.j 1.+0.j 0.+0.j]
mul_coef [0.5+0.1j 0. +0.j  0.5+0.1j 0. +0.j ]
sq_norm 0.52
normalized [0.69337525+0.13867505j 0.        +0.j         0.69337525+0.13867505j
 0.        +0.j        ]
sq_norm 0.9999999999999998
```
### 古典レジスタの操作

量子状態は可変長の整数配列を古典レジスタを持ちます。古典レジスタはInstrument操作の結果を書き込んだり、古典レジスタの結果を条件として実行するゲートを記述するのに用います。まだ書き込まれていない古典レジスタの値は0です。なお、古典レジスタは<code>copy,load</code>関数で量子状態を複製した際に同時に複製されます。

```python
from qulacs import QuantumState
state = QuantumState(3)
position = 0
# position番目にvalueを書き込みます
value = 20
state.set_classical_value(position, value)
# position番目のレジスタ値を得ます
obtained = state.get_classical_value(position)
print(obtained)
```
```
20
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
```
(-0.09816013083684771-0.03592414662479273j)
```

### GPUを用いた計算
Qulacsをqulacs-gpuパッケージからインストールした場合、<code>QuantumStateGpu</code>クラスが使用できます。クラス名が異なる以外、利用方法は<code>QuantumState</code>と同じです。
```python
from qulacs import QuantumStateGpu
state = QuantumStateGpu(2)
print(state)
print(state.get_device_name())
```
```
 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(1,0)
(0,0)
(0,0)
(0,0)

gpu
```
使い方は<code>QuantumState</code>と同様ですが、二点留意点があります。
1. <code>get_vector</code>関数はGPU/CPU間のコピーを要するため長い時間がかかります。出来る限りこの関数の利用を回避して計算を行うべきです。
2. CPU/GPUの状態間の<code>inner_product</code>は計算できません。GPUとCPUの状態ベクトルの間で状態ベクトルの<code>load</code>を行うことは可能ですが、時間がかかるので避けるべきです。

## 量子ゲート

### 量子ゲートに共通の操作

### 量子ゲートの種類
量子ゲートは特殊ゲートと一般ゲートの二種類にわかれます。なお、Qulacsではユニタリ演算子に限らず、InstrumentやCPTP-mapをはじめとする任意の量子状態を更新する操作をゲートと呼びます。

特殊ゲートは事前に指定されたゲート行列を持ち、量子ゲートに対し限定された変形しか行えないものを指します。例えば、パウリゲート、パウリでの回転ゲート、射影測定などが対応します。特殊ゲートの利点は、ゲートの特性が限定されているため量子状態の更新関数が一般ゲートに比べ効率的である点です。また、定義時に自身が各量子ビットで何らかのパウリの基底で対角化されるかの情報を保持しており、この情報は回路最適化の時に利用されます。特殊ゲートの欠点は上記の理由からゲートに対する可能な操作が限定されている点です。

作用するゲート行列を露に持つゲートを一般ゲートと呼びます。一般ゲートの利点はゲート行列を好きに指定できるところですが、欠点は特殊ゲートに比べ更新が低速である点です。

### 量子ゲートに共通の操作

生成した量子ゲートのゲート行列は<code>get_matrix</code>関数で取得できます。control量子ビットなどはゲート行列に含まれません。特にゲート行列を持たない種類のゲート（例えばn-qubitのパウリ回転ゲート）などは取得に非常に大きなメモリと時間を要するので気を付けてください。<code>print</code>関数を使用してゲートの情報を表示できます。
```python
import numpy as np
from qulacs.gate import X
gate = X(2)
mat = gate.get_matrix()
print(mat)
print(gate)
```
```
[[0.+0.j 1.+0.j]
 [1.+0.j 0.+0.j]]
 *** gate info ***
 * gate name : X
 * target    :
 2 : commute X
 * control   :
 * Pauli     : yes
 * Clifford  : yes
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 ```

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
target = 3
gate = T(target)
print(gate)
```
```
 *** gate info ***
 * gate name : T
 * target    :
 3 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : yes
 * Parametric: no
 * Diagonal  : no
 ```
<code>Identity</code>は量子状態を更新しませんが、量子回路に入れると1stepを消費するゲートとしてカウントされます。

#### 1量子ビット回転ゲート
第一引数に対象ビットの添え字を、第二引数に回転角を取ります。
```python
import numpy as np
from qulacs.gate import RX, RY, RZ
target = 0
angle = 0.1
gate = RX(target, angle)
print(gate)
print(gate.get_matrix())
```
```
 *** gate info ***
 * gate name : X-rotation
 * target    :
 0 : commute X
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 [[0.99875026+0.j         0.        +0.04997917j]
 [0.        +0.04997917j 0.99875026+0.j        ]]
 ```
回転操作の定義は\f$R_X(\theta) = \exp(i\frac{\theta}{2} X)\f$です。

#### IBMQの基底ゲート
IBMQのOpenQASMで定義されている、virtual-Z分解に基づくゲートです。
```python
from qulacs.gate import U1,U2,U3
print(U3(0, 0.1, 0.2, 0.3))
```
```
 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 0 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
            (0.99875,0) (-0.0477469,-0.0147699)
 (0.0489829,0.00992933)     (0.876486,0.478826)
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
control = 5
target = 2
target2 = 3
gate = CNOT(control, target)
print(gate)
gate = CZ(control, target)
gate = SWAP(target, target2)
```
```
 *** gate info ***
 * gate name : CNOT
 * target    :
 2 : commute X
 * control   :
 5 : value 1
 * Pauli     : no
 * Clifford  : yes
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
```

#### 多ビットパウリ操作
多ビットパウリ操作はターゲット量子ビットのリストとパウリ演算子のリストを引数としてゲートを定義します。n-qubitパウリ操作の更新測度は1-qubitパウリ操作の更新コストとオーダーが同じため、パウリのテンソル積はこの形式でゲートを定義した方が多くの場合得です。パウリ演算子の指定は1,2,3がそれぞれX,Y,Zに対応します。
```python
from qulacs.gate import Pauli
target_list = [0,3,5]
pauli_index = [1,3,1] # 1:X , 2:Y, 3:Z
gate = Pauli(target_list, pauli_index) # = X_0 Z_3 X_5
print(gate)
print(gate.get_matrix())
```
```
 *** gate info ***
 * gate name : Pauli
 * target    :
 0 : commute X
 3 : commute     Z
 5 : commute X
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j -1.-0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j -1.-0.j  0.+0.j]
 [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j -1.-0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j -1.-0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
 ```

#### 多ビットパウリ回転操作
多ビットパウリ演算子の回転操作です。多ビットのパウリ回転は愚直にゲート行列を計算すると大きいものになりますが、この形で定義すると効率的に更新が可能です。
```python
from qulacs.gate import PauliRotation
target_list = [0,3,5]
pauli_index = [1,3,1] # 1:X , 2:Y, 3:Z
angle = 0.5
gate = PauliRotation(target_list, pauli_index, angle) # = exp(i angle/2 X_0 Z_3 X_5)
print(gate)
print(gate.get_matrix().shape)
```
```
 *** gate info ***
 * gate name : Pauli-rotation
 * target    :
 0 : commute X
 3 : commute     Z
 5 : commute X
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 (8, 8)
 ```

#### 可逆回路
\f$2^n\f$個の添え字に対する全単射関数を与えることで、基底間の置換操作を行います。ゲート行列が置換行列になっていることと同義です。全単射でない場合正常に動作しないため注意してください。
```python
from qulacs.gate import ReversibleBoolean
def upper(val, dim):
	return (val+1)%dim
target_list = [0,1]
gate = ReversibleBoolean(target_list, upper)
print(gate)
state = QuantumState(3)
state.load(np.arange(2**3))
print(state.get_vector())
gate.update_quantum_state(state)
print(state.get_vector())
```
```
 *** gate info ***
 * gate name : ReversibleBoolean
 * target    :
 0 : commute
 1 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no

[0.+0.j 1.+0.j 2.+0.j 3.+0.j 4.+0.j 5.+0.j 6.+0.j 7.+0.j]
[3.+0.j 0.+0.j 1.+0.j 2.+0.j 7.+0.j 4.+0.j 5.+0.j 6.+0.j]
```
上記のコードは対象の量子ビットの部分空間でベクトルの要素を一つずつ下に下げます(一番下の要素は一番上に動きます)。

#### 状態反射
量子状態|a>を引数として定義される、(I-2|a><a|>)というゲートです。これは|a>という量子状態をもとに反射する操作に対応します。グローバー探索で登場するゲートです。このゲートが作用する相手の量子ビット数は、引数として与えた量子状態の量子ビット数と一致しなければいけません。
```python
from qulacs.gate import StateReflection
from qulacs import QuantumState
axis = QuantumState(2)
axis.set_Haar_random_state(0)
state = QuantumState(2)
gate = StateReflection(axis)
gate.update_quantum_state(state)
print("axis", axis.get_vector())
print("reflected", state.get_vector())
gate.update_quantum_state(state)
print("two reflection", state.get_vector())
```
```
axis [ 0.0531326 +0.55148118j  0.38247419+0.10120994j -0.49965781-0.09312274j
  0.47402911-0.231262j  ]
reflected [-0.38609087+0.j          0.15227445-0.41109954j -0.15580712+0.54120805j
 -0.20470048-0.54741137j]
two reflection [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
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
```
 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 0 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
(0,0) (1,0)
(1,0) (0,0)

 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 0 : commute
 1 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
(1,0) (0,0) (0,0) (0,0)
(0,0) (1,0) (0,0) (0,0)
(0,0) (0,0) (0,0) (1,0)
(0,0) (0,0) (1,0) (0,0)
```


#### 疎行列ゲート
疎行列を元に定義されるゲートです。要素が十分疎である場合、密行列より高速に更新が可能です。疎行列はscipyのcsc_matrixを用いて定義してください。

```python
from qulacs import QuantumState
from qulacs.gate import SparseMatrix
from scipy.sparse import csc_matrix
mat = csc_matrix((2,2))
mat[1,1] = 1
print("sparse matrix", mat)

gate = SparseMatrix([0], mat)
print(gate)

qs = QuantumState(2)
qs.load([1,2,3,4])
gate.update_quantum_state(qs)
print(qs.get_vector())
```
```
sparse matrix   (1, 1)  (1+0j)

 *** gate info ***
 * gate name : SparseMatrix
 * target    :
 0 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
0 0
0 (1,0)

[0.+0.j 2.+0.j 0.+0.j 4.+0.j]
```

#### コントロールビットの追加
一般ゲートは<code>add_control_qubit</code>関数を用いてcontrol qubitを加えることが可能です。control qubitが0,1のどちらの場合にtargetに作用が生じるかも指定できます。
```python
import numpy as np
from qulacs.gate import to_matrix_gate, X

index = 0
x_gate = X(index)
x_mat_gate = to_matrix_gate(x_gate)

# 1st-qubitが0の場合だけゲートを作用
control_index = 1
control_with_value = 0
x_mat_gate.add_control_qubit(control_index, control_with_value)
print(x_mat_gate)

from qulacs import QuantumState
state = QuantumState(3)
state.load(np.arange(2**3))
print(state.get_vector())

x_mat_gate.update_quantum_state(state)
print(state.get_vector())
```
```
 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 0 : commute X
 * control   :
 1 : value 0
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
(0,0) (1,0)
(1,0) (0,0)

[0.+0.j 1.+0.j 2.+0.j 3.+0.j 4.+0.j 5.+0.j 6.+0.j 7.+0.j]
[1.+0.j 0.+0.j 2.+0.j 3.+0.j 5.+0.j 4.+0.j 6.+0.j 7.+0.j]
```

### 複数のゲートから新たなゲートを作る操作


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
print(x_and_ry_gate)
```
```
 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 1 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
 (0.382683,0)   (0.92388,0)
  (0.92388,0) (-0.382683,0)
```

#### ゲートの和
複数のゲートの和を取り、新たなゲートを作ることが出来ます。例えばパウリ演算子\f$P\f$に対して\f$(I+P)/2\f$といった+1固有値空間への射影を作るときに便利です。
```python
import numpy as np
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
```
 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 0 : commute
 1 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
(1,0) (0,0) (0,0) (0,0)
(0,0) (0,0) (0,0) (0,0)
(0,0) (0,0) (0,0) (0,0)
(0,0) (0,0) (0,0) (1,0)

 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 0 : commute
 1 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
(0.5,0)   (0,0)   (0,0) (0.5,0)
  (0,0)   (0,0)   (0,0)   (0,0)
  (0,0)   (0,0)   (0,0)   (0,0)
(0.5,0)   (0,0)   (0,0) (0.5,0)
```

#### ランダムユニタリ
ハール測度でランダムなユニタリ行列をサンプリングし、密行列ゲートを生成するには<code>RandomUnitary</code>関数を用います。
```python
from qulacs.gate import RandomUnitary
target_list = [2,3]
gate = RandomUnitary(target_list)
print(gate)
```
```
 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 2 : commute
 3 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
 (-0.259248,-0.150756)  (-0.622614,-0.539728) (-0.0289836,-0.154895)    (0.437381,0.122371)
  (0.0853439,0.215282)    (-0.157238,0.20972)   (-0.471882,-0.39828)   (-0.201449,0.675116)
   (0.780141,0.161283)  (-0.374972,-0.279089)  (-0.0835221,0.258196)  (-0.238478,-0.127904)
(-0.469496,-0.0370718)  (-0.123494,-0.136361)   (-0.304384,0.653894)    (-0.457639,0.12122)
```

#### 確率的作用
<code>Probabilistic</code>関数を用いて、複数のゲート操作と確率分布を与えて作成します。与える確率分布の総和が1に満たない場合、満たない確率でIdentityが作用します。

```python
from qulacs.gate import Probabilistic, H, Z
distribution = [0.2, 0.2, 0.2]
gate_list = [H(0), Z(0), X(1)]
gate = Probabilistic(distribution, gate_list)
print(gate)

from qulacs import QuantumState
state = QuantumState(2)
for _ in range(10):
	gate.update_quantum_state(state)
	print(state.get_vector())
```
```
 *** gate info ***
 * gate name : Generic gate
 * target    :
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : yes

[ 1.+0.j -0.-0.j  0.+0.j -0.-0.j]
[ 1.+0.j -0.-0.j  0.+0.j -0.-0.j]
[ 1.+0.j -0.-0.j  0.+0.j -0.-0.j]
[ 1.+0.j -0.-0.j  0.+0.j -0.-0.j]
[0.70710678+0.j 0.70710678+0.j 0.        +0.j 0.        +0.j]
[0.70710678+0.j 0.70710678+0.j 0.        +0.j 0.        +0.j]
[0.70710678+0.j 0.70710678+0.j 0.        +0.j 0.        +0.j]
[0.70710678+0.j 0.70710678+0.j 0.        +0.j 0.        +0.j]
[0.        +0.j 0.        +0.j 0.70710678+0.j 0.70710678+0.j]
[0.+0.j 0.+0.j 1.+0.j 0.+0.j]
```

なお、確率的作用をするゲートとして、<code>BitFlipNoise</code>, <code>DephasingNoise</code>, <code>IndependentXZNoise</code>, <code>DepolarizingNoise</code>, <code>TwoQubitDepolarizingNoise</code>ゲートが定義されています。それぞれ、エラー確率を入れることで<code>Probabilistic</code>のインスタンスが生成されます。

```python
from qulacs.gate import BitFlipNoise, DephasingNoise, IndependentXZNoise, DepolarizingNoise, TwoQubitDepolarizingNoise
target = 0
second_target = 1
error_prob = 0.8
gate = BitFlipNoise(target, error_prob) # X: prob
gate = DephasingNoise(target, error_prob) # Z: prob
gate = IndependentXZNoise(target, error_prob) # X,Z : prob*(1-prob), Y: prob*prob
gate = DepolarizingNoise(target, error_prob) # X,Y,Z : prob/3
gate = TwoQubitDepolarizingNoise(target, second_target, error_prob) # {I,X,Y,Z} \times {I,X,Y,Z} \setminus {II} : prob/15

from qulacs import QuantumState
state = QuantumState(2)
for _ in range(10):
	gate.update_quantum_state(state)
	print(state.get_vector())
```
```
[0.-0.j 0.+0.j 0.-0.j 0.+1.j]
[0.-0.j 0.+0.j 0.-0.j 0.+1.j]
[0.+0.j 0.+1.j 0.+0.j 0.+0.j]
[0.-0.j 0.+0.j 0.+0.j 1.-0.j]
[-1.+0.j  0.+0.j  0.+0.j  0.+0.j]
[ 0.+0.j  0.+0.j -1.+0.j  0.+0.j]
[-1.+0.j -0.+0.j  0.+0.j -0.+0.j]
[-1.+0.j  0.-0.j  0.+0.j  0.-0.j]
[ 0.+0.j -1.+0.j  0.+0.j  0.+0.j]
[ 0.+0.j -1.+0.j  0.+0.j  0.+0.j]
```

#### CPTP写像
<code>CPTP</code>は完全性を満たすクラウス演算子のリストを与えて作成します。

```python
from qulacs.gate import merge,CPTP, P0,P1

gate00 = merge(P0(0),P0(1))
gate01 = merge(P0(0),P1(1))
gate10 = merge(P1(0),P0(1))
gate11 = merge(P1(0),P1(1))

gate_list = [gate00, gate01, gate10, gate11]
gate = CPTP(gate_list)

from qulacs import QuantumState
from qulacs.gate import H,merge
state = QuantumState(2)
for _ in range(10):
	state.set_zero_state()
	merge(H(0),H(1)).update_quantum_state(state)
	gate.update_quantum_state(state)
	print(state.get_vector())
```
```
[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
[0.+0.j 0.+0.j 1.+0.j 0.+0.j]
[0.+0.j 0.+0.j 1.+0.j 0.+0.j]
[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
[0.+0.j 1.+0.j 0.+0.j 0.+0.j]
[0.+0.j 0.+0.j 0.+0.j 1.+0.j]
[0.+0.j 0.+0.j 0.+0.j 1.+0.j]
[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
[0.+0.j 0.+0.j 1.+0.j 0.+0.j]
```
なお、CPTP-mapとして<code>AmplitudeDampingNoise</code>ゲートが定義されています。

```python
from qulacs.gate import AmplitudeDampingNoise
target = 0
damping_rate = 0.1
AmplitudeDampingNoise(target, damping_rate) # K_0: [[1,0],[0,sqrt(1-p)]], K_1: [[0,sqrt(p)], [0,0]]
```
#### Instrument
Instrumentは一般のCPTP-mapの操作に加え、ランダムに作用したクラウス演算子の添え字を取得する操作です。例えば、Z基底での測定は<code>P0</code>と<code>P1</code>からなるCPTP-mapを作用し、どちらが作用したかを知ることに相当します。
cppsimでは<code>Instrument</code>関数にCPTP-mapの情報と、作用したクラウス演算子の添え字を書きこむ古典レジスタのアドレスを指定することで実現します。

```python
from qulacs import QuantumState
from qulacs.gate import merge,Instrument, P0,P1

gate00 = merge(P0(0),P0(1))
gate01 = merge(P1(0),P0(1))
gate10 = merge(P0(0),P1(1))
gate11 = merge(P1(0),P1(1))

gate_list = [gate00, gate01, gate10, gate11]
classical_pos = 0
gate = Instrument(gate_list, classical_pos)

from qulacs import QuantumState
from qulacs.gate import H,merge
state = QuantumState(2)
for index in range(10):
	state.set_zero_state()
	merge(H(0),H(1)).update_quantum_state(state)
	gate.update_quantum_state(state)
	result = state.get_classical_value(classical_pos)
	print(index, format(result,"b").zfill(2), state.get_vector())
```
```
0 11 [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
1 11 [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
2 00 [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
3 01 [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
4 10 [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
5 10 [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
6 01 [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
7 01 [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
8 01 [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
9 11 [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
```
なお、Instrumentとして<code>Measurement</code>ゲートが定義されています。

```python
from qulacs.gate import Measurement
target = 0
classical_pos = 0
gate = Measurement(target, classical_pos)
```

#### Adaptive操作
古典レジスタの値の可変長リストを引数としブール値を返す関数を用いて、古典レジスタから求まる条件に応じて操作を行うか決定するゲートです。条件はpythonの関数として記述することができます。pythonの関数は<code>unsigned int</code>型のリストを引数として受け取り、<code>bool</code>型を返す関数でなくてはなりません。

```python
from qulacs.gate import Adaptive, X

def func(list):
	print("func is called! content is ",list)
	return list[0]==1


gate = Adaptive(X(0), func)

state = QuantumState(1)
state.set_zero_state()

# funcがFalseを返すため、Xは作用しない
state.set_classical_value(0,0)
gate.update_quantum_state(state)
print(state.get_vector())

# funcがTrueを返すため、Xが作用する
state.set_classical_value(0,1)
gate.update_quantum_state(state)
print(state.get_vector())
```
```
func is called! content is  [0]
[1.+0.j 0.+0.j]
func is called! content is  [1]
[0.+0.j 1.+0.j]
```



## 物理量
### パウリ演算子
オブザーバブルは実係数を持つパウリ演算子の線形結合として表現されます。<code>PauliOperator</code>クラスはその中のそれぞれの項を表す、$n$-qubitパウリ演算子の元に係数を付与したものを表現するクラスです。ゲートと異なり、量子状態の更新はできません。

#### パウリ演算子の生成と状態の取得
```python
from qulacs import PauliOperator
coef = 0.1
s = "X 0 Y 1 Z 3"
pauli = PauliOperator(s, coef)

# pauliの記号を後から追加
pauli.add_single_Pauli(3, 2)

# pauliの各記号の添え字を取得
index_list = pauli.get_index_list()

# pauliの各記号を取得 (I,X,Y,Z -> 0,1,2,3)
pauli_id_list = pauli.get_pauli_id_list()

# pauliの係数を取得
coef = pauli.get_coef()

# pauli演算子のコピーを作成
another_pauli = pauli.copy()

s = ["I","X","Y","Z"]
pauli_str = [s[i] for i in pauli_id_list]
terms_str = [item[0]+str(item[1]) for item in zip(pauli_str,index_list)]
full_str = str(coef) + " " + " ".join(terms_str)
print(full_str)
```
```
(0.1+0j) X0 Y1 Z3 Y3
```

#### パウリ演算子の期待値
状態に対してパウリ演算子の期待値や遷移モーメントを評価できます。
```python
from qulacs import PauliOperator, QuantumState

n = 5
coef = 2.0
Pauli_string = "X 0 X 1 Y 2 Z 4"
pauli = Pauli(Pauli_string,coef)

# 期待値の計算 <a|H|a>
state = QuantumState(n)
state.set_Haar_random_state()
value = pauli.get_expectation_value(state)
print("expect", value)

# 遷移モーメントの計算 <a|H|b>
# 第一引数がブラ側に来る
bra = QuantumState(n)
bra.set_Haar_random_state()
value = pauli.get_transition_amplitude(bra, state)
print("transition", value)
```

```
expect (-0.013936248917618807-0j)
transition (-0.009179829550387531-0.02931360609180049j)
```

### 一般の線形演算子
線形演算子<code>GeneralQuantumOperator</code>はパウリ演算子の複素数の線形結合で表されます。係数付きのPauliOperatorを項として<code>add_operator</code>で追加することが出来ます。
```python
from qulacs import GeneralQuantumOperator, PauliOperator, QuantumState

n = 5
operator = GeneralQuantumOperator(n)

# pauli演算子を追加できる
coef = 2.0+0.5j
Pauli_string = "X 0 X 1 Y 2 Z 4"
pauli = Pauli(Pauli_string,coef)
operator.add_operator(pauli)
# 直接係数と文字列から追加することもできる
operator.add_operator(0.5j, "Y 1 Z 4")

# 項の数を取得
term_count = operator.get_term_count()

# 量子ビット数を取得
qubit_count = operator.get_qubit_count()

# 特定の項をPauliOperatorとして取得
index = 1
pauli = operator.get_term(index)


# 期待値の計算 <a|H|a>
## 一般に自己随伴ではないので複素が帰りうる
state = QuantumState(n)
state.set_Haar_random_state()
value = operator.get_expectation_value(state)
print("expect", value)

# 遷移モーメントの計算 <a|H|b>
# 第一引数がブラ側に来る
bra = QuantumState(n)
bra.set_Haar_random_state()
value = operator.get_transition_amplitude(bra, state)
print("transition", value)
```
```
expect (0.01844802681960955+0.05946837146359432j)
transition (-0.00359496979054156+0.0640782452494485j)
```

#### OpenFermionを用いたオブザーバブルの生成
OpenFermionは化学計算で解くべきハミルトニアンをパウリ演算子の表現で与えてくれるツールです。このツールの出力をファイルまたは文字列の形で読み取り、演算子の形で使用することが可能です。

```python
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_file
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text

open_fermion_text = """
(-0.8126100000000005+0j) [] +
(0.04532175+0j) [X0 Z1 X2] +
(0.04532175+0j) [X0 Z1 X2 Z3] +
(0.04532175+0j) [Y0 Z1 Y2] +
(0.04532175+0j) [Y0 Z1 Y2 Z3] +
(0.17120100000000002+0j) [Z0] +
(0.17120100000000002+0j) [Z0 Z1] +
(0.165868+0j) [Z0 Z1 Z2] +
(0.165868+0j) [Z0 Z1 Z2 Z3] +
(0.12054625+0j) [Z0 Z2] +
(0.12054625+0j) [Z0 Z2 Z3] +
(0.16862325+0j) [Z1] +
(-0.22279649999999998+0j) [Z1 Z2 Z3] +
(0.17434925+0j) [Z1 Z3] +
(-0.22279649999999998+0j) [Z2]
"""

operator = create_quantum_operator_from_openfermion_text(open_fermion_text)
print(operator.get_term_count())
print(operator.get_qubit_count())
# create_quantum_operator_from_openfermion_fileの場合は上記が書かれたファイルのパスを引数で指定する。
```

### エルミート演算子/オブザーバブル
エルミート演算子はパウリ演算子の実数での線形結合で表されます。固有値あるいは期待値が実数であることを保証される以外、<code>GeneralQuatnumOperator</code>クラスと等価です。

外部ファイルから読み込んで処理をする関数は<code>quantum_operator</code>を<code>observable</code>に置き換えて
<code>create_observable_from_openfermion_file</code>
<code>create_observable_from_openfermion_text</code>
<code>create_split_observable</code>
などの関数で可能です。

#### 演算子を対角項と非対角な項に分離する
演算子をファイルからで読み込む際、<code>create_split_observable</code>関数で対角成分と非対角成分に分離できます。

```python
from qulacs.observable import create_split_observable, create_observable_from_openfermion_file

# 事前にH2.txtをopenfermonの形式で配置する必要があります。
operator = create_observable_from_openfermion_file("./H2.txt")
diag, nondiag = create_split_observable("./H2.txt")
print(operator.get_term_count(), diag.get_term_count(), nondiag.get_term_count())
print(operator.get_qubit_count(), diag.get_qubit_count(), nondiag.get_qubit_count())
```


## 量子回路

### 量子回路の構成
量子回路は量子ゲートの集合として表されます。
例えば以下のように量子回路を構成できます。

```python
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import Z
n = 3
state = QuantumState(n)
state.set_zero_state()

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
```
[ 0.35355339+0.j -0.35355339-0.j -0.35355339-0.j  0.35355339+0.j
 -0.35355339-0.j  0.35355339+0.j  0.35355339+0.j -0.35355339-0.j]
```

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
```
10
1
```

### 量子回路の情報デバッグ
量子回路を<code>print</code>すると、量子回路に含まれるゲートの統計情報などが表示されます。

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
```
*** Quantum Circuit Info ***
# of qubit: 5
# of step : 10
# of gate : 50
# of 1 qubit gate: 50
Clifford  : yes
Gaussian  : no
```


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
```
 *** Quantum State ***
 * Qubit Count : 5
 * Dimension   : 32
 * State vector :
   (0.187449,0.0161955)
 (-0.179316,0.00524451)
 (0.00347095,0.0476141)
   (0.106624,0.0178808)
(-0.0577202,0.00342914)
    (0.101846,0.280231)
  (-0.187288,-0.210536)
  (0.0558002,0.0337467)
  (0.0180534,0.0594363)
     (0.06433,0.076302)
   (0.109149,-0.167976)
(0.00171454,-0.0767738)
   (0.128206,-0.107731)
   (0.273894,-0.122758)
   (0.0994833,0.151077)
   (0.112889,-0.313486)
(-0.00982741,0.0165689)
   (0.11043,-0.0646575)
(-0.0923695,-0.0794695)
  (-0.0198962,0.150974)
(-0.0580249,-0.0885592)
  (-0.048759,-0.196734)
 (-0.0940465,-0.215696)
   (0.106312,0.0912926)
  (-0.177757,-0.128979)
   (0.0940203,0.149268)
  (0.0702079,0.0503984)
 (-0.232558,0.00717037)
   (0.150701,0.0325937)
  (0.0645294,-0.164578)
   (-0.092721,0.178244)
(-0.0107883,-0.0478668)

*** Quantum Circuit Info ***
# of qubit: 5
# of step : 41
# of gate : 171
# of 1 qubit gate: 150
# of 2 qubit gate: 20
# of 3 qubit gate: 0
# of 4 qubit gate: 1
Clifford  : no
Gaussian  : no

*** Parameter Info ***
# of parameter: 151
```