# Python Tutorial

## 量子状態
### 量子状態の生成
量子状態は<code>QuantumState</code>クラスを用いて生成します。生成した量子状態は \f$|0\rangle^{\otimes n}\f$ に初期化されています。量子状態を<code>print</code>することで量子状態の情報を表示できます。
```python
from qulacs import QuantumState

# 2-qubitの状態を生成
n = 2
state = QuantumState(n)
print(state)
```
```sh
 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(1,0)
(0,0)
(0,0)
(0,0)
```
メモリが不足している場合は量子状態を生成できません。一般的なノートパソコンやデスクトップPCでは26,27量子ビット程度が限界です。

### 量子状態の初期化
生成した量子状態は<code>set_computational_basis</code>関数で計算基底に初期化したり、<code>set_Haar_random_state</code>関数でランダムな状態に初期化することが出来ます。
なお、Qulacsでは量子ビットのカウントは0から始まり、かつ|0000>と書いたときに一番右のビットが0-th qubitになります(他のライブラリや教科書では一番左が0-th qubitであることもあります。)
```python
from qulacs import QuantumState

n = 2
state = QuantumState(n)

# |00>に初期化
state.set_zero_state()

# 基底を二進数と見た時の整数値を入れて、その状態に初期化
state.set_computational_basis(1)
print(state)

# シードを指定してランダムな初期状態を生成
# (シードを省略した場合は時刻を用いてシードを決定します。)
seed = 0
state.set_Haar_random_state(seed)
print(state)
```
```
 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(0,0)
(1,0)
(0,0)
(0,0)

 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
  (0.0531326,0.551481)
    (0.382474,0.10121)
(-0.499658,-0.0931227)
  (0.474029,-0.231262)
```

### 状態ベクトルの取得と設定
量子状態の状態ベクトルは<code>get_vector</code>関数でnumpy arrayとして取得できます。また、<code>load</code>関数でnumpy arrayを量子状態に設定できます。
```python
import numpy as np
from qulacs import QuantumState

n = 2
state = QuantumState(n)

# 状態ベクトルを取得
vec = state.get_vector()
print(type(vec), vec.dtype)
print(vec)

# 状態ベクトルを設定
myvec = np.array([0.5,0.5,0.5,0.5])
state.load(myvec)
print(state)
```
```
<class 'numpy.ndarray'> complex128
[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(0.5,0)
(0.5,0)
(0.5,0)
(0.5,0)
```

### 量子状態のデータのコピーとロード
量子状態は<code>copy</code>関数で複製できます。また、<code>load</code>関数に量子状態を入れることで他の量子状態のデータをロードできます。
```python
from qulacs import QuantumState

n = 2
state = QuantumState(n)
state.set_computational_basis(0b11)

# コピーして新たな量子状態を作成
second_state = state.copy()
print(second_state)

# 量子状態を新たに生成し、既存の状態のベクトルをコピー
third_state = QuantumState(n)
third_state.load(state)
print(third_state)
```
```
 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(0,0)
(0,0)
(0,0)
(1,0)

 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(0,0)
(0,0)
(0,0)
(1,0)
```

### 量子状態に関する情報の計算

量子状態に対して、量子状態の状態を変えずに量子状態に関する情報を計算できます。例えば、指定した添え字のqubitを測定した時に、0を得る確率は<code>get_zero_probability</code>で計算できます。

```python
from qulacs import QuantumState

n = 5
state = QuantumState(n)
state.set_Haar_random_state(0)

# 指定した添え字のqubitをZ基底で測定して0を得る確率の計算
index = 3
zero_probability = state.get_zero_probability(index)
print("prob_meas_3rd : ",zero_probability)
```
```
prob_meas_3rd :  0.6015549753834679
```

量子状態を測定した時の結果をサンプリングするには<code>sampling</code>関数が使えます。関数の引数はサンプリングするデータの個数です。
```python
import numpy as np
from qulacs import QuantumState

n = 2
state = QuantumState(n)
state.load([1/np.sqrt(2), 0, 0.5, 0.5])
data = state.sampling(10)
print(data)
print([format(value, "b").zfill(2) for value in data]) # 二進数表示
```
```
[0, 3, 3, 3, 3, 0, 2, 3, 0, 3]
['00', '11', '11', '11', '11', '00', '10', '11', '00', '11']
```

このほかにも多くの関数が用意されています。詳しくは次の章を参照してください。


### 量子状態の内積
量子状態の内積は<code>inner_product</code>関数で計算できます。
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
(0.1265907720817918+0.10220657039660046j)
```

## 量子ゲート

### 量子ゲートの生成
量子ゲートは<code>qulacs.gate</code>モジュールの中で定義されています。このモジュールでは幾つかの有名な量子ゲートが既に定義されています。ゲートは作用する量子ビットの添え字を引数として生成できます。量子ゲートを<code>print</code>することで情報を表示できます。

```python
from qulacs.gate import X

target_index = 1
x_gate = X(target_index)
print(x_gate)
```
```
 *** gate info ***
 * gate name : X
 * target    :
 1 : commute X
 * control   :
 * Pauli     : yes
 * Clifford  : yes
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 ```

### 量子ゲートの作用
量子ゲートは<code>update_quantum_state</code>関数で量子状態を更新できます。例えば、<code>X</code>ゲートを1st qubitに作用するには以下のようなコードを書きます。

```python
from qulacs import QuantumState
from qulacs.gate import X

n = 2
state = QuantumState(n)
print(state)

index = 1
x_gate = X(index)
x_gate.update_quantum_state(state)
print(state)
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

 *** Quantum State ***
 * Qubit Count : 2
 * Dimension   : 4
 * State vector :
(0,0)
(0,0)
(1,0)
(0,0)
```

### 様々な量子ゲート
下記にしばしば使う名前の付いたゲートを紹介します。どのゲートも<code>update_quantum_state</code>関数を用いて量子状態を更新できます。その他のゲートについては、次の章を参照してください。
```python
import numpy as np

# パウリゲート、アダマールゲート、Tゲート
from qulacs.gate import X, Y, Z, H, T
target = 2
x_gate = X(target)
y_gate = Y(target)
z_gate = Z(target)
h_gate = H(target)
t_gate = T(target)

# パウリ回転ゲート
from qulacs.gate import RX, RY, RZ
angle = np.pi / 4.0
rx_gate = RX(target, angle)
ry_gate = RY(target, angle)
rz_gate = RZ(target, angle)

# CNOT, CZ, SWAPゲート
from qulacs.gate import CNOT, CZ, SWAP
control = 1
target2 = 1
cnot_gate = CNOT(control, target)
cz_gate = CZ(control, target)
swap_gate = SWAP(target, target2)
```

### 一般的な量子ゲート
量子ゲートの行列をnumpy arrayで指定してゲートを生成するには、<code>DenseMatrix</code>クラスを用います。一つ目の引数が作用する添え字で、二つ目が行列です。1量子ビットゲートの場合は一つの整数と2×2行列を与えます。
```python
from qulacs.gate import DenseMatrix

gate = DenseMatrix(1, [[0,1],[1,0]])
print(gate)
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
(0,0) (1,0)
(1,0) (0,0)
```

２量子ビット以上の大きさのゲートを作るには、一つ目の引数に対象となる添え字のリストを、二つ目に行列を与えます。行列の大きさは \f$2^n\f$ 次元でなければなりません。

```python
from qulacs.gate import DenseMatrix

gate = DenseMatrix([0,1]], [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
print(gate)
```
```
 *** gate info ***
 * gate name : DenseMatrix
 * target    :
 5 : commute
 3 : commute
 * control   :
 * Pauli     : no
 * Clifford  : no
 * Gaussian  : no
 * Parametric: no
 * Diagonal  : no
 * Matrix
(0,0) (1,0) (0,0) (0,0)
(1,0) (0,0) (0,0) (0,0)
(0,0) (0,0) (0,0) (1,0)
(0,0) (0,0) (1,0) (0,0)
```
なお、添え字を指定するときは、与えた添え字の順番がゲート行列の列や行を数えるときの順番と一致しているため、上記の例で<code>[0,1]</code>と<code>[0,1]</code>では意味が異なることに注意してください。


### 量子ゲートのゲート行列の取得
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

