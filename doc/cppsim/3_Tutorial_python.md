# Python Tutorial

## 量子状態

### 量子状態の生成
以下のコードで<code>n</code>qubitの量子状態を生成します。
生成した量子状態は \f$|0\rangle^{\otimes n}\f$ に初期化されています。
```python
from pycppsim import QuantumState

# 5-qubitの状態を生成
n = 5
state = QuantumState(n)

# |00000>に初期化
state.set_zero_state()
```
メモリが不足している場合は量子状態を生成できません。


### 量子状態のデータの取得
量子状態を表す \f$2^n\f$ の長さの配列を取得します。特にGPUで量子状態を作成したり、大きい \f$n\f$ では非常に重い操作になるので注意してください。
```python
from pycppsim import QuantumState

n = 5
state = QuantumState(n)
state.set_zero_state()

# 状態ベクトルをnumpy arrayとして取得
data = state.get_vector()
print(data)
```

### 量子状態の初期化
生成した量子状態は、計算基底に初期化したり、ランダムな状態に初期化することが出来ます。
```python
from pycppsim import QuantumState

n = 5
state = QuantumState(n)
state.set_zero_state()

# |00101>に初期化
state.set_computational_basis(0b00101)
print(state.get_vector())

# ランダムな初期状態を生成
state.set_Haar_random_state()
print(state.get_vector())

# シードを指定してランダムな初期状態を生成
seed = 0
state.set_Haar_random_state(seed)
print(state.get_vector())
```

### 量子状態のデータのコピーとロード
量子状態を複製したり、他の量子状態のデータをロードできます。
```python
from pycppsim import QuantumState

n = 5
state = QuantumState(n)
state.set_computational_basis(0b00101)

# コピーして新たな量子状態を作成
second_state = state.copy()
print(second_state.get_vector())

# 量子状態を新たに生成し、既存の状態のベクトルをコピー
third_state = QuantumState(n)
third_state.load(state)
print(third_state.get_vector())
```



### 古典レジスタの操作
量子状態は古典レジスタを持っており、読み書きを行えます。
```python
from pycppsim import QuantumState

n = 5
state = QuantumState(n)
state.set_zero_state()

# 3rd classical registerに値1をセット
register_position = 3
register_value = 1
state.set_classical_value(register_position, register_value)

# 3rd classical registerの値を取得
obtained_value = state.get_classical_value(register_position)
print(obtained_value)
```

### 量子状態に関する計算
量子状態には種々の処理が可能です。
```python
from pycppsim import QuantumState

n = 5
state = QuantumState(n)
state.set_Haar_random_state()

# normの計算
norm = state.get_norm()
print("norm : ",norm)

# Z基底で測定した時のentropyの計算
entropy = state.get_entropy() 
print("entropy : ",entropy)

# index-th qubitをZ基底で測定して0を得る確率の計算
index = 3
zero_probability = state.get_zero_probability(index)
print("prob_meas_3rd : ",zero_probability)

# 周辺確率を計算 (以下は0,3-th qubitが0、1,2-th qubitが1と測定される確率の例)
value_list = [0,1,1,0,2]
marginal_probability = state.get_marginal_probability(value_list)
print("marginal_prob : ",marginal_probability)
```

### 量子状態の内積
<code>inner_product</code>関数で内積を計算できます。
```python
from pycppsim import QuantumState
from pycppsim.state import inner_product

n = 5
state_bra = QuantumState(n)
state_ket = QuantumState(n)
state_bra.set_Haar_random_state()
state_ket.set_computational_basis(0)

# 内積値の計算
value = inner_product(state_bra, state_ket)
print(value)
```

### 量子状態の解放
delを用いて量子状態を強制的にメモリから解放することができます。
delせずとも利用されなくなったタイミングで解放されますが、メモリがシビアな際に便利です。
```python
from pycppsim import QuantumState

n = 5
state = QuantumState(n)

# 量子状態を開放
del state
```

### 量子状態の詳細情報の取得
オブジェクトを直接printすると、量子状態の情報が出力されます。
```python
from pycppsim import QuantumState

n = 5
state = QuantumState(n)

print(state)
```


## 量子ゲート

### 量子ゲートの生成と作用
デフォルトで実装されている量子ゲートはgateモジュールで定義されます。

```python
import numpy as np
from pycppsim import QuantumState
from pycppsim.gate import X, RY, DenseMatrix

n = 3
state = QuantumState(n)
state.set_zero_state()
print(state.get_vector())

# 1st-qubitにX操作
index = 1
x_gate = X(index)
x_gate.update_quantum_state(state)
print(state.get_vector())

# 1st-qubitをYパウリでpi/4.0回転
angle = np.pi / 4.0
ry_gate = RY(index, angle)
ry_gate.update_quantum_state(state)
print(state.get_vector())

# 2nd-qubitにゲート行列で作成したゲートを作用
dense_gate = DenseMatrix(2, [[0,1],[1,0]])
dense_gate.update_quantum_state(state)
print(state.get_vector())

# ゲートの解放
del x_gate
del ry_gate
del dense_gate
```

事前に定義されているゲートは以下の通りです。
- single-qubit Pauli operation: Identity, X,Y,Z
- single-qubit Clifford operation : H,S,Sdag, T,Tdag,sqrtX,sqrtXdag,sqrtY,sqrtYdag
- two-qubit Clifford operation : CNOT, CZ, SWAP
- single-qubit Pauli rotation : RX, RY, RZ
- General Pauli operation : Pauli, PauliRotation
- IBMQ basis-gate : U1, U2, U3
- General gate : DenseMatrix
- Measurement : Measurement
- Noise : BitFlipNoise, DephasingNoise, IndepenedentXZNoise, DepolarizingNoise


### 量子ゲートの合成
続けて作用する量子ゲートを合成し、新たな単一の量子ゲートを生成できます。これにより量子状態へのアクセスを減らせます。
```python
import numpy as np
from pycppsim import QuantumState
from pycppsim.gate import X, RY, merge

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

### 量子ゲートのゲート行列の和
量子ゲートのゲート要素の和を取ることができます。
(control-qubitがある場合の和は現状動作が未定義なので利用しないでください。)
```python
import numpy as np
from pycppsim import QuantumState
from pycppsim.gate import P0,P1,add, merge, Identity, X, Z

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

### 特殊な量子ゲートと一般の量子ゲート
cppsimにおける基本量子ゲートは以下の二つに分けられます。
- 特殊ゲート：そのゲートの作用について、専用の高速化がなされた関数があるもの 。
- 一般ゲート：ゲート行列を保持し、行列をかけて作用するもの。

前者は後者に比べ専用の関数が作成されているため高速ですが、コントロール量子ビットを増やすなど、量子ゲートの作用を変更する操作が後から行えません。
こうした変更をしたい場合、特殊ゲートを一般ゲートに変換してやらねばなりません。
これは<code>gate.to_matrix_gate</code>で実現できます。以下がその例になります。

```python
import numpy as np
from pycppsim import QuantumState
from pycppsim.gate import to_matrix_gate, X
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


### 量子ゲートのゲート行列の取得
生成した量子ゲートのゲート行列を取得できます。control量子ビットなどはゲート行列に含まれません。露にゲート行列を持たない種類のゲート（例えばn-qubitのパウリ回転ゲート）などは取得に非常に大きなメモリと時間を要するので気を付けてください。
```python
import numpy as np
from pycppsim import QuantumState
from pycppsim.gate import X, RY, merge

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

### 量子ゲートの情報の取得
printに流し込むことで、量子ゲートの情報を取得できます。行列要素をあらわに持つゲート(一般ゲート)の場合のみ、ゲート行列も表示されます。
```python
from pycppsim.gate import X, to_matrix_gate
gate = X(0)
print(gate)
print(to_matrix_gate(gate))
```


### 一般的な量子ゲートの実現
cppsimでは量子情報における種々のマップを以下の形で実現します。

- ユニタリ操作
量子ゲートとして実現します。

- 射影演算子やクラウス演算子など
量子ゲートとして実現します。一般に作用後に量子状態のノルムは保存されません。<code>DenseMatrix</code>関数により生成できます。
```python
from pycppsim.gate import DenseMatrix

# 1-qubit gateの場合
gate = DenseMatrix(0, [[0,1],[1,0]])
print(gate)

# 2-qubit gateの場合
gate = DenseMatrix([0,1], [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
print(gate)
```



- 確率的なユニタリ操作
<code>Probabilistic</code>関数を用いて、複数のユニタリ操作と確率分布を与えて作成します。

```python
from pycppsim.gate import Probabilistic, X, Y

distribution = [0.1, 0.2, 0.3]
gate_list = [X(0), Y(0), X(1)]
gate = Probabilistic(distribution, gate_list)
```
確率の総和が1に満たない場合、残りの確率はIdentityの作用になります。

- CPTP-map
<code>CPTP</code>関数に完全性を満たすクラウス演算子のリストとして与えて作成します。

```python
from pycppsim.gate import merge,CPTP, P0,P1

gate00 = merge(P0(0),P0(1))
gate01 = merge(P0(0),P1(1))
gate10 = merge(P1(0),P0(1))
gate11 = merge(P1(0),P1(1))

gate_list = [gate00, gate01, gate10, gate11]
gate = CPTP(gate_list)
```

- POVM
数値計算上にはInstrumentと同じなので、Instrumentとして実現します。

- Instrument
Instrumentは一般のCPTP-mapの操作に加え、ランダムに作用したクラウス演算子の添え字を取得する操作です。例えば、Z基底での測定は<code>P0</code>と<code>P1</code>からなるCPTP-mapを作用し、どちらが作用したかを知ることに相当します。
cppsimでは<code>Instrument</code>関数にCPTP-mapの情報と、作用したクラウス演算子の添え字を書きこむ古典レジスタのアドレスを指定することで実現します。

```python
from pycppsim import QuantumState
from pycppsim.gate import merge,Instrument, P0,P1

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

<!--
- Adaptive
古典レジスタに書き込まれた値に応じて操作を行ったり行わなかったりします。cppsimでは<code>[unsigned int]</code>型のレジスタを引数として受け取り、<code>bool</code>型を返す関数を指定し、これを実現します。

```python
from pycppsim.gate import Adaptive, X

classical_pos = 0
def func(list):
    return list[2]==1
gate = Adaptive(X(0), func)

state = QuantumState(2)
state.set_Haar_random_state()
gate.update_quantum_state(state)
result = state.get_classical_value(classical_pos)
print(result)
```
-->

- CP-map
Kraus-rankが1の場合は、上記の単体のクラウス演算子として扱ってください。それ以外の場合は、TPになるようにクラウス演算子を調整した後、<code>multiply_scalar</code>関数で定数倍にした<code>Identity</code>オペレータを作用するなどして調整してください。






## 量子回路

### 量子回路の構成
量子回路は量子ゲートの集合として表されます。
例えば以下のように量子回路を構成できます。

```python
from pycppsim import QuantumState, QuantumCircuit
from pycppsim.gate import Z
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

量子ゲートをまとめて一つの量子ゲートとすることで、量子ゲートの数を減らすことができ、数値計算の時間を短縮できることがあります。（もちろん、対象となる量子ビットの数が増える場合や、専用関数を持つ量子ゲートを合成して専用関数を持たない量子ゲートにしてしまった場合は、トータルで計算時間が減少するかは状況に寄ります。）

下記のコードでは<code>optimize</code>関数を用いて、量子回路の量子ゲートをターゲットとなる量子ビットが3つになるまで貪欲法で合成を繰り返します。

```python
from pycppsim import QuantumCircuit
from pycppsim.circuit import QuantumCircuitOptimizer
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
from pycppsim import QuantumCircuit
from pycppsim.circuit import QuantumCircuitOptimizer
n = 5
depth = 10
circuit = QuantumCircuit(n)
for d in range(depth):
    for i in range(n):
        circuit.add_H_gate(i)
print(circuit)
```


## ハミルトニアン
### ハミルトニアンの生成
ハミルトニアンはパウリ演算子の集合として表現されます。
パウリ演算子は下記のように定義できます。
```python
from pycppsim import Hamiltonian
n = 5
coef = 2.0
# 2.0 X_0 X_1 Y_2 Z_4というパウリ演算子を設定
Pauli_string = "X 0 X 1 Y 2 Z 4"
hamiltonian = Hamiltonian(n)
hamiltonian.add_operator(coef,Pauli_string)
```

### ハミルトニアンの評価
状態に対してハミルトニアンの期待値を評価できます。
```python
from pycppsim import Hamiltonian, QuantumState

n = 5
coef = 2.0
Pauli_string = "X 0 X 1 Y 2 Z 4"
hamiltonian = Hamiltonian(n)
hamiltonian.add_operator(coef,Pauli_string)

state = QuantumState(n)
state.set_Haar_random_state()
# 期待値の計算
value = hamiltonian.get_expectation_value(state)
print(value)
```
