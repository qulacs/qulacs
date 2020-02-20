# C++ Tutorial

## 量子状態

### 量子状態の生成
以下のコードで<code>n</code>qubitの量子状態を生成します。
生成した量子状態は \f$|0\rangle^{\otimes n}\f$ に初期化されています。
```cpp
#include <cppsim/state.hpp>

int main(){
	// 5-qubitの状態を生成
	unsigned int n = 5;
	QuantumState state(n);
	// |00000>に初期化
	state.set_zero_state();
	return 0;
}
```
メモリが不足している場合はプログラムが終了します。


### 量子状態の初期化
生成した量子状態は、計算基底に初期化したり、ランダムな状態に初期化することが出来ます。
```cpp
#include <cppsim/state.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();
	// |00101>に初期化
	state.set_computational_basis(0b00101);
	// ランダムな初期状態を生成
	state.set_Haar_random_state();
	// シードを指定してランダムな初期状態を生成
	state.set_Haar_random_state(0);
	return 0;
}
```


### 量子状態のデータのコピーとロード
量子状態を複製したり、他の量子状態のデータをロードできます。
```cpp
#include <cppsim/state.hpp>

int main(){
	unsigned int n = 5;
	QuantumState state(n);
	state.set_computational_basis(0b00101);

	// コピーして新たな量子状態を作成
	auto second_state = state.copy();

	// 量子状態を新たに生成し、既存の状態のベクトルをコピー
	QuantumState third_state(n);
	third_state.load(&state);
	return 0;
}
```


### 古典レジスタの操作
量子状態は古典レジスタを持っており、読み書きを行えます。
```cpp
#include <cppsim/state.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	// registerの書き込み
	int register_position = 3;
	int register_value = 1;
	state.set_classical_bit(register_position, register_value);

	// registerの読み出し
	int obtained_value;
	obtained_value = state.get_classical_bit(register_position);
	return 0;
}
```

### 量子状態に関する計算
量子状態を変えない計算として、以下の処理が可能です。
量子状態を変える計算は必ず量子ゲート、量子回路を介して行われます。
```cpp
#include <cppsim/state.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	// normの計算
	double norm = state.get_squared_norm();
	// Z基底で測定した時のentropyの計算
	double entropy = state.get_entropy();

	// index-th qubitをZ基底で測定して0を得る確率の計算
	unsigned int index = 3;
	double zero_prob = state.get_zero_probability(index);

	// 周辺確率を計算 (以下は0,3-th qubitが0、1,2-th qubitが1と測定される確率の例)
	std::vector<unsigned int> value_list = { 0,1,1,0,2 };
	double marginal_prob = state.get_marginal_probability(value_list);
	return 0;
}
```

### 量子状態の内積
<code>inner_product</code>関数で内積を計算できます。
```cpp
#include <cppsim/state.hpp>

int main(){
	unsigned int n = 5;
	QuantumState state_ket(n);
	state_ket.set_zero_state();

	QuantumState state_bra(n);
	state_bra.set_Haar_random_state();

	std::complex<double> value = state::inner_product(&state_ket, &state_bra);
	return 0;
}
```


### 量子状態のデータの取得
量子状態を表す \f$2^n\f$ の長さの配列を取得します。
特にGPUで量子状態を作成したり、大きい \f$n\f$ では非常に重い操作になるので注意してください。
```cpp
#include <cppsim/state.hpp>

int main(){
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	// GNU C++の場合、double _Complex配列を取得
	// MSVCの場合はstd::complex<double>の配列を取得
	const CTYPE* raw_data_c = state.data_c();

	// std::complex<double>の配列を取得
	const CPPCTYPE* raw_data_cpp = state.data_cpp();
}
```

量子状態を直接指定の配列にセットしたい場合などは、該当する量子ゲートを作成し、量子ゲートの作用として行うことを推奨します。


## 量子ゲート

### 量子ゲートの生成と作用
デフォルトで実装されている量子ゲートはgate_factoryの関数を通じて生成され、量子状態のポインタを引数として作用させられます。gate_factoryで生成した量子ゲートは自動では解放されないため、ユーザが解放しなければいけません。

```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	// Xゲートの作用
	unsigned int index = 3;
	auto x_gate = gate::X(index);
	x_gate->update_quantum_state(&state);

	// YでのPI/2回転
	double angle = M_PI / 2.0;
	auto ry_gate = gate::RY(index, angle);
	ry_gate->update_quantum_state(&state);

	delete x_gate;
	delete ry_gate;
	return 0;
}
```

<code>gate</code>名前空間で定義されているゲートは以下の通りです。
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
量子ゲートを合成し、新たな量子ゲートを生成できます。
合成したゲートは自身で解放しなければいけません。
```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	unsigned int index = 3;
	auto x_gate = gate::X(index);

	double angle = M_PI / 2.0;
	auto ry_gate = gate::RY(index, angle);

	// X, RYの順番に作用するゲートの作成
	auto x_and_ry_gate = gate::merge(x_gate, ry_gate);

	x_and_ry_gate->update_quantum_state(&state);

	delete x_gate;
	delete ry_gate;
	delete x_and_ry_gate;
	return 0;
}
```

### 量子ゲートのゲート行列の和
量子ゲートのゲート要素の和を取ることができます。
(control-qubitがある場合の和は現状動作が未定義なので利用しないでください。)
```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>

int main() {
	auto gate00 = gate::merge(gate::P0(0), gate::P0(1));
	auto gate11 = gate::merge(gate::P1(0), gate::P1(1));
	// |00><00| + |11><11|
	auto proj_00_or_11 = gate::add(gate00, gate11);
	std::cout << proj_00_or_11 << std::endl;

	auto gate_ii_zz = gate::add(gate::Identity(0), gate::merge(gate::Z(0), gate::Z(1)));
	auto gate_ii_xx = gate::add(gate::Identity(0), gate::merge(gate::X(0), gate::X(1)));
	auto proj_00_plus_11 = gate::merge(gate_ii_zz, gate_ii_xx);
	// ((|00>+|11>)(<00|+<11|))/2 = (II + ZZ)(II + XX)/4
	proj_00_plus_11->multiply_scalar(0.25);
	std::cout << proj_00_plus_11 << std::endl;
	return 0;
}
```

### 特殊な量子ゲートと一般の量子ゲート
cppsimにおける基本量子ゲートは以下の二つに分けられます。
- 特殊ゲート：そのゲートの作用について、専用の高速化がなされた関数があるもの。
- 一般ゲート：ゲート行列を保持し、行列をかけて作用するもの。

前者は後者に比べ専用の関数が作成されているため高速ですが、コントロール量子ビットを増やすなど、量子ゲートの作用を変更する操作が後から行えません。
こうした変更をしたい場合、特殊ゲートを一般ゲートに変換してやらねばなりません。

これは<code>gate::convert_to_matrix_gate</code>で実現できます。
以下がその例になります。

```cpp
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	unsigned int index = 3;
	auto x_gate = gate::X(index);

	// 1st-qubitが0の場合だけ作用するようにcontrol qubitを追加
	auto x_mat_gate = gate::to_matrix_gate(x_gate);
	unsigned int control_index = 1;
	unsigned int control_with_value = 0;
	x_mat_gate->add_control_qubit(control_index, control_with_value);

	x_mat_gate->update_quantum_state(&state);

	delete x_gate;
	delete x_mat_gate;
	return 0;
}
```

専用の量子ゲートの一覧についてはAPIドキュメントをご覧ください。


### 量子ゲートのゲート行列の取得
生成した量子ゲートのゲート行列を取得できます。control量子ビットなどはゲート行列に含まれません。特にゲート行列を持たない種類のゲート（例えばn-qubitのパウリ回転ゲート）などは取得に非常に大きなメモリと時間を要するので気を付けてください。
```cpp
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(){
	unsigned int index = 3;
	auto x_gate = gate::X(index);

	// 行列要素の取得
	// ComplexMatrixはEigen::MatrixXcdでRowMajorにした複素行列型
	ComplexMatrix matrix;
	x_gate->set_matrix(matrix);
	std::cout << matrix << std::endl;
	return 0;
}
```


### 量子ゲートの情報の取得

<code>ostream</code>に流し込むことで、量子ゲートのデバッグ情報を表示できます。量子ゲートのゲート行列が非常に巨大な場合、とても時間がかかるので注意してください。専用関数を持つ量子ゲートは自身のゲート行列は表示しません。

```cpp
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(){

	unsigned int index = 3;
	auto x_gate = gate::X(index);

	std::cout << x_gate << std::endl;

	delete x_gate;
	return 0;
}
```


### 一般的な量子ゲートの実現
cppsimでは量子情報における種々のマップを以下の形で実現します。

#### ユニタリ操作
量子ゲートとして実現します。

#### 射影演算子やクラウス演算子など
量子ゲートとして実現します。一般に作用後に量子状態のノルムは保存されません。<code>DenseMatrix</code>関数により生成できます。
```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_general.hpp>

int main() {
	ComplexMatrix one_qubit_matrix(2, 2);
	one_qubit_matrix << 0, 1, 1, 0;
	auto one_qubit_gate = gate::DenseMatrix(0, one_qubit_matrix);
	std::cout << one_qubit_gate << std::endl;

	ComplexMatrix two_qubit_matrix(4,4);
	two_qubit_matrix <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 1,
		0, 0, 1, 0;
	auto two_qubit_gate = gate::DenseMatrix({0,1}, two_qubit_matrix);
	std::cout << two_qubit_gate << std::endl;
	return 0;
}
```



#### 確率的なユニタリ操作
<code>Probabilistic</code>関数を用いて、複数のユニタリ操作と確率分布を与えて作成します。
```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_general.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	unsigned int index = 3;
	auto x_gate = gate::X(index);
	auto z_gate = gate::Z(index);

	auto probabilistic_xz_gate = gate::Probabilistic({ 0.1,0.2 } , { x_gate,z_gate });
	auto depolarizing_gate = gate::DepolarizingNoise(index, 0.3);

	depolarizing_gate->update_quantum_state(&state);
	probabilistic_xz_gate->update_quantum_state(&state);
	return 0;
}
```


#### CPTP-map
<code>CPTP</code>関数に完全性を満たすクラウス演算子のリストとして与えて作成します。
```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_general.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	unsigned int index = 3;
	auto p0 = gate::P0(index);
	auto p1_fix = gate::merge(gate::P1(index), gate::X(index));

	auto correction = gate::CPTP({p0,p1_fix});
	auto noise = gate::BitFlipNoise(index,0.1);

	noise->update_quantum_state(&state);
	correction->update_quantum_state(&state);
	return 0;
}
```


#### POVM
数値計算上にはInstrumentと同じなので、Instrumentとして実現します。

#### Instrument
Instrumentは一般のCPTP-mapの操作に加え、ランダムに作用したクラウス演算子の添え字を取得する操作です。例えば、Z基底での測定は<code>P0</code>と<code>P1</code>からなるCPTP-mapを作用し、どちらが作用したかを知ることに相当します。
cppsimでは<code>Instrument</code>関数にCPTP-mapの情報と、作用したクラウス演算子の添え字を書きこむ古典レジスタのアドレスを指定することで実現します。

```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_general.hpp>

int main() {
	auto gate00 = gate::merge(gate::P0(0), gate::P0(1));
	auto gate01 = gate::merge(gate::P0(0), gate::P1(1));
	auto gate10 = gate::merge(gate::P1(0), gate::P0(1));
	auto gate11 = gate::merge(gate::P1(0), gate::P1(1));

	std::vector<QuantumGateBase*> gate_list = { gate00, gate01, gate10, gate11 };
	unsigned int classical_pos = 0;
	auto gate = gate::Instrument(gate_list, classical_pos);

	QuantumState state(2);
	state.set_Haar_random_state();

	std::cout << state << std::endl;
	gate->update_quantum_state(&state);
	unsigned int result = state.get_classical_value(classical_pos);
	std::cout << state << std::endl;
	std::cout << result << std::endl;
	return 0;
}
```


#### Adaptive
古典レジスタに書き込まれた値に応じて操作を行ったり行わなかったりします。cppsimでは<code>[unsigned int]</code>型のレジスタを引数として受け取り、<code>bool</code>型を返す関数を指定し、これを実現します。

```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_general.hpp>

int main() {
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	unsigned int index = 3;
	auto h = gate::H(index);
	h->update_quantum_state(&state);

	auto meas = gate::Measurement(index,0);
	meas->update_quantum_state(&state);

	auto condition = [](const std::vector<UINT> reg){
		return reg[0]==1;
	};
	auto correction = gate::Adaptive(gate::X(index), condition);
	correction->update_quantum_state(&state);
	return 0;
}
```

#### CP-map
Kraus-rankが1の場合は、上記の単体のクラウス演算子として扱ってください。それ以外の場合は、TPになるようにクラウス演算子を調整した後、<code>multiply_scalar</code>関数で定数倍にした<code>Identity</code>オペレータを作用するなどして調整してください。



## 量子回路

### 量子回路の構成
量子回路は量子ゲートの集合として表されます。
例えば以下のように量子回路を構成できます。

```cpp
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/circuit.hpp>

int main(){
	unsigned int n = 5;
	QuantumState state(n);
	state.set_zero_state();

	// 量子回路を定義
	QuantumCircuit circuit(n);

	// 量子回路にゲートを追加
	for(int i=0;i<n;++i){
		circuit.add_H_gate(i);
	}

	// 自身で定義したゲートも追加できる
	for(int i=0;i<n;++i){
		circuit.add_gate(gate::H(i));
	}

	// 量子回路を状態に作用
	circuit.update_quantum_state(&state);
	return 0;
}
```

なお、<code>add_gate</code>で追加された量子回路は量子回路の解放時に一緒に解放されます。従って、代入したゲートは再利用できません。
引数として与えたゲートを再利用したい場合は、<code>add_gate_copy</code>関数を用いてください。ただしこの場合自身でゲートを解放する必要があります。

### 量子回路の最適化

量子ゲートをまとめて一つの量子ゲートとすることで、量子ゲートの数を減らすことができ、数値計算の時間を短縮できることがあります。（もちろん、対象となる量子ビットの数が増える場合や、専用関数を持つ量子ゲートを合成して専用関数を持たない量子ゲートにしてしまった場合は、トータルで計算時間が減少するかは状況に依ります。）

下記のコードでは<code>optimize</code>関数を用いて、量子回路の量子ゲートをターゲットとなる量子ビットが3つになるまで貪欲法で合成を繰り返します。

```cpp
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>

int main() {
	unsigned int n = 5;
	unsigned int depth = 10;
	QuantumCircuit circuit(n);
	for (int d = 0; d < depth; ++d) {
		for (int i = 0; i < n; ++i) {
			circuit.add_gate(gate::H(i));
		}
	}

	// 量子回路の最適化
	QuantumCircuitOptimizer opt;
	unsigned int max_block_size = 3;
	opt.optimize(&circuit, max_block_size);
	return 0;
}
```

### 量子回路の情報デバッグ
量子ゲートと同様、量子回路も<code>ostream</code>に流し込むことでデバッグ情報を表示することができます。

```cpp
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/circuit.hpp>

int main() {
	unsigned int n = 5;
	unsigned int depth = 10;
	QuantumCircuit circuit(n);
	for (int d = 0; d < depth; ++d) {
		for (int i = 0; i < n; ++i) {
			circuit.add_gate(gate::H(i));
		}
	}

	// 量子回路の情報を出力
	std::cout << circuit << std::endl;
	return 0;
}
```

## オブザーバブル

### オブザーバブルの生成
オブザーバブルはパウリ演算子の集合として表現されます。
パウリ演算子は下記のように定義できます。
```cpp
#include <cppsim/observable.hpp>
#include <string>

int main() {
	unsigned int n = 5;
	double coef = 2.0;
	std::string Pauli_string = "X 0 X 1 Y 2 Z 4";
	Observable observable(n);
	observable.add_operator(coef,Pauli_string.c_str());
	return 0;
}
```

### OpenFermionとの連携
また、OpenFermionを用いて生成された以下のようなフォーマットのファイルから, オブザーバブルを生成することができます。このとき、オブザーバブルはそれを構成するのに必要最小限の大きさとなります。例えば、以下のようなopenfermionを用いて得られたオブザーバブルを読み込み、オブザーバブルを生成することが可能です。
```python
from openfermion.ops import FermionOperator
from openfermion.transforms import bravyi_kitaev

h_00 = h_11 = -1.252477
h_22 = h_33 = -0.475934
h_0110 = h_1001 = 0.674493
h_2332 = h_3323 = 0.697397
h_0220 = h_0330 = h_1221 = h_1331 = h_2002 = h_3003 = h_2112 = h_3113 = 0.663472
h_0202 = h_1313 = h_2130 = h_2310 = h_0312 = h_0132 = 0.181287

fermion_operator = FermionOperator('0^ 0', h_00)
fermion_operator += FermionOperator('1^ 1', h_11)
fermion_operator += FermionOperator('2^ 2', h_22)
fermion_operator += FermionOperator('3^ 3', h_33)

fermion_operator += FermionOperator('0^ 1^ 1 0', h_0110)
fermion_operator += FermionOperator('2^ 3^ 3 2', h_2332)
fermion_operator += FermionOperator('0^ 3^ 3 0', h_0330)
fermion_operator += FermionOperator('1^ 2^ 2 1', h_1221)

fermion_operator += FermionOperator('0^ 2^ 2 0', h_0220-h_0202)
fermion_operator += FermionOperator('1^ 3^ 3 1', h_1331-h_1313)

fermion_operator += FermionOperator('0^ 1^ 3 2', h_0132)
fermion_operator += FermionOperator('2^ 3^ 1 0', h_0132)

fermion_operator += FermionOperator('0^ 3^ 1 2', h_0312)
fermion_operator += FermionOperator('2^ 1^ 3 0', h_0312)

## Bravyi-Kitaev transformation
bk_operator = bravyi_kitaev(fermion_operator)

## output
fp = open("H2.txt", 'w')
fp.write(str(bk_operator))
fp.close()
```
このとき、上のpythonコードで生成された<code>H2.txt</code>ファイルは以下のような形式になっています。
```txt
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
```
このような形式のファイルからオブザーバブルを生成するには、以下のように関数を通してオブザーバブルを生成することができます。

```cpp
#include <cppsim/observable.hpp>
#include <string>

int main() {
	unsigned int n = 5;
	std::string filename = "H2.txt";
	Observable* observable = observable::create_observable_from_openfermion_file(filename);
	delete observable;
	return 0;
}
```


### オブザーバブルの評価
状態に対してオブザーバブルの期待値を評価できます。
```cpp
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <string>

int main() {
	unsigned int n = 5;
	double coef = 2.0;
	std::string Pauli_string = "X 0 X 1 Y 2 Z 4";
	Observable observable(n);
	observable.add_operator(coef, Pauli_string.c_str());
	
	QuantumState state(n);
	observable.get_expectation_value(&state);
	return 0;
}
```

### オブザーバブルの回転
オブザーバブル\f$H\f$の回転\f$e^{i\theta H}\f$をTrotter展開によって行います。<code>num_repeats</code>はデフォルト値では以下のコードのようになっていますが、ユーザがオプションで指定することが可能です。
```cpp
#include <cppsim/circuit.hpp>
#include <cppsim/state.hpp>
#include <cppsim/observable.hpp>

int main() {
	UINT n;
	UINT num_repeats;
	double theta = 0.1;
	Observable* observable = observable::create_observable_from_openfermion_file("../test/cppsim/H2.txt");

	n = observable->get_qubit_count();
	QuantumState state(n);
	state.set_computational_basis(0);

	QuantumCircuit circuit(n);
	num_repeats = (UINT)std::ceil(theta * (double)n* 100.);
	circuit.add_observable_rotation_gate(*observable, theta, num_repeats);
	circuit.update_quantum_state(&state);

	auto result = observable->get_expectation_value(&state);
	std::cout << result << std::endl;
	delete observable;
	return 0;
}
```


## 変分量子回路
量子回路をParametricQuantumCircuitクラスとして定義すると、通所のQuantumCircuitクラスの関数に加え、変分法を用いて量子回路を最適化するのに便利ないくつかの関数を利用することができます。

### 変分量子回路の利用例

一つの回転角を持つ量子ゲート(X-rot, Y-rot, Z-rot, multi_qubit_pauli_rotation)はパラメトリックな量子ゲートとして量子回路に追加することができます。パラメトリックなゲートとして追加された量子ゲートについては、量子回路の構成後にパラメトリックなゲート数を取り出したり、後から回転角を変更することができます。

```cpp
#include <cppsim/state.hpp>
#include <vqcsim/parametric_circuit.hpp>
#include <cppsim/utility.hpp>

int main(){
	const UINT n = 3;
	const UINT depth = 10;

	// create n-qubit parametric circuit
	ParametricQuantumCircuit* circuit = new ParametricQuantumCircuit(n);
	Random random;
	for (UINT d = 0; d < depth; ++d) {
		// add parametric X,Y,Z gate with random initial rotation angle
		for (UINT i = 0; i < n; ++i) {
			circuit->add_parametric_RX_gate(i, random.uniform());
			circuit->add_parametric_RY_gate(i, random.uniform());
			circuit->add_parametric_RZ_gate(i, random.uniform());
		}
		// add neighboring two-qubit ZZ rotation
		for (UINT i = d % 2; i + 1 < n; i+=2) {
			circuit->add_parametric_multi_Pauli_rotation_gate({ i,i + 1 }, { 3,3 }, random.uniform());
		}
	}

	// get parameter count
	UINT param_count = circuit->get_parameter_count();

	// get current parameter, and set shifted parameter
	for (UINT p = 0; p < param_count; ++p) {
		double current_angle = circuit->get_parameter(p);
		circuit->set_parameter(p, current_angle + random.uniform());
	}

	// create quantum state and update
	QuantumState state(n);
	circuit->update_quantum_state(&state);

	// output state and circuit info
	std::cout << state << std::endl;
	std::cout << circuit << std::endl;

	// release quantum circuit
	delete circuit;
}
```