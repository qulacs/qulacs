# C++ Quantum Circuit Simulator for research

C++ Quantum Circuit Simulator for researchは量子回路を用いた研究を補助するためのC++の数値計算ライブラリです。
このライブラリはCSIM, CPPSIM, VQCSIMの三つのパッケージからなります。
VQCSIMはCPPSIMに、CPPSIMはCSIMに依存するライブラリです。

## CSIM (Core Simulator)
低級言語で記述された、ゲートの作用による量子状態の更新や、量子状態の性質の計算を高速に行うためのライブラリです。
今後、GPUを用いた高速化など様々なデバイスに最適化していく予定です。

## CPPSIM (C++ Simulator)
C++言語で記述された、CSIMをより扱いやすくするためのツール群です。
Eigenに依存してます。
今後、Openfermionとの接続やQASMの読み込みなどを実装していく予定です。

## VQCSIM (Variational Quantum Circuit Simulator)
CPPSIMでVariationalな量子計算のアプローチをシミュレートするための拡張です。現在、Variational Quantum Eigensolverなどの代表的なアルゴリズムが実装されています。今後、種類を増やしていく予定です。


