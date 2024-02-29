"""回路のグラフ化をできます。量子状態の棒グラフ、縮約した後の玉表示"""

import matplotlib.pyplot as plt
import numpy as np

from qulacs import Observable, QuantumState

# bar(x, y, color=["red", "blue", "green", "pink", "orange"], width=0.5)


def cmp_to_color(z):
    """複素数を受け取って、色を返す"""
    arg_rad = np.angle(z)

    color = (
        np.cos(arg_rad) * 0.5 + 0.5,
        np.cos(arg_rad + np.pi * 0.667) * 0.5 + 0.5,
        np.cos(arg_rad + np.pi * 1.333) * 0.5 + 0.5,
    )
    return color


def show_amplitude(state):
    """純粋状態量子状態を受け取って、棒グラフを表示する"""
    n_qubit = state.get_qubit_count()
    aaa = state.get_vector()
    bits = []
    ys = []
    colors = []
    for i in range(2**n_qubit):
        print(aaa[i])
        moziretu = "{:b}".format(i)
        while len(moziretu) < n_qubit:
            moziretu = "0" + moziretu
        bits.append(moziretu)

        ys.append(abs(aaa[i]))
        colors.append(cmp_to_color(aaa[i]))
    plt.bar(bits, ys, color=colors)
    plt.show()


def show_blochsphere(state, bit):
    """ブロッホ球の表示をします"""
    n_qubit = state.get_qubit_count()
    observableX = Observable(n_qubit)
    observableX.add_operator(1.0, f"X {bit}")  # オブザーバブルを設定
    observableY = Observable(n_qubit)
    observableY.add_operator(1.0, f"Y {bit}")  # オブザーバブルを設定
    observableZ = Observable(n_qubit)
    observableZ.add_operator(1.0, f"Z {bit}")  # オブザーバブルを設定

    X = observableX.get_expectation_value(state)
    Y = observableY.get_expectation_value(state)
    Z = observableZ.get_expectation_value(state)
    print(X, Y, Z)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    # sphere
    u, v = np.mgrid[0 : 2 * np.pi : 8j, 0 : np.pi : 8j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="lightskyblue", linewidth=0.5)
    ax.quiver(0, 0, 0, X, Y, Z, color="red")
    ax.scatter(X, Y, Z, color="red")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


def show_probability(state):
    """量子状態を受け取って、　出現確率の棒グラフを表示します。"""
    n_qubit = state.get_qubit_count()
    bits = []
    ys = np.zeros(2**n_qubit)
    for i in range(2**n_qubit):
        moziretu = "{:b}".format(i)
        while len(moziretu) < n_qubit:
            moziretu = "0" + moziretu
        bits.append(moziretu)

    if isinstance(state, QuantumState):
        aaa = state.get_vector()
        for i in range(2**n_qubit):
            ys[i] += abs(aaa[i]) ** 2
    else:
        aaa = state.get_matrix()
        for h in range(len(aaa)):
            for i in range(2**n_qubit):
                ys[i] += abs(aaa[h][i]) ** 2
    plt.bar(bits, ys)
    plt.show()
