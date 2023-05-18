import time

import matplotlib.pyplot as plt

from qulacs import Observable, PauliOperator
from qulacs.observable._new_get_matrix import _new_get_matrix
from qulacs.observable._old_get_matrix import _old_get_matrix

OMP_NUM_THREADS = 4

# get_matrix()のベンチマークテストを行う
# 量子ビット数を変えて、get_matrix()の実行時間を計測する

# 量子ビット数をnに入れる
# nは2から始めて、1ずつ増やしていく
# 量子ビット数がnのときのget_matrix()の実行時間を計測する
# 横軸が量子ビット数、縦軸がget_matrix()の実行時間のグラフを描く

x = []
old_y = []
new_y = []

fig, ax = plt.subplots()

for n in range(2, 5):
    print("n =", n)
    # get_matrix()の実行時間を計測\
    repeat = 1
    sum1 = 0
    sum2 = 0
    for i in range(repeat):
        # RandomなObservableを生成
        observable = Observable(n)
        coef = 1.0
        s = "X 1 Y 0"
        pauli = PauliOperator(s, coef)
        observable.add_operator(pauli)

        start1 = time.time()
        print(_new_get_matrix(observable).toarray(), "\n")
        end1 = time.time()
        sum1 += end1 - start1

        start2 = time.time()
        print(_old_get_matrix(observable).toarray(), "\n")
        end2 = time.time()
        sum2 += end2 - start2
        # print("  py", i, ": {:.4f}".format(end1 - start1), ", cpp", i, ": {:.4f}".format(end2 - start2))

    avg1, avg2 = sum1 / repeat, sum2 / repeat
    print(avg1, ",", avg2)
    # get_matrix()の実行時間をyに入れる
    old_y.append(avg1)
    new_y.append(avg2)
    # 量子ビット数をxに入れる
    x.append(n)

# グラフを保存
c1, c2 = "blue", "green"
l1, l2 = "cpp_get_matrix", "py_get_matrix"
ax.set_title("get_matrix benchmark")
ax.grid()
ax.plot(x, old_y, color=c1, label=l1)
ax.plot(x, new_y, color=c2, label=l2)
fig.legend(loc="upper left")
fig.tight_layout()
plt.show()
# ~/fujiikenn/qulacs_dir/get_matrixディレクトリに保存する
fig.savefig("get_matrix_benchmark_cpp.png")
