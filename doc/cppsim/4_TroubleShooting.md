

# トラブルシューティング

## C/C++のライブラリに関するエラー

### csim_sharedのリンク時にエラーが発生する
コンパイラがgcc/g++になっていません。cmakeコマンドを行う際にver7以降のgcc/g++を指定してください。

### cmakeコマンドが終わらない、CMAKE_CXX_COMPILERがフルパスではないというエラーが出る
CMakeLists.txtでset関数でCMAKE_CXX_COMPILERを設定し、かつ./vqcsim/python/CMakeLists.txtでpybind11がadd_subdirectoryされている場合に生じます。
CMAKE_CXX_COMPILERをcmakeコマンドの引数として指定することで回避できます。

### mingwでpycppsimをコンパイルする時に、Posix系のヘッダファイル(crypt.h, sys/select.hなど)が見つからないというエラーが出る。リンク時に-lintlが見つからないというエラーが出る。

gcc/g++とpythonで32bit/64bitが異なると生じます。


## pythonライブラリに関するエラー

### 複数のpython3を抱えており、特定のバージョンに対してビルドを行いたい

cmakeの際に-D PYTHON_EXECUTABLE:FILEPATH=/usr/bin/pythonx.xなどpythonのバイナリのパスを直接指定することでできるかもしれません。

### 以下のいずれかのエラーが出る
- cmake時にCould NOT find PythonLibsというエラーが出る
- make python時に<Python.h>が見つかりませんというエラーが出る
- make pythonでのpycppsimのリンク時に"Py_hoge"という名前のシンボルが見つからないというエラーが大量に出る

pythonのライブラリの個所が正しく特定できていません。いくつかの環境で、pythonがインストールされているにもかかわらずこのエラーが出ることを確認しています。

以下の手順で正しくビルドできることがあります。

1.

```sh
python3-config --includes
python3-config --libs
python3-config --ldflags
```
の三つのコマンドが実行可能で、一行ずつ出力があることを確認してください。python3-configが存在しない場合、python3-devがインストールされているか確認してください。

2.

./vqcsim/pythonのCMakeList.txtをCMakeLists.oldにリネームし、CMakeLists.subをCMakeLists.txtにリネームします。
次にエディタでCMakeLists.txtを開き、
```
<paste output of python3-config --includes>
<paste output of python3-config --libs>
<paste output of python3-config --ldflags>
```
の３か所を、それぞれ1.のコマンドの出力の一行に差し替えます。

3.

この状態で再度cmakeを実行してください。


## import時に「init関数がない」旨のエラーが出る。
生成した.dll,.pydは名前とモジュール名が一致している必要があるため、ライブラリ名を変更してimportを試みるとこのエラーが出ます。

そうでない場合は、ビルドしたpythonとそれをimportしているpythonのバージョンが異なる可能性が高いです。pycppsimはpython3でビルドされているため、python3からimportしてください。

### import時にSegmentation faultが生じる。あるいは何も出力せずにpythonが終了する

ビルドしたpythonとそれをimportしているpythonのバージョンが異なる可能性が高いです。pycppsimはpython3でビルドされているため、python3からimportしてください。

そうでない場合、LTO(Link Time Optimization)が悪さをしている可能性があります。一部の環境では-fltoオプションによる最適化を行うとライブラリのロード時にエラーが発生して終了するようです。主にMinGWでその現象が確認されていますが、-fltoオプションを外すことで問題が解決するかもしれません。

上記でも改善しない場合は「cmake時にCould NOT found PythonLibsというエラーが出る」の対策を試してください。

