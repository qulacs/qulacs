

# トラブルシューティング

## C/C++のライブラリに関するエラー

### csim_sharedのリンク時にエラーが発生する
コンパイラがgcc/g++になっていません。cmakeコマンドを行う際にver7以降のgcc/g++を指定してください。Macの標準のgccはバックエンドがclangになっているので注意してください。

### mingwでpycppsimをコンパイルする時に、Posix系のヘッダファイル(crypt.h, sys/select.hなど)が見つからないというエラーが出る。リンク時に-lintlが見つからないというエラーが出る。

gcc/g++とpythonで32bit/64bitが異なると生じることを確認しています。


## pythonライブラリに関するエラー

### 複数のpythonを抱えており、特定のバージョンに対してビルドを行いたい

cmakeの際に-D PYTHON_EXECUTABLE:FILEPATH=/usr/bin/pythonx.xなどpythonのバイナリのパスを直接指定することでできます。しかし、原則的にpyenvやcondaを用いて標準のpythonを変えてからビルドすることをお勧めします。

## import時に「init関数がない」旨のエラーが出る。
生成した.dll,.pydは名前とモジュール名が一致している必要があるため、ライブラリ名を変更してimportを試みるとこのエラーが出ます。

そうでない場合は、ビルドしたpythonとそれをimportしているpythonのバージョンが異なる可能性が高いです。

### import時にSegmentation faultが生じる。何も出力せずにpythonが終了する。Py_...という関数が見つからないと言われる。importしようとしてもファイルが存在しないと言われる。など。

ビルドしたpythonとそれをimportしているpythonのバージョンが異なる可能性が高いです。

