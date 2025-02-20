#pragma once
/**
 * プロジェクト独自の例外
 *
 * @file exception.hpp
 */

#include <stdexcept>

/**
 * \~japanese-en 未実装なものに対する例外
 */
class NotImplementedException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    NotImplementedException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en StateVectorとDensityMatrixが混ざっていて処理不能な例外
 */
class InoperatableQuantumStateTypeException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InoperatableQuantumStateTypeException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en QuantumStateCpuとQuantumStateGpuを同じ演算中で用いている例外
 */
class QuantumStateProcessorException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    QuantumStateProcessorException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en 一致すべき演算対象の量子ビット数が不一致である例外
 */
class InvalidQubitCountException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidQubitCountException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en 対象ビットのインデックスが重複している例外
 */
class DuplicatedQubitIndexException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    DuplicatedQubitIndexException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en 制御ビットのインデックスが不適切な例外
 */
class InvalidControlQubitException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidControlQubitException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en GeneralQuantumOperatorにPauliOperatorが1つも含まれていない例外
 */
class InvalidQuantumOperatorException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidQuantumOperatorException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en Observableがdiagonalでないなど条件を満たさない例外
 */
class InvalidObservableException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidObservableException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en
 * hermitianにしか使えない演算にhermitianでないOperatorやObservableを渡した例外
 */
class NonHermitianException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    NonHermitianException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en MatrixGateのサイズが対象の量子ビット数に合わない例外
 */
class InvalidMatrixGateSizeException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidMatrixGateSizeException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en std::vectorの状態のstateが量子ビット数に合わない例外
 */
class InvalidStateVectorSizeException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidStateVectorSizeException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en openfermionのファイルフォーマットが不適切であるという例外
 */
class InvalidOpenfermionFormatException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidOpenfermionFormatException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en 係数リストが不適切という例外
 */
class InvalidCoefListException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidCoefListException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en 確率分布が不適切という例外
 */
class InvalidProbabilityDistributionException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidProbabilityDistributionException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en ParametricGateのupdate_funcが定義されていないという例外
 */
class UndefinedUpdateFuncException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    UndefinedUpdateFuncException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en
 * ptreeのプロパティの値が想定外という例外
 */
class UnknownPTreePropertyValueException : public std::logic_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    UnknownPTreePropertyValueException(const std::string& message)
        : std::logic_error(message) {}
};

/**
 * \~japanese-en
 * GeneralQuantumOperator中のPauliOperatorのインデックスが範囲外という例外
 */
class OperatorIndexOutOfRangeException : public std::out_of_range {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    OperatorIndexOutOfRangeException(const std::string& message)
        : std::out_of_range(message) {}
};

/**
 * \~japanese-en 対象の量子ビットのインデックスが範囲外という例外
 */
class QubitIndexOutOfRangeException : public std::out_of_range {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    QubitIndexOutOfRangeException(const std::string& message)
        : std::out_of_range(message) {}
};

/**
 * \~japanese-en 行列中の対象の要素のインデックスが範囲外という例外
 */
class MatrixIndexOutOfRangeException : public std::out_of_range {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    MatrixIndexOutOfRangeException(const std::string& message)
        : std::out_of_range(message) {}
};

/**
 * \~japanese-en QuantumCircuit内のGateのインデックスが範囲外という例外
 */
class GateIndexOutOfRangeException : public std::out_of_range {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    GateIndexOutOfRangeException(const std::string& message)
        : std::out_of_range(message) {}
};

/**
 * \~japanese-en ParametricCircuitのパラメータのインデックスが範囲外という例外
 */
class ParameterIndexOutOfRangeException : public std::out_of_range {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    ParameterIndexOutOfRangeException(const std::string& message)
        : std::out_of_range(message) {}
};

/**
 * \~japanese-en パウリ演算子のIDや名前が不適切という例外
 */
class InvalidPauliIdentifierException : public std::domain_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidPauliIdentifierException(const std::string& message)
        : std::domain_error(message) {}
};

/**
 * \~japanese-en ノイズの種類の名前が不適切という例外
 */
class InvalidNoiseTypeIdentifierException : public std::domain_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidNoiseTypeIdentifierException(const std::string& message)
        : std::domain_error(message) {}
};

/**
 * \~japanese-en ファイルを開くのに失敗した例外
 */
class IOException : public std::runtime_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    IOException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * \~japanese-en MPIの実行時に失敗した例外
 */
class MPIRuntimeException : public std::runtime_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    MPIRuntimeException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * \~japanese-en mpi-size のエラー
 */
class MPISizeException : public std::runtime_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    MPISizeException(const std::string& message)
        : std::runtime_error(message) {}
};
