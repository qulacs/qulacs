#pragma once
/**
 * プロジェクト独自の例外
 *
 * @file exception.hpp
 */

#include <stdexcept>

/**
 * \~japanese-en 量子ビットの数が不整合である例外
 */
class InvalidQubitCountException : public std::runtime_error {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param message エラーメッセージ
     */
    InvalidQubitCountException(const std::string& message)
        : std::runtime_error(message) {}
};
