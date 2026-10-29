#pragma once

#include <stdexcept>
#include <string>

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error(message);
}
