#pragma once

// Farbige Konsole. spdlog 1.17 liegt header-only unter third_party/spdlog (MIT).
#define SPDLOG_HEADER_ONLY
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

inline void setup_log() {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("kugel", std::move(sink));
    logger->set_pattern("%H:%M:%S %^[%l] %v%$");
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::info);
    spdlog::set_default_logger(std::move(logger));
}
