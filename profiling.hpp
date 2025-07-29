#pragma once
#include <chrono>
#include <utility>
#include <optional>
#include <sstream>
#include <iomanip>
namespace profiling
{
    using std::chrono::duration;
    using std::chrono::high_resolution_clock;

    struct StopWatch
    {
        std::optional<high_resolution_clock::time_point> start_point{};
        double elapsed_total{};

        inline void startnew()
        {
            elapsed_total = 0.0;
            start();
        }

        inline void start()
        {
            if (!start_point)
                start_point = high_resolution_clock::now();
        }

        inline bool is_running() const { return start_point.has_value(); }

        inline double elapsed() const
        {
            return elapsed_total + ((start_point) ? duration<double>(high_resolution_clock::now() - *start_point).count() : 0);
        }
        void stop()
        {
            if (start_point)
            {
                elapsed_total += duration<double>(high_resolution_clock::now() - *start_point).count();
                start_point.reset();
            }
        }
        std::string elapsed_str() const
        {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(9) << elapsed();
            return std::string(ss.str());
        };
    };
}
