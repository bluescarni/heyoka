#include <chrono>
#include <iostream>
#include <random>

#include <heyoka/gp.hpp>
#include <heyoka/detail/splitmix64.hpp>


using namespace heyoka;

using namespace std::chrono;
int main()
{
    unsigned N = 10000;
    std::random_device rd;
    detail::random_engine_type engine(rd());
    // 1 - We time the number of random expressions we may create
    expression_generator generator({"x", "y"}, engine());
    auto start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        auto ex = generator(2u, 6u);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of expressions generated: " << 1. / (static_cast<double>(duration.count()) / N) << "M\n";
}