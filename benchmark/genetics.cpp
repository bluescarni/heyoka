#include <chrono>
#include <iostream>
#include <random>

#include <heyoka/detail/splitmix64.hpp>
#include <heyoka/gp.hpp>

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
        auto ex = generator(2u, 4u);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of expressions generated per second: " << N / static_cast<double>(duration.count()) << "M\n";

    // 2 - We time the node counter
    auto ex = generator(2u, 4u);
    start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        auto n = count_nodes(ex);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of calls to node counter per second: " << N / static_cast<double>(duration.count()) << "M\n";

    // 3 - We time the number of mutations we can do
    start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        mutate(ex, generator, 0.1, engine);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of mutations per second: " << N / static_cast<double>(duration.count()) << "M\n";

    // 4 - We time the number of crossovers we can do
    auto ex2 = generator(2u, 4u);
    auto ex3 = generator(2u, 4u);
    start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        crossover(ex2, ex3, engine);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of crossovers per second: " << N / static_cast<double>(duration.count()) << "M\n";
}