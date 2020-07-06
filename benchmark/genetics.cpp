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
    // Here we define the type of expression (two variables, default choices for the operators)
    expression_generator generator({"x", "y"}, engine());
    // 0 - We generate N expressions and count the nodes
    std::vector<expression> exs_original;
    std::vector<std::size_t> n_nodes;
    for (auto i = 0u; i < N; ++i) {
        exs_original.push_back(generator(2u, 4u));
        n_nodes.push_back(count_nodes(exs_original.back()));
    }
    // 1 - We time the number of random expressions we may create
    auto start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        auto ex = generator(2u, 4u);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of expressions generated per second: " << N / static_cast<double>(duration.count()) << "M\n";

    // 2 - We time the node counter
    start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        auto n = count_nodes(exs_original[i]);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of calls to node counter per second: " << N / static_cast<double>(duration.count()) << "M\n";

    // 3 - We time the number of mutations we can do
    auto exs = exs_original;
    start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        mutate(exs[i], generator, 0.1, engine);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of mutations per second: " << N / static_cast<double>(duration.count()) << "M\n";

    // 4 - We time the number of crossovers we can do (including counting the nodes)
    exs = exs_original;
    start = high_resolution_clock::now();
    for (auto i = 0u; i < (N - 1); ++i) {
        crossover(exs[i], exs[i + 1], engine);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of crossovers per second: " << N / static_cast<double>(duration.count()) << "M\n";

    // 5 - We time the number of crossovers we can do (excluding counting the nodes)
    exs = exs_original;
    start = high_resolution_clock::now();
    for (auto i = 0u; i < N / 2; ++i) {
        std::size_t n2 = std::uniform_int_distribution<std::size_t>(0, n_nodes[2 * i] - 1u)(engine);
        std::size_t n3 = std::uniform_int_distribution<std::size_t>(0, n_nodes[2 * i + 1] - 1u)(engine);
        crossover(exs[2 * i], exs[2 * i + 1], n2, n3, engine);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Millions of crossovers per second (no node count): " << N / static_cast<double>(duration.count()) / 2
              << "M\n";
}