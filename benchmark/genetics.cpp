#include <chrono>
#include <iostream>
#include <random>

#include <heyoka/detail/splitmix64.hpp>
#include <heyoka/gp.hpp>

using namespace heyoka;
using namespace std::chrono;

static std::mt19937 gen(12345u);
std::vector<std::vector<double>> random_args_vv(unsigned N, unsigned n)
{
    std::uniform_real_distribution<double> rngm11(-1, 1.);
    std::vector<std::vector<double>> retval(N, std::vector<double>(n, 0u));
    for (auto &vec : retval) {
        for (auto &el : vec) {
            el = rngm11(gen);
        }
    }
    return retval;
}

std::unordered_map<std::string, std::vector<double>> vv_to_dv(const std::vector<std::vector<double>> &in)
{
    std::unordered_map<std::string, std::vector<double>> retval;
    std::vector<double> x_vec(in.size()), y_vec(in.size());
    for (decltype(in.size()) i = 0u; i < in.size(); ++i) {
        x_vec[i] = in[i][0];
        y_vec[i] = in[i][1];
    }
    retval["x"] = x_vec;
    retval["y"] = y_vec;
    return retval;
}

int main()
{
    unsigned N = 10000;
    std::random_device rd;
    detail::splitmix64 engine(rd());
    // Here we define the type of expression (two variables, default choices for the operators)
    expression_generator generator({"x", "y"}, engine);
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
        mutate(exs[i], generator, 0.1, engine, 2, 5, 0);
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

    // 6 - We time the single step of a genetic algorithm, evaluation, crossover over data with 200 points of
    // dimension 3.
    auto data = random_args_vv(200u, 3u);
    auto data_batch = vv_to_dv(data);
    auto out = std::vector<double>(200u, 0.123);

    start = high_resolution_clock::now();
    for (auto i = 0u; i < N; ++i) {
        // We need to evaluate the output of the expression
        eval_batch_dbl(out, exs[i], data_batch);
        // And to make some crossover
        crossover(exs[i], exs[std::uniform_int_distribution<size_t>(0, 9999)(engine)], engine);
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Models tried per second - evaluation (200 points) and crossover: "
              << (N / static_cast<double>(duration.count())) * 1000000 << "\n";
}
