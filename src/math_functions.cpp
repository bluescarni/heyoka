// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/math_functions.hpp>

namespace heyoka
{

expression sin(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.dbl_name() = "llvm.sin";
    fc.ldbl_name() = "llvm.sin";
    fc.display_name() = "sin";
    fc.ty() = function::type::builtin;
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the sine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return cos(args[0]) * diff(args[0], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the sine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::sin(eval_dbl(args[0], map));
    };

    return expression{std::move(fc)};
}

expression cos(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.dbl_name() = "llvm.cos";
    fc.ldbl_name() = "llvm.cos";
    fc.display_name() = "cos";
    fc.ty() = function::type::builtin;
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument("Inconsistent number of arguments when taking the derivative of the cosine (1 "
                                        "argument was expected, but "
                                        + std::to_string(args.size()) + " arguments were provided");
        }

        return -sin(args[0]) * diff(args[0], s);
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the cosine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::cos(eval_dbl(args[0], map));
    };

    return expression{std::move(fc)};
}

expression log(expression e)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e));

    function fc{std::move(args)};
    fc.dbl_name() = "llvm.log";
    fc.ldbl_name() = "llvm.log";
    fc.display_name() = "log";
    fc.ty() = function::type::builtin;
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the logarithm (1 "
                "arguments were expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return expression{number(1.)} / args[0] * diff(args[0], s);
    };

    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the logarithm (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::log(eval_dbl(args[0], map));
    };
    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when evaluating the cosine (1 argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return std::cos(eval_dbl(args[0], map));
    };
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 1u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the logarithm (1 "
                "argument was expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }

        return expression{number(1.)} / args[0] * diff(args[0], s);
    };

    return expression{std::move(fc)};
}

 expression pow(expression e1, expression e2)
{
    std::vector<expression> args;
    args.emplace_back(std::move(e1));
    args.emplace_back(std::move(e2));

    function fc{std::move(args)};
    fc.dbl_name() = "llvm.pow";
    fc.ldbl_name() = "llvm.pow";
    fc.display_name() = "pow";
    fc.ty() = function::type::builtin;
    fc.diff_f() = [](const std::vector<expression> &args, const std::string &s) {
        if (args.size() != 2u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the exponentiation (2 "
                "arguments were expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        return args[1] * pow(args[0], args[1] - expression{number(1.)}) * diff(args[0], s)
               + pow(args[0], args[1]) * log(args[0]) * diff(args[1], s);
    };

    fc.eval_dbl_f() = [](const std::vector<expression> &args, const std::unordered_map<std::string, double> &map) {
        if (args.size() != 2u) {
            throw std::invalid_argument(
                "Inconsistent number of arguments when taking the derivative of the exponentiation (2 "
                "arguments were expected, but "
                + std::to_string(args.size()) + " arguments were provided");
        }
        return std::pow(eval_dbl(args[0], map), eval_dbl(args[1], map));
    };

    return expression{std::move(fc)};
}

} // namespace heyoka
