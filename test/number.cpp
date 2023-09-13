// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/math/constants/constants.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

#if defined(HEYOKA_HAVE_REAL128)

using namespace mppp::literals;

#endif

TEST_CASE("number basic")
{
    REQUIRE(number{} == number{0.});
    REQUIRE(std::holds_alternative<double>(number{}.value()));
    REQUIRE(std::holds_alternative<double>(number{1.1}.value()));
    REQUIRE(std::holds_alternative<float>(number{1.1f}.value()));
    REQUIRE(std::holds_alternative<long double>(number{1.1l}.value()));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(std::holds_alternative<mppp::real128>(number{1.1_rq}.value()));

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(std::holds_alternative<mppp::real>(number{mppp::real{1.1, 23}}.value()));
    REQUIRE(std::get<mppp::real>(number{mppp::real{1.1, 23}}.value()).get_prec() == 23);

#endif
}

TEST_CASE("number lt")
{
    REQUIRE(number{1.1} < number{2.});
    REQUIRE(!(number{1.1} < number{1.1}));
    REQUIRE(number{1.1} < number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(!(number{std::numeric_limits<double>::quiet_NaN()} < number{std::numeric_limits<double>::quiet_NaN()}));
    REQUIRE(!(number{std::numeric_limits<double>::quiet_NaN()} < number{1.1}));
    REQUIRE(number{3.1f} < number{2.});
    REQUIRE(!(number{1.1l} < number{2.}));
}

TEST_CASE("number hash eq")
{
    auto hash_number = [](const number &n) { return std::hash<number>{}(n); };

    REQUIRE(number{1.1} == number{1.1});
    REQUIRE(number{1.1} != number{1.2});

    REQUIRE(number{1.} != number{1.l});
    REQUIRE(number{1.l} != number{1.});
    REQUIRE(number{0.} != number{-0.l});
    REQUIRE(number{0.l} != number{-0.});
    REQUIRE(number{1.1} != number{1.2l});
    REQUIRE(number{1.2l} != number{1.1});
    REQUIRE(number{1.} != number{1.f});
    REQUIRE(number{1.f} != number{1.});
    REQUIRE(number{0.} != number{-0.f});
    REQUIRE(number{0.f} != number{-0.});
    REQUIRE(number{1.1} != number{1.2f});
    REQUIRE(number{1.2f} != number{1.1});

    REQUIRE(hash_number(number{1.1f}) == std::hash<float>{}(1.1f));
    REQUIRE(hash_number(number{1.1}) == std::hash<double>{}(1.1));
    REQUIRE(hash_number(number{1.1l}) == std::hash<long double>{}(1.1l));

    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} == number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<float>::quiet_NaN()} == number{std::numeric_limits<float>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<long double>::quiet_NaN()}
            == number{std::numeric_limits<long double>::quiet_NaN()});

    REQUIRE(hash_number(number{std::numeric_limits<double>::quiet_NaN()})
            != hash_number(number{std::numeric_limits<long double>::quiet_NaN()}));
    REQUIRE(hash_number(number{std::numeric_limits<double>::quiet_NaN()})
            != hash_number(number{std::numeric_limits<float>::quiet_NaN()}));

    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{0.l});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{0.f});
    REQUIRE(number{0.l} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{0.f} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-0.l});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-0.f});
    REQUIRE(number{-0.l} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{-0.f} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-1.23l});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-1.23f});
    REQUIRE(number{1.23l} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{1.23f} != number{std::numeric_limits<double>::quiet_NaN()});

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(number{1.} != number{1._rq});
    REQUIRE(number{1.f} != number{1._rq});
    REQUIRE(number{1._rq} != number{1.});
    REQUIRE(number{1._rq} != number{1.f});
    REQUIRE(number{0.} != number{-0._rq});
    REQUIRE(number{0.f} != number{-0._rq});
    REQUIRE(number{0._rq} != number{-0.});
    REQUIRE(number{0._rq} != number{-0.f});
    REQUIRE(number{1.1} != number{1.2_rq});
    REQUIRE(number{1.1f} != number{1.2_rq});
    REQUIRE(number{1.2_rq} != number{1.1});
    REQUIRE(number{1.2_rq} != number{1.1f});

    REQUIRE(number{1.1} != number{1.1_rq});
    REQUIRE(number{1.1f} != number{1.1_rq});
    REQUIRE(number{1.1_rq} != number{1.1});
    REQUIRE(number{1.1_rq} != number{1.1f});

    REQUIRE(hash_number(number{1.1_rq}) == std::hash<mppp::real128>{}(1.1_rq));

    REQUIRE(number{std::numeric_limits<mppp::real128>::quiet_NaN()}
            == number{std::numeric_limits<mppp::real128>::quiet_NaN()});

    REQUIRE(hash_number(number{std::numeric_limits<double>::quiet_NaN()})
            != hash_number(number{std::numeric_limits<mppp::real128>::quiet_NaN()}));

    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{0._rq});
    REQUIRE(number{std::numeric_limits<float>::quiet_NaN()} != number{0._rq});
    REQUIRE(number{0._rq} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{0._rq} != number{std::numeric_limits<float>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-0._rq});
    REQUIRE(number{std::numeric_limits<float>::quiet_NaN()} != number{-0._rq});
    REQUIRE(number{-0._rq} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{-0._rq} != number{std::numeric_limits<float>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-1.23_rq});
    REQUIRE(number{std::numeric_limits<float>::quiet_NaN()} != number{-1.23_rq});
    REQUIRE(number{1.23_rq} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{1.23_rq} != number{std::numeric_limits<float>::quiet_NaN()});

#endif

    auto [x, y] = make_vars("x", "y");

    {
        llvm_state s{kw::opt_level = 0u};

        auto dc = taylor_add_jet<double>(s, "jet", {prime(x) = (y + 1.) + (y + 1.l), prime(y) = x}, 1, 1, false, true);

        REQUIRE(dc.size() == 6u);
    }

    {
        llvm_state s{kw::opt_level = 0u};

        auto dc = taylor_add_jet<double>(s, "jet", {prime(x) = (y + 1.) + (y + expression{number{1.f}}), prime(y) = x},
                                         1, 1, false, true);

        REQUIRE(dc.size() == 6u);
    }
}

TEST_CASE("number s11n")
{
    std::stringstream ss;

    number n{4.5l};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0.};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{4.5l});

    ss.str("");

    n = number{1.2};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0.l};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{1.2});

    ss.str("");

    n = number{1.1f};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0.l};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{1.1f});

#if defined(HEYOKA_HAVE_REAL128)
    ss.str("");

    n = number{1.1_rq};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0.};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{1.1_rq});

    ss.str("");

    n = number{1.1};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0._rq};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{1.1});
#endif
}

TEST_CASE("number ostream")
{
    // Test with double, as the codepath is identical.
    std::string cmp;

    {
        std::ostringstream oss;
        oss.precision(std::numeric_limits<double>::max_digits10);
        oss.imbue(std::locale::classic());
        oss << std::showpoint;

        oss << 1.1;

        cmp = oss.str();
    }

    {
        std::ostringstream oss;
        oss << number(1.1);

        REQUIRE(oss.str() == cmp);
    }

#if defined(HEYOKA_HAVE_REAL)

    {
        std::ostringstream oss;
        oss << number{mppp::real{1.1, 23}};

        REQUIRE(oss.str() == std::get<mppp::real>(number{mppp::real{1.1, 23}}.value()).to_string());
    }

#endif
}

TEST_CASE("llvm_codegen")
{
    using std::isnan;

    // Pi double.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = builder.getDoubleTy();

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{boost::math::constants::pi<double>()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<double (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == boost::math::constants::pi<double>());
    }

    // Non-finite doubles.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = builder.getDoubleTy();

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{-std::numeric_limits<double>::infinity()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<double (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == -std::numeric_limits<double>::infinity());
    }

    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = builder.getDoubleTy();

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{-std::numeric_limits<double>::quiet_NaN()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<double (*)()>(s.jit_lookup("test"));

        REQUIRE(isnan(f_ptr()));
    }

#if !defined(HEYOKA_ARCH_PPC)

    // Pi long double.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = detail::to_llvm_type<long double>(context);

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{boost::math::constants::pi<long double>()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<long double (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == boost::math::constants::pi<long double>());
    }

    // Non-finite long doubles.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = detail::to_llvm_type<long double>(context);

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{-std::numeric_limits<long double>::infinity()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<long double (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == -std::numeric_limits<long double>::infinity());
    }

    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = detail::to_llvm_type<long double>(context);

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{-std::numeric_limits<long double>::quiet_NaN()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<long double (*)()>(s.jit_lookup("test"));

        REQUIRE(isnan(f_ptr()));
    }

#endif

#if defined(HEYOKA_HAVE_REAL128)

    // Pi real128.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = detail::to_llvm_type<mppp::real128>(context);

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{mppp::pi_128}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<mppp::real128 (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == mppp::pi_128);
    }

    // real128 non-finite.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = detail::to_llvm_type<mppp::real128>(context);

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{-std::numeric_limits<mppp::real128>::infinity()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<mppp::real128 (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == -std::numeric_limits<mppp::real128>::infinity());
    }

    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = detail::to_llvm_type<mppp::real128>(context);

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{-std::numeric_limits<mppp::real128>::quiet_NaN()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<mppp::real128 (*)()>(s.jit_lookup("test"));

        REQUIRE(isnan(f_ptr()));
    }

#endif

    // Small float test.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = builder.getFloatTy();

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{boost::math::constants::pi<float>()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<float (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == boost::math::constants::pi<float>());
    }

    // Mix float/double in the definition of the number vs codegen.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = builder.getFloatTy();

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{boost::math::constants::pi<double>()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<float (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == static_cast<float>(boost::math::constants::pi<double>()));
    }

    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = builder.getDoubleTy();

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{boost::math::constants::pi<float>()}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<double (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == boost::math::constants::pi<float>());
    }

#if defined(HEYOKA_HAVE_REAL)

    // Codegen high-precision pi real to double.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *fp_t = detail::to_llvm_type<double>(context);

        auto *ft = llvm::FunctionType::get(fp_t, {}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        builder.CreateRet(llvm_codegen(s, fp_t, number{mppp::real_pi(256)}));

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<double (*)()>(s.jit_lookup("test"));

        REQUIRE(f_ptr() == boost::math::constants::pi<double>());
    }

    // Codegen high-precision pi real to real256 and store in output variable.
    {
        llvm_state s{kw::opt_level = 0u};

        auto &md = s.module();
        auto &builder = s.builder();
        auto &context = s.context();

        auto *real_t = detail::to_llvm_type<mppp::real>(context);

        const auto real_pi_256 = mppp::real_pi(256);

        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), {llvm::PointerType::getUnqual(real_t)}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "test", &md);

        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        auto *real_val = llvm_codegen(s, detail::llvm_type_like(s, real_pi_256), number{real_pi_256});

        detail::ext_store_vector_to_memory(s, f->arg_begin(), real_val);

        builder.CreateRetVoid();

        s.verify_function(f);

        s.compile();

        auto f_ptr = reinterpret_cast<void (*)(mppp::real *)>(s.jit_lookup("test"));

        mppp::real out{0, 256};
        f_ptr(&out);

        REQUIRE(out == real_pi_256);
        REQUIRE(out.get_prec() == real_pi_256.get_prec());
    }

#endif
}

TEST_CASE("number_like")
{
    using Catch::Matchers::Message;

    llvm_state s;

    auto &builder = s.builder();

    auto num = detail::number_like(s, builder.getFloatTy(), 42);
    REQUIRE(num == number{42.f});
    REQUIRE(std::holds_alternative<float>(num.value()));

    num = detail::number_like(s, builder.getDoubleTy(), 42);
    REQUIRE(num == number{42.});
    REQUIRE(std::holds_alternative<double>(num.value()));

    if (std::numeric_limits<long double>::is_iec559 && std::numeric_limits<long double>::radix == 2) {
        if (std::numeric_limits<long double>::digits == 53) {
            // NOTE: here we are on Windows + MSVC, where long double == double
            // and thus C++ long double associates to LLVM double.
            // In number_like(), we check tp == to_llvm_type<double> *before*
            // tp == to_llvm_type<long double>, so we get a number containing
            // double rather than long double.
            num = detail::number_like(s, llvm::Type::getDoubleTy(s.context()), 42);
            REQUIRE(num == number{42.});
            REQUIRE(std::holds_alternative<double>(num.value()));
        } else if (std::numeric_limits<long double>::digits == 64) {
            num = detail::number_like(s, llvm::Type::getX86_FP80Ty(s.context()), 42);
            REQUIRE(num == number{42.l});
            REQUIRE(std::holds_alternative<long double>(num.value()));
        } else if (std::numeric_limits<long double>::digits == 113) {
            num = detail::number_like(s, llvm::Type::getFP128Ty(s.context()), 42);
            REQUIRE(num == number{42.l});
            REQUIRE(std::holds_alternative<long double>(num.value()));
        }
    }

#if defined(HEYOKA_HAVE_REAL128)

    num = detail::number_like(s, llvm::Type::getFP128Ty(s.context()), 42);
    REQUIRE(num == number{42_rq});
    REQUIRE(std::holds_alternative<mppp::real128>(num.value()));

#endif

    REQUIRE_THROWS_MATCHES(detail::number_like(s, llvm::Type::getVoidTy(s.context()), 42), std::invalid_argument,
                           Message("Unable to create a number of type 'void' from the input value 42"));
}

TEST_CASE("exp")
{
    REQUIRE(exp(number{1.f}) == number{std::exp(1.f)});
    REQUIRE(exp(number{1.}) == number{std::exp(1.)});

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(exp(number{1._rq}) == number{exp(1._rq)});

#endif
}

TEST_CASE("binomial")
{
    using Catch::Matchers::Message;

    auto n = binomial(number(4.f), number(2.f));
    REQUIRE(n == number(6.f));
    REQUIRE(std::holds_alternative<float>(n.value()));

    n = binomial(number(4.), number(2.));
    REQUIRE(n == number(6.));
    REQUIRE(std::holds_alternative<double>(n.value()));

#if !defined(HEYOKA_ARCH_PPC)

    n = binomial(number(4.l), number(2.l));
    REQUIRE(n == number(6.l));
    REQUIRE(std::holds_alternative<long double>(n.value()));

#endif

#if defined(HEYOKA_HAVE_REAL128)

    n = binomial(number(4._rq), number(2._rq));
    REQUIRE(n == number(6._rq));
    REQUIRE(std::holds_alternative<mppp::real128>(n.value()));

#endif

#if defined(HEYOKA_HAVE_REAL)

    using namespace mppp::literals;

    n = binomial(number(4._r128), number(2._r128));
    REQUIRE(n == number(6._r128));
    REQUIRE(std::holds_alternative<mppp::real>(n.value()));
    REQUIRE(std::get<mppp::real>(n.value()).get_prec() == 128);

    n = binomial(number(4._r128), number(2._r256));
    REQUIRE(n == number(6._r256));
    REQUIRE(std::holds_alternative<mppp::real>(n.value()));
    REQUIRE(std::get<mppp::real>(n.value()).get_prec() == 256);

    n = binomial(number(4._r256), number(2._r128));
    REQUIRE(n == number(6._r256));
    REQUIRE(std::holds_alternative<mppp::real>(n.value()));
    REQUIRE(std::get<mppp::real>(n.value()).get_prec() == 256);

#endif

    REQUIRE_THROWS_MATCHES(binomial(number(4.), number(2.f)), std::invalid_argument,
                           Message("Cannot compute the binomial coefficient of two numbers of different type"));

    REQUIRE_THROWS_MATCHES(binomial(number(4.), number(std::numeric_limits<double>::infinity())), std::invalid_argument,
                           Message("Cannot compute the binomial coefficient of non-finite values"));

    REQUIRE_THROWS_MATCHES(binomial(number(4.), number(3.1)), std::invalid_argument,
                           Message("Cannot compute the binomial coefficient non-integral values"));
}

TEST_CASE("nextafter")
{
    using Catch::Matchers::Message;

    using std::nextafter;

    REQUIRE(nextafter(number(1.f), number(0.f)) == number(nextafter(1.f, 0.f)));
    REQUIRE(nextafter(number(1.), number(0.)) == number(nextafter(1., 0.)));
    REQUIRE(nextafter(number(1.l), number(0.l)) == number(nextafter(1.l, 0.l)));

#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE(nextafter(number(1._rq), number(0._rq)) == number(nextafter(1._rq, 0._rq)));
#endif

    REQUIRE_THROWS_MATCHES(nextafter(number(4.), number(2.f)), std::invalid_argument,
                           Message("Cannot invoke nextafter() on two numbers of different type"));
}

TEST_CASE("is_zero")
{
    REQUIRE(is_zero(number{0.}));
    REQUIRE(!is_zero(number{1.}));

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(is_zero(number{mppp::real{0.}}));
    REQUIRE(!is_zero(number{mppp::real{1.}}));

#endif
}

TEST_CASE("is_one")
{
    REQUIRE(!is_one(number{0.}));
    REQUIRE(is_one(number{1.}));

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(!is_one(number{mppp::real{0.}}));
    REQUIRE(is_one(number{mppp::real{1.}}));

#endif
}

TEST_CASE("is_negative_one")
{
    REQUIRE(!is_negative_one(number{0.}));
    REQUIRE(is_negative_one(number{-1.}));

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(!is_negative_one(number{mppp::real{0.}}));
    REQUIRE(is_negative_one(-number{mppp::real{1.}}));

#endif
}

TEST_CASE("move semantics")
{
    REQUIRE(std::is_nothrow_move_assignable_v<number>);
    REQUIRE(std::is_nothrow_move_constructible_v<number>);

    auto x = number{3.};

    // Check that move construction sets the moved-from
    // object to zero.
    auto x2(std::move(x));
    REQUIRE(x2 == number{3.});
    REQUIRE(x == number{0.});

    // Check that move assignment sets the moved-from
    // object to zero.
    auto x3 = number{1.};
    x3 = std::move(x2);
    REQUIRE(x3 == number{3.});
    REQUIRE(x2 == number{0.});
}
