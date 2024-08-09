// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/serialization/array.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/task_arena.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/variant_s11n.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Utilities to create and destroy tape arrays for compiled functions
// in compact mode. These may have custom alignment requirements due
// to the use of SIMD instructions, hence we need to use aligned new/delete
// and a custom deleter for the unique ptr.
struct aligned_array_deleter {
    std::align_val_t al{};
    void operator()(void *ptr) const noexcept
    {
        // NOTE: here we are using directly the delete operator (which does not invoke destructors),
        // rather than a delete expression (which would also invoke destructors). However, because
        // ptr points to a bytes array, we do not need to explicitly call the destructor here, deallocation will be
        // sufficient.
        ::operator delete[](ptr, al);
    }
};

using aligned_array_t = std::unique_ptr<std::byte[], aligned_array_deleter>;

aligned_array_t make_aligned_array(std::size_t sz, std::size_t al)
{
    assert(al > 0u);
    assert((al & (al - 1u)) == 0u);

    if (sz == 0u) {
        return {};
    } else {
#if defined(_MSC_VER)
        // MSVC workaround for this issue:
        // https://developercommunity.visualstudio.com/t/using-c17-new-stdalign-val-tn-syntax-results-in-er/528320

        // Allocate the raw memory.
        auto *buf = ::operator new[](sz, std::align_val_t{al});

        // Formally construct the bytes array.
        auto *ptr = ::new (buf) std::byte[sz];

        // Constrcut and return the unique ptr.
        return aligned_array_t{ptr, {.al = std::align_val_t{al}}};
#else
        return aligned_array_t{::new (std::align_val_t{al}) std::byte[sz], {.al = std::align_val_t{al}}};
#endif
    }
}

} // namespace

} // namespace detail

template <typename T>
struct cfunc<T>::impl {
    // The compiled function types.
    // Non-compact mode.
    using cfunc_ptr_t = void (*)(T *, const T *, const T *, const T *) noexcept;
    using cfunc_ptr_s_t = void (*)(T *, const T *, const T *, const T *, std::size_t) noexcept;
    // Compact-mode. These have an additional argument - the tape pointer.
    using c_cfunc_ptr_t = void (*)(T *, const T *, const T *, const T *, void *) noexcept;
    using c_cfunc_ptr_s_t = void (*)(T *, const T *, const T *, const T *, void *, std::size_t) noexcept;

    // Thread-local storage for parallel operations.
    using ets_item_t = detail::aligned_array_t;
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_item_t, oneapi::tbb::cache_aligned_allocator<ets_item_t>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;

    // Data members.
    std::vector<expression> m_fn;
    std::vector<expression> m_vars;
    std::variant<std::array<llvm_state, 3>, llvm_multi_state> m_states;
    std::uint32_t m_batch_size = 0;
    std::vector<expression> m_dc;
    std::vector<std::array<std::size_t, 2>> m_tape_sa;
    std::vector<detail::aligned_array_t> m_tapes;
    std::variant<cfunc_ptr_t, c_cfunc_ptr_t> m_fptr_scal;
    std::variant<cfunc_ptr_s_t, c_cfunc_ptr_s_t> m_fptr_scal_s;
    std::variant<cfunc_ptr_s_t, c_cfunc_ptr_s_t> m_fptr_batch_s;
    std::uint32_t m_nparams = 0;
    bool m_is_time_dependent = false;
    std::uint32_t m_nouts = 0;
    std::uint32_t m_nvars = 0;
    long long m_prec = 0;
    bool m_check_prec = false;
    bool m_high_accuracy = false;
    bool m_compact_mode = false;
    bool m_parallel_mode = false;

    // Serialization.
    void save(boost::archive::binary_oarchive &ar, unsigned) const
    {
        ar << m_fn;
        ar << m_vars;
        ar << m_states;
        ar << m_batch_size;
        ar << m_dc;
        ar << m_tape_sa;
        ar << m_nparams;
        ar << m_is_time_dependent;
        ar << m_nouts;
        ar << m_nvars;
        ar << m_prec;
        ar << m_check_prec;
        ar << m_high_accuracy;
        ar << m_compact_mode;
        ar << m_parallel_mode;
    }
    void load(boost::archive::binary_iarchive &ar, unsigned)
    {
        ar >> m_fn;
        ar >> m_vars;
        ar >> m_states;
        ar >> m_batch_size;
        ar >> m_dc;
        ar >> m_tape_sa;
        ar >> m_nparams;
        ar >> m_is_time_dependent;
        ar >> m_nouts;
        ar >> m_nvars;
        ar >> m_prec;
        ar >> m_check_prec;
        ar >> m_high_accuracy;
        ar >> m_compact_mode;
        ar >> m_parallel_mode;

        // Recover the function pointers.
        assign_fptrs();

        // Reconstruct the tapes.
        m_tapes.clear();
        construct_tapes();
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // Small helper to fetch the function pointers from the states. Used after des11n or copy.
    void assign_fptrs()
    {
        if (auto *arr_ptr = std::get_if<0>(&m_states)) {
            assert(!m_compact_mode);

            m_fptr_scal = reinterpret_cast<cfunc_ptr_t>((*arr_ptr)[0].jit_lookup("cfunc"));
            m_fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>((*arr_ptr)[1].jit_lookup("cfunc"));
            m_fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>((*arr_ptr)[2].jit_lookup("cfunc"));
        } else {
            assert(m_compact_mode);

            auto &ms = std::get<1>(m_states);

            m_fptr_scal = reinterpret_cast<c_cfunc_ptr_t>(ms.jit_lookup("cfunc.unstrided.batch_size_1"));
            m_fptr_scal_s = reinterpret_cast<c_cfunc_ptr_s_t>(ms.jit_lookup("cfunc.strided.batch_size_1"));
            m_fptr_batch_s = reinterpret_cast<c_cfunc_ptr_s_t>(
                ms.jit_lookup(fmt::format("cfunc.strided.batch_size_{}", m_batch_size)));
        }
    }

    // Small helper to construct the tapes from the information in m_tape_sa.
    // Assumes an empty m_tapes.
    void construct_tapes()
    {
        assert(m_tapes.empty());

        for (const auto [sz, al] : m_tape_sa) {
            m_tapes.push_back(detail::make_aligned_array(sz, al));
        }
    }

    // NOTE: this is necessary only for s11n.
    impl() = default;

    // NOTE: we use a single llvm_state for construction - all the internal
    // llvm_state instances will be either copied from s, or we will use
    // s as a template.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    explicit impl(std::vector<expression> fn, std::vector<expression> vars, llvm_state s,
                  std::optional<std::uint32_t> batch_size, bool high_accuracy, bool compact_mode, bool parallel_mode,
                  long long prec, bool check_prec)
        : m_fn(std::move(fn)), m_vars(std::move(vars)), m_states(std::array{s, s, s}), m_prec(prec),
          m_check_prec(check_prec), m_high_accuracy(high_accuracy), m_compact_mode(compact_mode),
          m_parallel_mode(parallel_mode)
    {
        // Setup the batch size.
        // NOTE: manually specified batch size of zero is interpreted as undefined.
        if (batch_size && *batch_size != 0u) {
            m_batch_size = *batch_size;
        } else {
            m_batch_size = recommended_simd_size<T>();
        }

#if defined(HEYOKA_HAVE_REAL)

        // NOTE: batch size > 1u not supported for real.
        if (std::same_as<T, mppp::real> && m_batch_size > 1u) {
            throw std::invalid_argument("Batch size > 1 is not supported for mppp::real");
        }

#endif

        if (compact_mode) {
            // Build the multi cfunc, and assign the internal members.
            std::tie(m_states, m_dc, m_tape_sa) = detail::make_multi_cfunc<T>(s, "cfunc", m_fn, m_vars, m_batch_size,
                                                                              high_accuracy, m_parallel_mode, prec);

            // Compile.
            std::get<1>(m_states).compile();

            // Assign the function pointers.
            assign_fptrs();

            // Construct the tapes.
            construct_tapes();
        } else {
            auto &s_arr = std::get<0>(m_states);

            // Add the compiled functions.
            oneapi::tbb::parallel_invoke(
                [&]() {
                    // Scalar unstrided.
                    // NOTE: we fetch the decomposition from the scalar
                    // unstrided invocation of add_cfunc().
                    m_dc = add_cfunc<T>(s_arr[0], "cfunc", m_fn, m_vars, kw::high_accuracy = high_accuracy,
                                        kw::prec = prec,
                                        // NOTE: be explicit about the lack of compact mode,
                                        // because the default setting for mppp::real is different
                                        // from the other types and if we leave this unset we will
                                        // get the wrong function.
                                        kw::compact_mode = false);

                    s_arr[0].compile();

                    m_fptr_scal = reinterpret_cast<cfunc_ptr_t>(s_arr[0].jit_lookup("cfunc"));
                },
                [&]() {
                    // Scalar strided.
                    add_cfunc<T>(s_arr[1], "cfunc", m_fn, m_vars, kw::high_accuracy = high_accuracy, kw::prec = prec,
                                 kw::strided = true,
                                 // NOTE: be explicit about the lack of compact mode,
                                 // because the default setting for mppp::real is different
                                 // from the other types and if we leave this unset we will
                                 // get the wrong function.
                                 kw::compact_mode = false);

                    s_arr[1].compile();

                    m_fptr_scal_s = reinterpret_cast<cfunc_ptr_s_t>(s_arr[1].jit_lookup("cfunc"));
                },
                [&]() {
                    // Batch strided.
                    add_cfunc<T>(s_arr[2], "cfunc", m_fn, m_vars, kw::batch_size = m_batch_size,
                                 kw::high_accuracy = high_accuracy, kw::prec = prec, kw::strided = true,
                                 // NOTE: be explicit about the lack of compact mode,
                                 // because the default setting for mppp::real is different
                                 // from the other types and if we leave this unset we will
                                 // get the wrong function.
                                 kw::compact_mode = false);

                    s_arr[2].compile();

                    m_fptr_batch_s = reinterpret_cast<cfunc_ptr_s_t>(s_arr[2].jit_lookup("cfunc"));
                });
        }

        // Let's figure out if fn contains params and if it is time-dependent.
        m_nparams = get_param_size(m_fn);
        m_is_time_dependent = heyoka::is_time_dependent(m_fn);

        // Cache the number of variables and outputs.
        // NOTE: static casts should also be fine here, because add_cfunc()
        // succeeded and that guarantees that the number of vars and outputs
        // fits in a 32-bit int.
        m_nouts = boost::numeric_cast<std::uint32_t>(m_fn.size());
        m_nvars = boost::numeric_cast<std::uint32_t>(m_vars.size());
    }
    impl(const impl &other)
        : m_fn(other.m_fn), m_vars(other.m_vars), m_states(other.m_states), m_batch_size(other.m_batch_size),
          m_dc(other.m_dc), m_tape_sa(other.m_tape_sa), m_nparams(other.m_nparams),
          m_is_time_dependent(other.m_is_time_dependent), m_nouts(other.m_nouts), m_nvars(other.m_nvars),
          m_prec(other.m_prec), m_check_prec(other.m_check_prec), m_high_accuracy(other.m_high_accuracy),
          m_compact_mode(other.m_compact_mode), m_parallel_mode(other.m_parallel_mode)
    {
        // Recover the function pointers.
        assign_fptrs();

        // Construct the tapes.
        construct_tapes();
    }

    // These are never needed.
    impl(impl &&) noexcept = delete;
    impl &operator=(const impl &) = delete;
    impl &operator=(impl &&) noexcept = delete;

    ~impl() = default;
};

template <typename T>
cfunc<T>::cfunc() noexcept = default;

template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
cfunc<T>::cfunc(std::vector<expression> fn, std::vector<expression> vars,
                // NOLINTNEXTLINE(performance-unnecessary-value-param)
                std::tuple<bool, bool, bool, long long, std::optional<std::uint32_t>, llvm_state, bool> tup)
{
    // Unpack the tuple.
    auto &[high_accuracy, compact_mode, parallel_mode, prec, batch_size, s, check_prec] = tup;

    // Construct the impl.
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_impl = std::make_unique<impl>(std::move(fn), std::move(vars), std::move(s), batch_size, high_accuracy,
                                    compact_mode, parallel_mode, prec, check_prec);
}

template <typename T>
cfunc<T>::cfunc(const cfunc &other)
    : m_impl(
          // NOTE: support copy-construction from invalid object.
          other.m_impl ? std::make_unique<impl>(*other.m_impl) : nullptr)
{
}

// NOTE: document that other is left into the def-cted
// state afterwards.
template <typename T>
cfunc<T>::cfunc(cfunc &&) noexcept = default;

template <typename T>
cfunc<T> &cfunc<T>::operator=(const cfunc &other)
{
    if (this != &other) {
        *this = cfunc(other);
    }

    return *this;
}

// NOTE: document that other is left into the def-cted
// state afterwards.
template <typename T>
cfunc<T> &cfunc<T>::operator=(cfunc &&) noexcept = default;

template <typename T>
cfunc<T>::~cfunc() = default;

template <typename T>
bool cfunc<T>::is_valid() const noexcept
{
    return static_cast<bool>(m_impl);
}

template <typename T>
const std::vector<expression> &cfunc<T>::get_fn() const
{
    check_valid(__func__);

    return m_impl->m_fn;
}

template <typename T>
const std::vector<expression> &cfunc<T>::get_vars() const
{
    check_valid(__func__);

    return m_impl->m_vars;
}

template <typename T>
const std::vector<expression> &cfunc<T>::get_dc() const
{
    check_valid(__func__);

    return m_impl->m_dc;
}

template <typename T>
const std::variant<std::array<llvm_state, 3>, llvm_multi_state> &cfunc<T>::get_llvm_states() const
{
    check_valid(__func__);

    return m_impl->m_states;
}

template <typename T>
bool cfunc<T>::get_high_accuracy() const
{
    check_valid(__func__);

    return m_impl->m_high_accuracy;
}

template <typename T>
bool cfunc<T>::get_compact_mode() const
{
    check_valid(__func__);

    return m_impl->m_compact_mode;
}

template <typename T>
bool cfunc<T>::get_parallel_mode() const
{
    check_valid(__func__);

    return m_impl->m_parallel_mode;
}

template <typename T>
std::uint32_t cfunc<T>::get_batch_size() const
{
    check_valid(__func__);

    return m_impl->m_batch_size;
}

template <typename T>
std::uint32_t cfunc<T>::get_nparams() const
{
    check_valid(__func__);

    return m_impl->m_nparams;
}

template <typename T>
std::uint32_t cfunc<T>::get_nvars() const
{
    check_valid(__func__);

    return m_impl->m_nvars;
}

template <typename T>
std::uint32_t cfunc<T>::get_nouts() const
{
    check_valid(__func__);

    return m_impl->m_nouts;
}

template <typename T>
bool cfunc<T>::is_time_dependent() const
{
    check_valid(__func__);

    return m_impl->m_is_time_dependent;
}

#if defined(HEYOKA_HAVE_REAL)

template <typename T>
mpfr_prec_t cfunc<T>::get_prec() const
    requires std::same_as<T, mppp::real>
{
    check_valid(__func__);

    // NOTE: the cast here is safe because the value of
    // m_prec was checked inside add_cfunc() during the
    // construction of this object.
    return static_cast<mpfr_prec_t>(m_impl->m_prec);
}

#endif

template <typename T>
void cfunc<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_impl;
}

template <typename T>
void cfunc<T>::load(boost::archive::binary_iarchive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<cfunc<T>>::type::value)) {
        throw std::invalid_argument(
            fmt::format("Unable to load a cfunc: the archive version ({}) is too old", version));
    }
    // LCOV_EXCL_STOP

    try {
        ar >> m_impl;
        // LCOV_EXCL_START
    } catch (...) {
        *this = cfunc{};

        throw;
    }
    // LCOV_EXCL_STOP
}

template <typename T>
void cfunc<T>::check_valid(const char *name) const
{
    if (!m_impl) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The function '{}' cannot be invoked on an invalid cfunc object", name));
    }
}

template <typename T>
void cfunc<T>::single_eval(out_1d outputs, in_1d inputs, std::optional<in_1d> pars, std::optional<T> time)
{
    check_valid(__func__);

    // Check the arguments.
    if (outputs.size() != m_impl->m_nouts) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid outputs array passed to a cfunc: the number of function "
                                                "outputs is {}, but the outputs array has a size of {}",
                                                m_impl->m_nouts, outputs.size()));
    }

    if (inputs.size() != m_impl->m_nvars) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid inputs array passed to a cfunc: the number of function "
                                                "inputs is {}, but the inputs array has a size of {}",
                                                m_impl->m_nvars, inputs.size()));
    }

    if (m_impl->m_nparams != 0u && !pars) [[unlikely]] {
        throw std::invalid_argument(
            "An array of parameter values must be passed in order to evaluate a function with parameters");
    }

    if (pars) {
        if (pars->size() != m_impl->m_nparams) [[unlikely]] {
            throw std::invalid_argument(fmt::format("The array of parameter values provided for the evaluation "
                                                    "of a compiled function has {} element(s), "
                                                    "but the number of parameters in the function is {}",
                                                    pars->size(), m_impl->m_nparams));
        }
    }

    if (m_impl->m_is_time_dependent && !time) [[unlikely]] {
        throw std::invalid_argument("A time value must be provided in order to evaluate a time-dependent function");
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::same_as<T, mppp::real>) {
        if (m_impl->m_check_prec) {
            const auto prec_checker = [&](const auto &x) {
                if (x.get_prec() != m_impl->m_prec) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "An mppp::real with an invalid precision of {} was detected in the arguments to the evaluation "
                        "of a compiled function - the expected precision value is {}",
                        x.get_prec(), m_impl->m_prec));
                }
            };

            for (std::size_t i = 0; i < outputs.extent(0); ++i) {
                prec_checker(outputs(i));
            }

            for (std::size_t i = 0; i < inputs.extent(0); ++i) {
                prec_checker(inputs(i));
            }

            if (pars) {
                for (std::size_t i = 0; i < pars->extent(0); ++i) {
                    prec_checker((*pars)(i));
                }
            }

            if (time) {
                prec_checker(*time);
            }
        }
    }

#endif

    // Invoke the compiled function.
    if (m_impl->m_compact_mode) {
        assert(!m_impl->m_tapes.empty());

        std::get<1>(m_impl->m_fptr_scal)(outputs.data_handle(), inputs.data_handle(),
                                         pars ? pars->data_handle() : nullptr, time ? &*time : nullptr,
                                         m_impl->m_tapes[0].get());
    } else {
        std::get<0>(m_impl->m_fptr_scal)(outputs.data_handle(), inputs.data_handle(),
                                         pars ? pars->data_handle() : nullptr, time ? &*time : nullptr);
    }
}

template <typename T>
void cfunc<T>::multi_eval_st(out_2d outputs, in_2d inputs, std::optional<in_2d> pars, std::optional<in_1d> times)
{
    // Cache the number of evals.
    const auto nevals = outputs.extent(1);

    // Cache the batch size.
    const auto batch_size = m_impl->m_batch_size;

    // Cache the function pointers.
    auto fptr_batch_s = m_impl->m_fptr_batch_s;
    auto fptr_scal_s = m_impl->m_fptr_scal_s;

    // Cache the compact mode flag.
    const auto compact_mode = m_impl->m_compact_mode;

    // Cache a reference to the tapes.
    const auto &tapes = m_impl->m_tapes;

    // Number of simd blocks in the arrays.
    const auto n_simd_blocks = nevals / batch_size;

    // Fetch the pointers.
    auto *out_data = outputs.data_handle();
    const auto *in_data = inputs.data_handle();
    const auto *par_data = pars ? pars->data_handle() : nullptr;
    const auto *time_data = times ? times->data_handle() : nullptr;

    // NOTE: the idea of these booleans is that we want to do arithmetics
    // on the inputs/pars/time pointers only if we know that we **must** read from them,
    // in which case the validation steps taken earlier ensure that
    // arithmetics on them is safe. Otherwise, there are certain corner cases in which
    // we might end up doing pointer arithmetics which leads to UB. Two examples:
    //
    // - the function has no inputs, or
    // - the function has no params but the user anyway passed an empty
    //   array of par values.
    //
    // In these two cases we are dealing with input and/or pars arrays of shape
    // (0, nevals). If the pointers stored in the mdspans are null, then we would
    // be committing UB by doing arithmetic on them.
    //
    // NOTE: if nevals is zero, then the two for loops below are never
    // entered and we never end up doing arithmetics on potentially-null
    // pointers.
    //
    // NOTE: in case of the outputs array, the data pointer can never be null
    // (unless nevals is zero) because we do not allow to construct a cfunc
    // object for a function with zero outputs. Hence, no checks are needed.
    const auto read_inputs = m_impl->m_nvars > 0u;
    const auto read_pars = m_impl->m_nparams > 0u;
    const auto read_time = m_impl->m_is_time_dependent;
    assert(m_impl->m_nouts > 0u);

    // Evaluate over the simd blocks.
    if (compact_mode) {
        // NOTE: the batch-mode tape is at index 1 only if the batch
        // size is > 1, otherwise we are using the scalar tape.
        auto *batch_tape_ptr = tapes[batch_size > 1u].get();

        auto *fptr = std::get<1>(fptr_batch_s);

        for (std::size_t k = 0; k < n_simd_blocks; ++k) {
            const auto start_offset = k * batch_size;

            fptr(out_data + start_offset, read_inputs ? in_data + start_offset : nullptr,
                 read_pars ? par_data + start_offset : nullptr, read_time ? time_data + start_offset : nullptr,
                 batch_tape_ptr, nevals);
        }
    } else {
        auto *fptr = std::get<0>(fptr_batch_s);

        for (std::size_t k = 0; k < n_simd_blocks; ++k) {
            const auto start_offset = k * batch_size;

            fptr(out_data + start_offset, read_inputs ? in_data + start_offset : nullptr,
                 read_pars ? par_data + start_offset : nullptr, read_time ? time_data + start_offset : nullptr, nevals);
        }
    }

    // Handle the remainder, if present.
    if (compact_mode) {
        auto *scalar_tape_ptr = tapes[0].get();

        auto *fptr = std::get<1>(fptr_scal_s);

        for (auto k = n_simd_blocks * batch_size; k < nevals; ++k) {
            fptr(out_data + k, read_inputs ? in_data + k : nullptr, read_pars ? par_data + k : nullptr,
                 read_time ? time_data + k : nullptr, scalar_tape_ptr, nevals);
        }
    } else {
        auto *fptr = std::get<0>(fptr_scal_s);

        for (auto k = n_simd_blocks * batch_size; k < nevals; ++k) {
            fptr(out_data + k, read_inputs ? in_data + k : nullptr, read_pars ? par_data + k : nullptr,
                 read_time ? time_data + k : nullptr, nevals);
        }
    }
}

template <typename T>
void cfunc<T>::multi_eval_mt(out_2d outputs, in_2d inputs, std::optional<in_2d> pars, std::optional<in_1d> times)
{
    // Cache the number of evals.
    const auto nevals = outputs.extent(1);

    // Cache the batch size.
    const auto batch_size = m_impl->m_batch_size;

    // Cache the function pointers.
    auto fptr_batch_s = m_impl->m_fptr_batch_s;
    auto fptr_scal_s = m_impl->m_fptr_scal_s;

    // Cache the compact mode flag.
    const auto compact_mode = m_impl->m_compact_mode;

    // Cache a reference to the tapes.
    const auto &tapes = m_impl->m_tapes;

    // Fetch the pointers.
    auto *out_data = outputs.data_handle();
    const auto *in_data = inputs.data_handle();
    const auto *par_data = pars ? pars->data_handle() : nullptr;
    const auto *time_data = times ? times->data_handle() : nullptr;

    // NOTE: the idea of these booleans is that we want to do arithmetics
    // on the inputs/pars/time pointers only if we know that we **must** read from them,
    // in which case the validation steps taken earlier ensure that
    // arithmetics on them is safe. Otherwise, there are certain corner cases in which
    // we might end up doing pointer arithmetics which leads to UB. Two examples:
    //
    // - the function has no inputs, or
    // - the function has no params but the user anyway passed an empty
    //   array of par values.
    //
    // In these two cases we are dealing with input and/or pars arrays of shape
    // (0, nevals). If the pointers stored in the mdspans are null, then we would
    // be committing UB by doing arithmetic on them.
    //
    // NOTE: if nevals is zero, then the two for loops below are never
    // entered and we never end up doing arithmetics on potentially-null
    // pointers.
    //
    // NOTE: in case of the outputs array, the data pointer can never be null
    // (unless nevals is zero) because we do not allow to construct a cfunc
    // object for a function with zero outputs. Hence, no checks are needed.
    const auto read_inputs = m_impl->m_nvars > 0u;
    const auto read_pars = m_impl->m_nparams > 0u;
    const auto read_time = m_impl->m_is_time_dependent;
    assert(m_impl->m_nouts > 0u);

    // Number of simd blocks in the arrays.
    const auto n_simd_blocks = nevals / batch_size;

    // The functor to evaluate the scalar remainder, if present. It will be run concurrently with
    // the batch-parallel iterations.
    // NOTE: in compact mode, this function uses the existing scalar tape. It is invoked concurrently
    // with batch_iter, which, in compact mode, uses thread-local tapes and does not use
    // any of the existing tapes in the class. Thus, there is no danger of data races.
    const auto scalar_rem = [n_simd_blocks, batch_size, fptr_scal_s, nevals, out_data, read_inputs, in_data, read_pars,
                             par_data, read_time, time_data, &tapes]<bool CM>() {
        auto *fptr = std::get<(CM ? 1 : 0)>(fptr_scal_s);
        auto *scalar_tape_ptr = CM ? tapes[0].get() : nullptr;

        for (auto k = n_simd_blocks * batch_size; k < nevals; ++k) {
            if constexpr (CM) {
                fptr(out_data + k, read_inputs ? in_data + k : nullptr, read_pars ? par_data + k : nullptr,
                     read_time ? time_data + k : nullptr, scalar_tape_ptr, nevals);
            } else {
                fptr(out_data + k, read_inputs ? in_data + k : nullptr, read_pars ? par_data + k : nullptr,
                     read_time ? time_data + k : nullptr, nevals);
            }
        }
    };

    // The functor to evaluate the batch-parallel iterations.
    const auto batch_iter = [batch_size, out_data, read_inputs, in_data, read_pars, par_data, read_time, time_data,
                             nevals, fptr_batch_s]<bool CM>(const auto &range, void *tape_ptr) {
        auto *fptr = std::get<(CM ? 1 : 0)>(fptr_batch_s);

        for (auto k = range.begin(); k != range.end(); ++k) {
            const auto start_offset = k * batch_size;

            if constexpr (CM) {
                assert(tape_ptr != nullptr);
                fptr(out_data + start_offset, read_inputs ? in_data + start_offset : nullptr,
                     read_pars ? par_data + start_offset : nullptr, read_time ? time_data + start_offset : nullptr,
                     tape_ptr, nevals);
            } else {
                assert(tape_ptr == nullptr);
                fptr(out_data + start_offset, read_inputs ? in_data + start_offset : nullptr,
                     read_pars ? par_data + start_offset : nullptr, read_time ? time_data + start_offset : nullptr,
                     nevals);
            }
        }
    };

    // NOTE: in compact mode each batch-parallel iteration needs its own tape.
    if (compact_mode) {
        // Construct the thread-specific storage for batch parallel operations.
        typename impl::ets_t ets_batch([this, batch_size]() {
            // NOTE: the batch-mode tape is at index 1 only if the batch
            // size is > 1, otherwise we are using the scalar tape.
            return detail::make_aligned_array(m_impl->m_tape_sa[batch_size > 1u][0],
                                              m_impl->m_tape_sa[batch_size > 1u][1]);
        });

        oneapi::tbb::parallel_invoke(
            [&ets_batch, &batch_iter, n_simd_blocks]() {
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_simd_blocks),
                                          [&ets_batch, &batch_iter](const auto &range) {
                                              // Fetch the local tape.
                                              auto *tape_ptr = ets_batch.local().get();

                                              // NOTE: there are well-known pitfalls when using thread-specific
                                              // storage with nested parallelism:
                                              //
                                              // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
                                              //
                                              // If parallel mode is active in the cfunc, then the current thread
                                              // will block as execution in the parallel region of the cfunc begins. The
                                              // blocked thread could then grab another task from the parallel for loop
                                              // we are currently in, and it would then start writing for a second time
                                              // into the same tape it already begun writing into, leading to UB.
                                              oneapi::tbb::this_task_arena::isolate(
                                                  [&]() { batch_iter.template operator()<true>(range, tape_ptr); });
                                          });
            },
            [&scalar_rem]() { scalar_rem.template operator()<true>(); });
    } else {
        oneapi::tbb::parallel_invoke(
            [fptr_batch_s = m_impl->m_fptr_batch_s, &batch_iter, n_simd_blocks]() {
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0, n_simd_blocks),
                    [&batch_iter](const auto &range) { batch_iter.template operator()<false>(range, nullptr); });
            },
            [&scalar_rem]() { scalar_rem.template operator()<false>(); });
    }
}

template <typename T>
void cfunc<T>::multi_eval(out_2d outputs, in_2d inputs, std::optional<in_2d> pars, std::optional<in_1d> times)
{
    check_valid(__func__);

    // Arguments validation.
    if (outputs.extent(0) != m_impl->m_nouts) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid outputs array passed to a cfunc: the number of function "
                                                "outputs is {}, but the number of rows in the outputs array is {}",
                                                m_impl->m_nouts, outputs.extent(0)));
    }

    // Fetch the number of columns from outputs.
    const auto ncols = outputs.extent(1);

    if (inputs.extent(0) != m_impl->m_nvars) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid inputs array passed to a cfunc: the number of function "
                                                "inputs is {}, but the number of rows in the inputs array is {}",
                                                m_impl->m_nvars, inputs.extent(0)));
    }

    if (inputs.extent(1) != ncols) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid inputs array passed to a cfunc: the expected number of columns deduced from the "
                        "outputs array is {}, but the number of columns in the inputs array is {}",
                        ncols, inputs.extent(1)));
    }

    if (m_impl->m_nparams != 0u && !pars) [[unlikely]] {
        throw std::invalid_argument(
            "An array of parameter values must be passed in order to evaluate a function with parameters");
    }

    if (pars) {
        if (pars->extent(0) != m_impl->m_nparams) [[unlikely]] {
            throw std::invalid_argument(fmt::format("The array of parameter values provided for the evaluation "
                                                    "of a compiled function has {} row(s), "
                                                    "but the number of parameters in the function is {}",
                                                    pars->extent(0), m_impl->m_nparams));
        }

        if (pars->extent(1) != ncols) [[unlikely]] {
            throw std::invalid_argument(fmt::format("The array of parameter values provided for the evaluation "
                                                    "of a compiled function has {} column(s), "
                                                    "but the expected number of columns deduced from the "
                                                    "outputs array is {}",
                                                    pars->extent(1), ncols));
        }
    }

    if (m_impl->m_is_time_dependent && !times) [[unlikely]] {
        throw std::invalid_argument(
            "An array of time values must be provided in order to evaluate a time-dependent function");
    }

    if (times) {
        if (times->size() != ncols) [[unlikely]] {
            throw std::invalid_argument(fmt::format("The array of time values provided for the evaluation "
                                                    "of a compiled function has a size of {}, "
                                                    "but the expected size deduced from the "
                                                    "outputs array is {}",
                                                    times->size(), ncols));
        }
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::same_as<T, mppp::real>) {
        if (m_impl->m_check_prec) {
            const auto prec_checker = [&](const auto &x) {
                if (x.get_prec() != m_impl->m_prec) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "An mppp::real with an invalid precision of {} was detected in the arguments to the evaluation "
                        "of a compiled function - the expected precision value is {}",
                        x.get_prec(), m_impl->m_prec));
                }
            };

            for (std::size_t i = 0; i < outputs.extent(0); ++i) {
                for (std::size_t j = 0; j < outputs.extent(1); ++j) {
                    prec_checker(outputs(i, j));
                }
            }

            for (std::size_t i = 0; i < inputs.extent(0); ++i) {
                for (std::size_t j = 0; j < inputs.extent(1); ++j) {
                    prec_checker(inputs(i, j));
                }
            }

            if (pars) {
                for (std::size_t i = 0; i < pars->extent(0); ++i) {
                    for (std::size_t j = 0; j < pars->extent(1); ++j) {
                        prec_checker((*pars)(i, j));
                    }
                }
            }

            if (times) {
                for (std::size_t i = 0; i < times->extent(0); ++i) {
                    prec_checker((*times)(i));
                }
            }
        }
    }

#endif

    // A simple cost model for deciding when to parallelise over ncols. We consider:
    //
    // - an estimate of the total number of elementary operations necessary to
    //   evaluate the function,
    // - the value of ncols,
    // - the floating-point type in use,
    // - the batch size.

    // Cost of a scalar fp operation.
    constexpr auto fp_unit_cost = []() -> double {
        if constexpr (std::same_as<float, T> || std::same_as<double, T>) {
            // float and double.
            return 1;
        } else if constexpr (std::same_as<long double, T>) {
            // long double.
            if constexpr (detail::is_ieee754_binary64<T>) {
                return 1;
            } else if constexpr (detail::is_x86_fp80<T>) {
                return 5;
            } else if constexpr (detail::is_ieee754_binary128<T>) {
#if defined(HEYOKA_ARCH_PPC)
                return 10;
#else
                return 100;
#endif
            } else {
#if defined(HEYOKA_ARCH_PPC)
                // Double-double implementation.
                return 5;
#else
                static_assert(detail::always_false_v<T>, "Unknown fp cost model.");
#endif
            }
        }
#if defined(HEYOKA_HAVE_REAL128)
        else if constexpr (std::same_as<mppp::real128, T>) {
#if defined(HEYOKA_ARCH_PPC)
            return 10;
#else
            return 100;
#endif
        }
#endif
#if defined(HEYOKA_HAVE_REAL)
        else if constexpr (std::same_as<mppp::real, T>) {
            // NOTE: this should be improved to take into account
            // the selected precision.
            // NOTE: for reference, mppp::real with 113 bits of precision
            // is slightly slower than software-implemented quadmath.
            return 1000;
        }
#endif
        else {
            static_assert(detail::always_false_v<T>, "Unknown fp cost model.");
        }
    }();

    // Total number of fp operations: number of elementary subexpressions in the
    // decomposition * ncols.
    assert(m_impl->m_dc.size() >= m_impl->m_nouts);
    assert(m_impl->m_dc.size() - m_impl->m_nouts >= m_impl->m_nvars);
    const auto tot_n_flops
        = static_cast<double>(m_impl->m_dc.size() - m_impl->m_nouts - m_impl->m_nvars) * static_cast<double>(ncols);

    // The total work.
    // NOTE: clearly if the user specifies a custom value larger than the native SIMD size here
    // for m_batch_size the estimation will be off. Using recommended_simd_size() here is not much
    // better in that respect either (since the user could still specify a smaller batch size), plus
    // it is going to be somewhat expensive due to accessing a thread-safe static global.
    const auto tot_work = (fp_unit_cost / m_impl->m_batch_size) * static_cast<double>(tot_n_flops);

    // NOTE: the commonly-quoted figure on the internet is a threshold of 10'000 clock cycles
    // for parallel work.
    if (tot_work >= 1e4) {
        multi_eval_mt(outputs, inputs, pars, times);
    } else {
        multi_eval_st(outputs, inputs, pars, times);
    }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const cfunc<T> &cf)
{
    os << fmt::format("C++ datatype: {}\n", boost::core::demangle(typeid(T).name()));

    if (cf.is_valid()) {
#if defined(HEYOKA_HAVE_REAL)

        if constexpr (std::same_as<T, mppp::real>) {
            os << fmt::format("Precision: {}\n", cf.get_prec());
        }

#endif

        os << fmt::format("Variables: {}\n", cf.get_vars());

        for (decltype(cf.get_fn().size()) i = 0; i < cf.get_fn().size(); ++i) {
            os << fmt::format("Output #{}: {}\n", i, cf.get_fn()[i]);
        }
    } else {
        os << "Default-constructed state.\n";
    }

    return os;
}

// Explicit instantiations.
#define HEYOKA_CFUNC_CLASS_EXPLICIT_INST(T)                                                                            \
    template class HEYOKA_DLL_PUBLIC cfunc<T>;                                                                         \
    template HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const cfunc<T> &);

HEYOKA_CFUNC_CLASS_EXPLICIT_INST(float)
HEYOKA_CFUNC_CLASS_EXPLICIT_INST(double)
HEYOKA_CFUNC_CLASS_EXPLICIT_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_CFUNC_CLASS_EXPLICIT_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_CFUNC_CLASS_EXPLICIT_INST(mppp::real)

#endif

#undef HEYOKA_CFUNC_CLASS_EXPLICIT_INST

HEYOKA_END_NAMESPACE
