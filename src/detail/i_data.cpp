// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include <boost/serialization/array.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/aligned_buffer.hpp>
#include <heyoka/detail/i_data.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/variant_s11n.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

// NOTE: implement a workaround for the serialisation of tuples whose first element
// is a taylor outcome. We need this because Boost.Serialization treats all enums
// as ints, which is not ok for taylor_outcome (whose underyling type will not
// be an int on most platforms). Because it is not possible to override Boost's
// enum implementation, we override the serialisation of tuples with outcomes
// as first elements, which is all we need in the serialisation of the batch
// integrator. The implementation below will be preferred over the generic tuple
// s11n because it is more specialised.
// NOTE: this workaround is not necessary for the other enums in heyoka because
// those all have ints as underlying type.
namespace boost::serialization
{

template <typename Archive, typename... Args>
// NOLINTNEXTLINE(misc-use-internal-linkage)
void save(Archive &ar, const std::tuple<heyoka::taylor_outcome, Args...> &tup, unsigned)
{
    auto tf = [&ar](const auto &x) {
        if constexpr (std::is_same_v<decltype(x), const heyoka::taylor_outcome &>) {
            ar << static_cast<std::underlying_type_t<heyoka::taylor_outcome>>(x);
        } else {
            ar << x;
        }
    };

    // NOTE: this is a right fold, which, in conjunction with the
    // builtin comma operator, ensures that the serialisation of
    // the tuple elements proceeds in the correct order and with
    // the correct sequencing.
    std::apply([&tf](const auto &...x) { (tf(x), ...); }, tup);
}

template <typename Archive, typename... Args>
// NOLINTNEXTLINE(misc-use-internal-linkage)
void load(Archive &ar, std::tuple<heyoka::taylor_outcome, Args...> &tup, unsigned)
{
    auto tf = [&ar](auto &x) {
        if constexpr (std::is_same_v<decltype(x), heyoka::taylor_outcome &>) {
            std::underlying_type_t<heyoka::taylor_outcome> val{};
            ar >> val;

            x = static_cast<heyoka::taylor_outcome>(val);
        } else {
            ar >> x;
        }
    };

    std::apply([&tf](auto &...x) { (tf(x), ...); }, tup);
}

template <typename Archive, typename... Args>
// NOLINTNEXTLINE(misc-use-internal-linkage)
void serialize(Archive &ar, std::tuple<heyoka::taylor_outcome, Args...> &tup, unsigned v)
{
    split_free(ar, tup, v);
}

} // namespace boost::serialization

HEYOKA_BEGIN_NAMESPACE

// Helper to initialise the compact-mode tape. Assumes an empty tape.
template <typename T>
void taylor_adaptive<T>::i_data::init_cm_tape()
{
    assert(!m_tape);

    const auto [sz, al] = m_tape_sa;

    if (m_compact_mode) {
        assert(sz != 0u); // LCOV_EXCL_LINE
        assert(al != 0u); // LCOV_EXCL_LINE

        m_tape = detail::make_aligned_buffer(sz, al);
    } else {
        assert(sz == 0u); // LCOV_EXCL_LINE
        assert(al == 0u); // LCOV_EXCL_LINE
    }
}

template <typename T>
void taylor_adaptive<T>::i_data::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_state;
    ar << m_time;
    ar << m_llvm_state;
    ar << m_tplt_state;
    ar << m_dim;
    ar << m_dc;
    ar << m_order;
    ar << m_tol;
    ar << m_high_accuracy;
    ar << m_compact_mode;
    ar << m_tape_sa;
    ar << m_pars;
    ar << m_tc;
    ar << m_last_h;
    ar << m_d_out;
    ar << m_vsys;
    ar << m_tm_data;
}

template <typename T>
void taylor_adaptive<T>::i_data::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_state;
    ar >> m_time;
    ar >> m_llvm_state;
    ar >> m_tplt_state;
    ar >> m_dim;
    ar >> m_dc;
    ar >> m_order;
    ar >> m_tol;
    ar >> m_high_accuracy;
    ar >> m_compact_mode;
    ar >> m_tape_sa;
    ar >> m_pars;
    ar >> m_tc;
    ar >> m_last_h;
    ar >> m_d_out;
    ar >> m_vsys;
    ar >> m_tm_data;

    // Recover the function pointers.
    // NOTE: here we are recovering only the dense output function pointer because recovering
    // the correct stepper requires information which is available only from the integrator
    // class (hence, we do it from there).
    m_d_out_f = std::visit([](auto &s) { return reinterpret_cast<d_out_f_t>(s.jit_lookup("d_out_f")); }, m_llvm_state);

    // Reconstruct the compact mode tape, if necessary.
    m_tape.reset();
    init_cm_tape();
}

// NOTE: this ctor provides only partial initialisation of the data members.
// The rest of the initialisation is performed from the integrator ctor.
// NOTE: m_llvm_state is inited as a single llvm_state regardless of the use
// of compact mode. It will be converted into a multi state if needed at a
// later stage.
template <typename T>
taylor_adaptive<T>::i_data::i_data(llvm_state s)
    : m_llvm_state(std::move(s)), m_tplt_state(std::get<0>(m_llvm_state).make_similar())
{
}

template <typename T>
taylor_adaptive<T>::i_data::i_data(const i_data &other)
    : m_state(other.m_state), m_time(other.m_time), m_llvm_state(other.m_llvm_state), m_tplt_state(other.m_tplt_state),
      m_dim(other.m_dim), m_dc(other.m_dc), m_order(other.m_order), m_tol(other.m_tol),
      m_high_accuracy(other.m_high_accuracy), m_compact_mode(other.m_compact_mode), m_tape_sa(other.m_tape_sa),
      m_pars(other.m_pars), m_tc(other.m_tc), m_last_h(other.m_last_h), m_d_out(other.m_d_out), m_vsys(other.m_vsys),
      m_tm_data(other.m_tm_data)
{
    // Recover the function pointers.
    // NOTE: here we are recovering only the dense output function pointer because recovering
    // the correct stepper requires information which is available only from the integrator
    // class (hence, we do it from there).
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_d_out_f = std::visit([](auto &s) { return reinterpret_cast<d_out_f_t>(s.jit_lookup("d_out_f")); }, m_llvm_state);

    // Init the compact mode tape, if necessary.
    init_cm_tape();
}

template <typename T>
taylor_adaptive<T>::i_data::i_data() = default;

template <typename T>
taylor_adaptive<T>::i_data::~i_data() = default;

// Explicit instantiations.
#define HEYOKA_TAYLOR_ADAPTIVE_I_DATA_INST(F) template struct taylor_adaptive<F>::i_data;

HEYOKA_TAYLOR_ADAPTIVE_I_DATA_INST(float)
HEYOKA_TAYLOR_ADAPTIVE_I_DATA_INST(double)
HEYOKA_TAYLOR_ADAPTIVE_I_DATA_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_ADAPTIVE_I_DATA_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_TAYLOR_ADAPTIVE_I_DATA_INST(mppp::real)

#endif

#undef HEYOKA_TAYLOR_ADAPTIVE_I_DATA_INST

// Helper to initialise the compact-mode tape. Assumes an empty tape.
template <typename T>
void taylor_adaptive_batch<T>::i_data::init_cm_tape()
{
    assert(!m_tape);

    const auto [sz, al] = m_tape_sa;

    if (m_compact_mode) {
        assert(sz != 0u); // LCOV_EXCL_LINE
        assert(al != 0u); // LCOV_EXCL_LINE

        m_tape = detail::make_aligned_buffer(sz, al);
    } else {
        assert(sz == 0u); // LCOV_EXCL_LINE
        assert(al == 0u); // LCOV_EXCL_LINE
    }
}

template <typename T>
void taylor_adaptive_batch<T>::i_data::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_batch_size;
    ar << m_state;
    ar << m_time_hi;
    ar << m_time_lo;
    ar << m_llvm_state;
    ar << m_tplt_state;
    ar << m_dim;
    ar << m_dc;
    ar << m_order;
    ar << m_tol;
    ar << m_high_accuracy;
    ar << m_compact_mode;
    ar << m_tape_sa;
    ar << m_pars;
    ar << m_tc;
    ar << m_last_h;
    ar << m_d_out;
    ar << m_pinf;
    ar << m_minf;
    ar << m_delta_ts;
    ar << m_step_res;
    ar << m_prop_res;
    ar << m_ts_count;
    ar << m_min_abs_h;
    ar << m_max_abs_h;
    ar << m_cur_max_delta_ts;
    ar << m_pfor_ts;
    ar << m_t_dir;
    ar << m_rem_time;
    ar << m_time_copy_hi;
    ar << m_time_copy_lo;
    ar << m_nf_detected;
    ar << m_d_out_time;
    ar << m_vsys;
    ar << m_tm_data;
}

template <typename T>
void taylor_adaptive_batch<T>::i_data::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_batch_size;
    ar >> m_state;
    ar >> m_time_hi;
    ar >> m_time_lo;
    ar >> m_llvm_state;
    ar >> m_tplt_state;
    ar >> m_dim;
    ar >> m_dc;
    ar >> m_order;
    ar >> m_tol;
    ar >> m_high_accuracy;
    ar >> m_compact_mode;
    ar >> m_tape_sa;
    ar >> m_pars;
    ar >> m_tc;
    ar >> m_last_h;
    ar >> m_d_out;
    ar >> m_pinf;
    ar >> m_minf;
    ar >> m_delta_ts;
    ar >> m_step_res;
    ar >> m_prop_res;
    ar >> m_ts_count;
    ar >> m_min_abs_h;
    ar >> m_max_abs_h;
    ar >> m_cur_max_delta_ts;
    ar >> m_pfor_ts;
    ar >> m_t_dir;
    ar >> m_rem_time;
    ar >> m_time_copy_hi;
    ar >> m_time_copy_lo;
    ar >> m_nf_detected;
    ar >> m_d_out_time;
    ar >> m_vsys;
    ar >> m_tm_data;

    // Recover the function pointers.
    // NOTE: here we are recovering only the dense output function pointer because recovering
    // the correct stepper requires information which is available only from the integrator
    // class (hence, we do it from there).
    m_d_out_f = std::visit([](auto &s) { return reinterpret_cast<d_out_f_t>(s.jit_lookup("d_out_f")); }, m_llvm_state);

    // Reconstruct the compact mode tape, if necessary.
    m_tape.reset();
    init_cm_tape();
}

// NOTE: this ctor provides only partial initialisation of the data members.
// The rest of the initialisation is performed from the integrator ctor.
// NOTE: m_llvm_state is inited as a single llvm_state regardless of the use
// of compact mode. It will be converted into a multi state if needed at a
// later stage.
template <typename T>
taylor_adaptive_batch<T>::i_data::i_data(llvm_state s)
    : m_llvm_state(std::move(s)), m_tplt_state(std::get<0>(m_llvm_state).make_similar())
{
}

template <typename T>
taylor_adaptive_batch<T>::i_data::i_data(const i_data &other)
    : m_batch_size(other.m_batch_size), m_state(other.m_state), m_time_hi(other.m_time_hi), m_time_lo(other.m_time_lo),
      m_llvm_state(other.m_llvm_state), m_tplt_state(other.m_tplt_state), m_dim(other.m_dim), m_dc(other.m_dc),
      m_order(other.m_order), m_tol(other.m_tol), m_high_accuracy(other.m_high_accuracy),
      m_compact_mode(other.m_compact_mode), m_tape_sa(other.m_tape_sa), m_pars(other.m_pars), m_tc(other.m_tc),
      m_last_h(other.m_last_h), m_d_out(other.m_d_out), m_pinf(other.m_pinf), m_minf(other.m_minf),
      m_delta_ts(other.m_delta_ts), m_step_res(other.m_step_res), m_prop_res(other.m_prop_res),
      m_ts_count(other.m_ts_count), m_min_abs_h(other.m_min_abs_h), m_max_abs_h(other.m_max_abs_h),
      m_cur_max_delta_ts(other.m_cur_max_delta_ts), m_pfor_ts(other.m_pfor_ts), m_t_dir(other.m_t_dir),
      m_rem_time(other.m_rem_time), m_time_copy_hi(other.m_time_copy_hi), m_time_copy_lo(other.m_time_copy_lo),
      m_nf_detected(other.m_nf_detected), m_d_out_time(other.m_d_out_time), m_vsys(other.m_vsys),
      m_tm_data(other.m_tm_data)
{
    // Recover the function pointers.
    // NOTE: here we are recovering only the dense output function pointer because recovering
    // the correct stepper requires information which is available only from the integrator
    // class (hence, we do it from there).
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_d_out_f = std::visit([](auto &s) { return reinterpret_cast<d_out_f_t>(s.jit_lookup("d_out_f")); }, m_llvm_state);

    // Init the compact mode tape, if necessary.
    init_cm_tape();
}

template <typename T>
taylor_adaptive_batch<T>::i_data::i_data() = default;

template <typename T>
taylor_adaptive_batch<T>::i_data::~i_data() = default;

// Explicit instantiations.
#define HEYOKA_TAYLOR_ADAPTIVE_BATCH_I_DATA_INST(F) template struct taylor_adaptive_batch<F>::i_data;

HEYOKA_TAYLOR_ADAPTIVE_BATCH_I_DATA_INST(float)
HEYOKA_TAYLOR_ADAPTIVE_BATCH_I_DATA_INST(double)
HEYOKA_TAYLOR_ADAPTIVE_BATCH_I_DATA_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_TAYLOR_ADAPTIVE_BATCH_I_DATA_INST(mppp::real128)

#endif

#undef HEYOKA_TAYLOR_ADAPTIVE_BATCH_I_DATA_INST

HEYOKA_END_NAMESPACE
