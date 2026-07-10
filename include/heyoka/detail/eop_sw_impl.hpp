// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EOP_SW_IMPL_HPP
#define HEYOKA_DETAIL_EOP_SW_IMPL_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>

// This header declares common base classes and macros for the implementation of EOP/SW quantities and their derivatives
// in the expression system.

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

// Base class common to the implementation of EOP/SW quantities and their derivatives.
template <typename Data>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS eop_sw_impl_base : public func_base
{
    // The data table.
    //
    // NOTE: this is wrapped into an optional so that default construction (which is necessary only for serialisation
    // purposes) is cheap and does not require the actual storage of any EOP/SW data. In normal usage scenarios, we are
    // always expecting the optional to be non-empty.
    std::optional<Data> m_data;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
    // NOTE: we never expect to use this class directly, thus we can mark special member functions as protected.
    eop_sw_impl_base();
    explicit eop_sw_impl_base(const std::string &, const std::string &, expression, Data);
    eop_sw_impl_base(const eop_sw_impl_base &);
    eop_sw_impl_base(eop_sw_impl_base &&) noexcept;
    eop_sw_impl_base &operator=(const eop_sw_impl_base &);
    eop_sw_impl_base &operator=(eop_sw_impl_base &&) noexcept;
    // NOTE: make this virtual. Even if we never explicitly destroy derived classes via base pointers, the Boost s11n
    // machinery seemingly ends up compiling code that does. It is not clear 100% whether or not this code ends up
    // actually executing at runtime, but let us be defensive about this potential issue.
    virtual ~eop_sw_impl_base();

    [[nodiscard]] const Data &checked_get_data() const;
    [[nodiscard]] llvm::Value *llvm_eval_helper(llvm_state &, unsigned, llvm::Value *, llvm::Type *,
                                                std::uint32_t) const;
    // NOTE: this is used to generate/fetch the LLVM function that evaluates at the same time the EOP/SW quantity and
    // its derivative.
    [[nodiscard]] virtual llvm::Function *get_llvm_eval_f(llvm_state &, llvm::Type *, std::uint32_t, const Data &) const
        = 0;
};

// Base class for the implementation of an EOP/SW quantity.
template <typename Data>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS eop_sw_impl : public eop_sw_impl_base<Data>
{
    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
    // NOTE: we never expect to use this class directly, thus we can mark special member functions as protected.
    eop_sw_impl();
    explicit eop_sw_impl(const std::string &, const std::string &, expression, Data);
    eop_sw_impl(const eop_sw_impl &);
    eop_sw_impl(eop_sw_impl &&) noexcept;
    eop_sw_impl &operator=(const eop_sw_impl &);
    eop_sw_impl &operator=(eop_sw_impl &&) noexcept;
    ~eop_sw_impl() override;

    // NOTE: this is used in taylor_decompose() to move-init an expression from this.
    [[nodiscard]] virtual expression ex_from_this() && = 0;

public:
    [[nodiscard]] virtual std::vector<expression> gradient() const = 0;

    [[nodiscard]] llvm::Value *llvm_evaluate(llvm_state &, const std::vector<llvm::Value *> &, llvm::Type *,
                                             llvm::Value *, bool) const;

    [[nodiscard]] taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) &&;

    [[nodiscard]] llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                           const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                           std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t,
                                                     bool) const;
};

// Base class for the implementation of the derivative of an EOP/SW quantity.
template <typename Data>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS eop_sw_p_impl : public eop_sw_impl_base<Data>
{
    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
    // NOTE: we never expect to use this class directly, thus we can mark special member functions as protected.
    eop_sw_p_impl();
    explicit eop_sw_p_impl(const std::string &, const std::string &, expression, Data);
    eop_sw_p_impl(const eop_sw_p_impl &);
    eop_sw_p_impl(eop_sw_p_impl &&) noexcept;
    eop_sw_p_impl &operator=(const eop_sw_p_impl &);
    eop_sw_p_impl &operator=(eop_sw_p_impl &&) noexcept;
    ~eop_sw_p_impl() override;

public:
    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] llvm::Value *llvm_evaluate(llvm_state &, const std::vector<llvm::Value *> &, llvm::Type *,
                                             llvm::Value *, bool) const;

    [[nodiscard]] llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                           const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                           std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t,
                                                     bool) const;
};

} // namespace model::detail

HEYOKA_END_NAMESPACE

// NOTE: this is a macro to be used when declaring the actual EOP/SW quantities in the expression system.
//
// It takes care of:
//
// - declaring classes for the EOP/SW quantity and its derivative,
// - declaring function wrappers for the construction of the EOP/SW quantity and its derivative,
// - defining the inline function objects intended for public use,
// - invoking the s11n key exporting machinery.
#define HEYOKA_MODEL_DECLARE_EOP_SW(name, data, kw_cfg, common_opts)                                                   \
    HEYOKA_BEGIN_NAMESPACE                                                                                             \
    namespace model                                                                                                    \
    {                                                                                                                  \
    namespace detail                                                                                                   \
    {                                                                                                                  \
    /* EOP/SW quantity. */                                                                                             \
    class HEYOKA_DLL_PUBLIC name##_impl final : public eop_sw_impl<data>                                               \
    {                                                                                                                  \
        friend class boost::serialization::access;                                                                     \
        void save(boost::archive::binary_oarchive &, unsigned) const;                                                  \
        void load(boost::archive::binary_iarchive &, unsigned);                                                        \
        BOOST_SERIALIZATION_SPLIT_MEMBER()                                                                             \
    private:                                                                                                           \
        [[nodiscard]] llvm::Function *get_llvm_eval_f(llvm_state &, llvm::Type *, std::uint32_t,                       \
                                                      const data &) const final;                                       \
        [[nodiscard]] expression ex_from_this() && final;                                                              \
                                                                                                                       \
    public:                                                                                                            \
        name##_impl();                                                                                                 \
        explicit name##_impl(expression, data);                                                                        \
        [[nodiscard]] std::vector<expression> gradient() const final;                                                  \
    };                                                                                                                 \
    [[nodiscard]] HEYOKA_DLL_PUBLIC expression name##_func_impl(expression, data);                                     \
    }                                                                                                                  \
    inline constexpr auto name = []<typename... KwArgs>                                                                \
        requires igor::validate<kw_cfg, KwArgs...>                                                                     \
    (KwArgs &&...kw_args) -> expression {                                                                              \
        return std::apply(detail::name##_func_impl, detail::common_opts(kw_args...));                                  \
    };                                                                                                                 \
    namespace detail                                                                                                   \
    {                                                                                                                  \
    /* Derivative of an EOP/SW quantity. */                                                                            \
    class HEYOKA_DLL_PUBLIC name##p_impl final : public eop_sw_p_impl<data>                                            \
    {                                                                                                                  \
        friend class boost::serialization::access;                                                                     \
        void save(boost::archive::binary_oarchive &, unsigned) const;                                                  \
        void load(boost::archive::binary_iarchive &, unsigned);                                                        \
        BOOST_SERIALIZATION_SPLIT_MEMBER()                                                                             \
    private:                                                                                                           \
        [[nodiscard]] llvm::Function *get_llvm_eval_f(llvm_state &, llvm::Type *, std::uint32_t,                       \
                                                      const data &) const final;                                       \
                                                                                                                       \
    public:                                                                                                            \
        name##p_impl();                                                                                                \
        explicit name##p_impl(expression, data);                                                                       \
    };                                                                                                                 \
    [[nodiscard]] HEYOKA_DLL_PUBLIC expression name##p_func_impl(expression, data);                                    \
    }                                                                                                                  \
    inline constexpr auto name##p = []<typename... KwArgs>                                                             \
        requires igor::validate<kw_cfg, KwArgs...>                                                                     \
    (KwArgs &&...kw_args) -> expression {                                                                              \
        return std::apply(detail::name##p_func_impl, detail::common_opts(kw_args...));                                 \
    };                                                                                                                 \
    }                                                                                                                  \
    HEYOKA_END_NAMESPACE                                                                                               \
    /* s11n macros. */                                                                                                 \
    HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::model::detail::name##_impl)                                                    \
    HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::model::detail::name##p_impl)

// NOTE: this is the definition counterpart of HEYOKA_MODEL_DECLARE_EOP_SW().
#define HEYOKA_MODEL_DEFINE_EOP_SW(descr, name, data_tp)                                                               \
    HEYOKA_BEGIN_NAMESPACE                                                                                             \
    namespace model                                                                                                    \
    {                                                                                                                  \
    namespace detail                                                                                                   \
    {                                                                                                                  \
    /* EOP/SW quantity. */                                                                                             \
    void name##_impl::save(boost::archive::binary_oarchive &oa, unsigned) const                                        \
    {                                                                                                                  \
        oa << boost::serialization::base_object<eop_sw_impl<data_tp>>(*this);                                          \
    }                                                                                                                  \
    void name##_impl::load(boost::archive::binary_iarchive &ia, unsigned)                                              \
    {                                                                                                                  \
        ia >> boost::serialization::base_object<eop_sw_impl<data_tp>>(*this);                                          \
    }                                                                                                                  \
    name##_impl::name##_impl() = default;                                                                              \
    name##_impl::name##_impl(expression time_expr, data_tp data)                                                       \
        : eop_sw_impl<data_tp>(#descr, #name, std::move(time_expr), std::move(data))                                   \
    {                                                                                                                  \
    }                                                                                                                  \
    llvm::Function *name##_impl::get_llvm_eval_f(llvm_state &s, llvm::Type *const fp_t,                                \
                                                 const std::uint32_t batch_size, const data_tp &data) const            \
    {                                                                                                                  \
        return llvm_get_##name##_##name##p_func(s, fp_t, batch_size, data);                                            \
    }                                                                                                                  \
    expression name##_impl::ex_from_this() &&                                                                          \
    {                                                                                                                  \
        return expression{func{std::move(*this)}};                                                                     \
    }                                                                                                                  \
    std::vector<expression> name##_impl::gradient() const                                                              \
    {                                                                                                                  \
        return {name##p(kw::time_expr = args()[0], kw::data_tp = checked_get_data())};                                 \
    }                                                                                                                  \
    expression name##_func_impl(expression time_expr, data_tp data)                                                    \
    {                                                                                                                  \
        return expression{func{name##_impl{std::move(time_expr), std::move(data)}}};                                   \
    }                                                                                                                  \
    /* Derivative of an EOP/SW quantity. */                                                                            \
    void name##p_impl::save(boost::archive::binary_oarchive &oa, unsigned) const                                       \
    {                                                                                                                  \
        oa << boost::serialization::base_object<eop_sw_p_impl<data_tp>>(*this);                                        \
    }                                                                                                                  \
    void name##p_impl::load(boost::archive::binary_iarchive &ia, unsigned)                                             \
    {                                                                                                                  \
        ia >> boost::serialization::base_object<eop_sw_p_impl<data_tp>>(*this);                                        \
    }                                                                                                                  \
    name##p_impl::name##p_impl() = default;                                                                            \
    name##p_impl::name##p_impl(expression time_expr, data_tp data)                                                     \
        : eop_sw_p_impl<data_tp>(#descr, #name "p", std::move(time_expr), std::move(data))                             \
    {                                                                                                                  \
    }                                                                                                                  \
    llvm::Function *name##p_impl::get_llvm_eval_f(llvm_state &s, llvm::Type *const fp_t,                               \
                                                  const std::uint32_t batch_size, const data_tp &data) const           \
    {                                                                                                                  \
        return llvm_get_##name##_##name##p_func(s, fp_t, batch_size, data);                                            \
    }                                                                                                                  \
    expression name##p_func_impl(expression time_expr, data_tp data)                                                   \
    {                                                                                                                  \
        return expression{func{name##p_impl{std::move(time_expr), std::move(data)}}};                                  \
    }                                                                                                                  \
    }                                                                                                                  \
    }                                                                                                                  \
    HEYOKA_END_NAMESPACE                                                                                               \
    /* s11n macros. */                                                                                                 \
    HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::model::detail::name##_impl)                                              \
    HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::model::detail::name##p_impl)

#endif
