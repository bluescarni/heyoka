// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNC_HPP
#define HEYOKA_FUNC_HPP

#include <concepts>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/tanuki.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/func_args.hpp>
#include <heyoka/s11n.hpp>

// Current archive version is 2.
//
// Changelog:
// - version 2: re-implemented with tanuki.
BOOST_CLASS_VERSION(heyoka::func, 2)

// Current archive version is 2.
//
// Changelog:
// - version 2: introduction of the func_args class.
BOOST_CLASS_VERSION(heyoka::func_base, 2)

HEYOKA_BEGIN_NAMESPACE

class HEYOKA_DLL_PUBLIC func_base
{
    std::string m_name;
    func_args m_args;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    explicit func_base(std::string, std::vector<expression>, bool = false);
    explicit func_base(std::string, func_args::shared_args_t);
    explicit func_base(std::string, func_args);

    func_base(const func_base &);
    func_base(func_base &&) noexcept;

    func_base &operator=(const func_base &);
    func_base &operator=(func_base &&) noexcept;

    ~func_base();

    [[nodiscard]] const std::string &get_name() const noexcept;
    [[nodiscard]] const func_args &get_func_args() const noexcept;
    [[nodiscard]] const std::vector<expression> &args() const noexcept;
    [[nodiscard]] func_args::shared_args_t shared_args() const noexcept;

    // NOTE: these are supposed to be private, but there are issues making friends
    // with concept constraints on clang. Leave them public and undocumented for now.
    // NOTE: these are the only non-const functions in the interface, and they are supposed
    // to be used only in the implementation of the func::make_copy_with_new_args() function.
    void replace_args(std::vector<expression>);
    void replace_args(func_args::shared_args_t);
};

// UDF concept.
template <typename T>
concept is_udf = std::default_initializable<T> && std::copyable<T> && std::derived_from<T, func_base>;

namespace detail
{

// Default implementation of output streaming for func.
HEYOKA_DLL_PUBLIC void func_default_to_stream(std::ostringstream &, const func_base &);

template <typename T>
concept func_has_gradient = requires(const T &x) {
    { x.gradient() } -> std::same_as<std::vector<expression>>;
};

template <typename T>
concept func_has_taylor_decompose = requires(T &&x, taylor_dc_t &dc) {
    { std::move(x).taylor_decompose(dc) } -> std::same_as<taylor_dc_t::size_type>;
};

// Function interface implementation.
template <typename Base, typename Holder, typename T>
    requires is_udf<T>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS func_iface_impl : public Base {
    [[nodiscard]] const std::string &get_name() const noexcept final
    {
        // NOTE: make sure we are invoking the member function from func_base,
        // as in principle there could be a get_name() function in the derived
        // function class that hides it.
        return static_cast<const func_base &>(getval<Holder>(this)).get_name();
    }
    [[nodiscard]] const func_args &get_func_args() const noexcept final
    {
        // NOTE: make sure we are invoking the member function from func_base,
        // as in principle there could be a get_func_args() function in the derived
        // function class that hides it.
        return static_cast<const func_base &>(getval<Holder>(this)).get_func_args();
    }

    void to_stream(std::ostringstream &oss) const final
    {
        if constexpr (requires(const T &x) { static_cast<void>(x.to_stream(oss)); }) {
            static_cast<void>(getval<Holder>(this).to_stream(oss));
        } else {
            func_default_to_stream(oss, static_cast<const func_base &>(getval<Holder>(this)));
        }
    }

    [[nodiscard]] bool is_time_dependent() const final
    {
        if constexpr (requires(const T &x) { static_cast<bool>(x.is_time_dependent()); }) {
            return static_cast<bool>(getval<Holder>(this).is_time_dependent());
        } else {
            return false;
        }
    }

    [[nodiscard]] const std::vector<expression> &args() const noexcept final
    {
        // NOTE: make sure we are invoking the member function from func_base,
        // as in principle there could be an args() function in the derived
        // function class that hides it.
        return static_cast<const func_base &>(getval<Holder>(this)).args();
    }
    [[nodiscard]] func_args::shared_args_t shared_args() const noexcept final
    {
        // NOTE: make sure we are invoking the member function from func_base,
        // as in principle there could be a shared_args() function in the derived
        // function class that hides it.
        return static_cast<const func_base &>(getval<Holder>(this)).shared_args();
    }
    // NOTE: the implementation of the first overload must go in expression.hpp
    // as the definition of the expression class must be available.
    void replace_args(std::vector<expression>) final;
    void replace_args(func_args::shared_args_t new_args) final
    {
        static_cast<func_base &>(getval<Holder>(this)).replace_args(std::move(new_args));
    }

    // gradient.
    [[nodiscard]] bool has_gradient() const final
    {
        return func_has_gradient<T>;
    }
    [[nodiscard]] std::vector<expression> gradient() const final
    {
        if constexpr (func_has_gradient<T>) {
            return getval<Holder>(this).gradient();
        }

        // LCOV_EXCL_START
        assert(false);
        throw;
        // LCOV_EXCL_STOP
    }

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                         llvm::Value *par_ptr, llvm::Value *time_ptr, llvm::Value *stride,
                                         std::uint32_t batch_size, bool high_accuracy) const final
    {
        if constexpr (requires(const T &x) {
                          {
                              x.llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size, high_accuracy)
                          } -> std::same_as<llvm::Value *>;
                      }) {
            return getval<Holder>(this).llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size,
                                                  high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("llvm_eval() is not implemented for the function '{}'", get_name()));
        }
    }

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                                   bool high_accuracy) const final
    {
        if constexpr (requires(const T &x) {
                          { x.llvm_c_eval_func(s, fp_t, batch_size, high_accuracy) } -> std::same_as<llvm::Function *>;
                      }) {
            return getval<Holder>(this).llvm_c_eval_func(s, fp_t, batch_size, high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("llvm_c_eval_func() is not implemented for the function '{}'", get_name()));
        }
    }

    // Taylor.
    taylor_dc_t::size_type taylor_decompose(taylor_dc_t &dc) && final
    {
        if constexpr (func_has_taylor_decompose<T>) {
            return std::move(getval<Holder>(this)).taylor_decompose(dc);
        }

        // LCOV_EXCL_START
        assert(false);
        throw;
        // LCOV_EXCL_STOP
    }
    [[nodiscard]] bool has_taylor_decompose() const final
    {
        return func_has_taylor_decompose<T>;
    }
    llvm::Value *taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size,
                             bool high_accuracy) const final
    {
        if constexpr (requires(const T &x) {
                          {
                              x.taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size,
                                            high_accuracy)
                          } -> std::same_as<llvm::Value *>;
                      }) {
            return getval<Holder>(this).taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx,
                                                    batch_size, high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("Taylor diff is not implemented for the function '{}'", get_name()));
        }
    }
    llvm::Function *taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars, std::uint32_t batch_size,
                                       bool high_accuracy) const final
    {
        if constexpr (requires(const T &x) {
                          {
                              x.taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy)
                          } -> std::same_as<llvm::Function *>;
                      }) {
            return getval<Holder>(this).taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy);
        } else {
            throw not_implemented_error(
                fmt::format("Taylor diff in compact mode is not implemented for the function '{}'", get_name()));
        }
    }
};

// The function interface.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions,cppcoreguidelines-virtual-class-destructor)
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS func_iface {
    [[nodiscard]] virtual const std::string &get_name() const noexcept = 0;
    [[nodiscard]] virtual const func_args &get_func_args() const noexcept = 0;

    virtual void to_stream(std::ostringstream &) const = 0;

    [[nodiscard]] virtual bool is_time_dependent() const = 0;

    [[nodiscard]] virtual const std::vector<expression> &args() const noexcept = 0;
    [[nodiscard]] virtual func_args::shared_args_t shared_args() const noexcept = 0;
    virtual void replace_args(std::vector<expression>) = 0;
    virtual void replace_args(func_args::shared_args_t) = 0;

    [[nodiscard]] virtual bool has_gradient() const = 0;
    [[nodiscard]] virtual std::vector<expression> gradient() const = 0;

    [[nodiscard]] virtual llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &,
                                                 llvm::Value *, llvm::Value *, llvm::Value *, std::uint32_t, bool) const
        = 0;

    [[nodiscard]] virtual llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const = 0;

    // NOTE: this is the last remaining trace of mutability
    // related to decomposition. Note, however, that this is never
    // exposed in the public API.
    virtual taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) && = 0;
    [[nodiscard]] virtual bool has_taylor_decompose() const = 0;

    virtual llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                     const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                                     std::uint32_t, std::uint32_t, std::uint32_t, bool) const
        = 0;

    virtual llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const
        = 0;

    template <typename Base, typename Holder, typename T>
    using impl = func_iface_impl<Base, Holder, T>;
};

// The udf used in the default construction of a func.
struct HEYOKA_DLL_PUBLIC null_func : func_base {
    null_func();

private:
    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace detail

HEYOKA_DLL_PUBLIC void swap(func &, func &) noexcept;

HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &) noexcept;
HEYOKA_DLL_PUBLIC bool operator!=(const func &, const func &) noexcept;
HEYOKA_DLL_PUBLIC bool operator<(const func &, const func &);

namespace detail
{

taylor_dc_t::size_type func_taylor_decompose_impl(func &&, taylor_dc_t &);

} // namespace detail

class HEYOKA_DLL_PUBLIC func
{
    friend HEYOKA_DLL_PUBLIC void swap(func &, func &) noexcept;
    friend HEYOKA_DLL_PUBLIC bool operator==(const func &, const func &) noexcept;
    friend HEYOKA_DLL_PUBLIC bool operator<(const func &, const func &);
    friend taylor_dc_t::size_type detail::func_taylor_decompose_impl(func &&, taylor_dc_t &);

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#endif

    using func_wrap_t = tanuki::wrap<detail::func_iface,
                                     tanuki::config<detail::null_func>{.semantics = tanuki::wrap_semantics::reference}>;

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

    func_wrap_t m_func;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    func();
    template <typename T>
        requires(!std::same_as<func, std::remove_cvref_t<T>>) && std::constructible_from<func_wrap_t, T &&>
    explicit func(T &&f) : m_func(std::forward<T>(f))
    {
    }
    func(const func &) noexcept;
    func(func &&) noexcept;
    func &operator=(const func &) noexcept;
    func &operator=(func &&) noexcept;
    ~func();

    [[nodiscard]] const void *get_ptr() const noexcept;

    [[nodiscard]] const std::vector<expression> &args() const noexcept;
    [[nodiscard]] func_args::shared_args_t shared_args() const noexcept;

    // NOTE: these create a new func containing a copy of the inner object in which the original function arguments have
    // been replaced by the provided set of arguments.
    [[nodiscard]] func make_copy_with_new_args(std::vector<expression>) const;
    [[nodiscard]] func make_copy_with_new_args(func_args::shared_args_t) const;

    template <typename T>
    [[nodiscard]] const T *extract() const noexcept
    {
        return value_ptr<T>(m_func);
    }

    [[nodiscard]] bool is_time_dependent() const;

    [[nodiscard]] const std::string &get_name() const noexcept;

    [[nodiscard]] const func_args &get_func_args() const noexcept;

    [[nodiscard]] std::type_index get_type_index() const;

    void to_stream(std::ostringstream &) const;

    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint32_t, bool) const;
    llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const;
};

namespace detail
{

[[nodiscard]] llvm::Value *cfunc_nc_param_codegen(llvm_state &, const param &, std::uint32_t, llvm::Type *,
                                                  llvm::Value *, llvm::Value *);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Value *
llvm_eval_helper(const std::function<llvm::Value *(const std::vector<llvm::Value *> &, bool)> &, const func_base &,
                 llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                 std::uint32_t, bool);

std::pair<std::string, std::vector<llvm::Type *>> llvm_c_eval_func_name_args(llvm::LLVMContext &, llvm::Type *,
                                                                             const std::string &, std::uint32_t,
                                                                             const std::vector<expression> &);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Function *
llvm_c_eval_func_helper(const std::string &,
                        const std::function<llvm::Value *(const std::vector<llvm::Value *> &, bool)> &,
                        const func_base &, llvm_state &, llvm::Type *, std::uint32_t, bool);

} // namespace detail

HEYOKA_END_NAMESPACE

// Macros for the registration of s11n for concrete functions.
// NOTE: by default, we build a custom name and pass it to TANUKI_S11N_WRAP_EXPORT_KEY2.
// This allows us to reduce the size of the final guid wrt to what TANUKI_S11N_WRAP_EXPORT_KEY
// would synthesise, and thus to ameliorate the "class name too long" issue.
#define HEYOKA_S11N_FUNC_EXPORT_KEY(f) TANUKI_S11N_WRAP_EXPORT_KEY2(f, "heyoka::func@" #f, heyoka::detail::func_iface)

#define HEYOKA_S11N_FUNC_EXPORT_KEY2(f, gid) TANUKI_S11N_WRAP_EXPORT_KEY2(f, gid, heyoka::detail::func_iface)

#define HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(f) TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(f, heyoka::detail::func_iface)

#define HEYOKA_S11N_FUNC_EXPORT(f)                                                                                     \
    HEYOKA_S11N_FUNC_EXPORT_KEY(f)                                                                                     \
    HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(f)

#define HEYOKA_S11N_FUNC_EXPORT2(f, gid)                                                                               \
    HEYOKA_S11N_FUNC_EXPORT_KEY2(f, gid)                                                                               \
    HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(f)

// Export the key for null_func.
HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::null_func)

#endif
