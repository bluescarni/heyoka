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
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

// This header declares common base classes for the implementation of EOP/SW quantities and their derivatives in the
// expression system.

namespace heyoka::model::detail
{

template <typename DataTable>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS eop_sw_impl : public func_base
{
    // The descriptor (used to distinguish between EOP and SW).
    std::string m_descr;
    // The name of the EOP/SW quantity.
    std::string m_name;
    // The data table.
    //
    // NOTE: this is wrapped into an optional so that default construction (which is necessary only for serialisation
    // purposes) is cheap and does not require the actual storage of any EOP/SW data. In normal usage scenarios, we are
    // always expecting the optional to be non-empty.
    std::optional<DataTable> m_data;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    [[nodiscard]] const DataTable &checked_get_data() const;
    [[nodiscard]] llvm::Value *llvm_eval_helper(llvm_state &, llvm::Value *, llvm::Type *, std::uint32_t) const;

protected:
    // NOTE: this is used to generate/fetch the LLVM function that evaluates at the same time the EOP/SW quantity and
    // its derivative.
    [[nodiscard]] virtual llvm::Function *get_llvm_eval_f(llvm_state &, llvm::Type *, std::uint32_t,
                                                          const DataTable &) const = 0;
    // NOTE: this is used in taylor_decompose() to move-init an expression from this.
    [[nodiscard]] virtual expression ex_from_this() && = 0;

    // NOTE: we never expect to use this class directly, thus we can mark special member functions as protected.
    eop_sw_impl();
    explicit eop_sw_impl(std::string, std::string, expression, DataTable);
    eop_sw_impl(const eop_sw_impl &);
    eop_sw_impl(eop_sw_impl &&) noexcept;
    eop_sw_impl &operator=(const eop_sw_impl &);
    eop_sw_impl &operator=(eop_sw_impl &&) noexcept;
    // NOTE: a protected destructor allows us to satisfy:
    //
    // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#rc-dtor-virtual
    ~eop_sw_impl();

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

} // namespace heyoka::model::detail

#endif
