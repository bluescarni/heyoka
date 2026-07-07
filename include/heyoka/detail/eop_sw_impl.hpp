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

namespace heyoka::model::detail
{

template <typename DataTable>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS eop_sw_impl : public func_base
{
    std::string m_name;
    // NOTE: we wrap the data table into an optional because we do not want to pay the cost of storing the full data or
    // a default-constructed object, which is anyway only used during serialisation.
    std::optional<DataTable> m_data;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
    [[nodiscard]] virtual llvm::Function *get_llvm_eval(llvm_state &, llvm::Type *, std::uint32_t,
                                                        const DataTable &) const = 0;

public:
    eop_sw_impl();
    explicit eop_sw_impl(std::string, expression, DataTable);

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
