// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_LLVM_FWD_HPP
#define HEYOKA_DETAIL_LLVM_FWD_HPP

namespace llvm
{

class Value;
class Function;
class Module;
class LLVMContext;
class Type;
class ArrayType;

// NOTE: IRBuilder is a template with default
// parameters, hence we declare the default parameters
// here and we use them in the definition of the
// ir_builder helper below.
class ConstantFolder;
class IRBuilderDefaultInserter;
template <typename, typename>
class IRBuilder;

} // namespace llvm

namespace heyoka
{

using ir_builder = llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>;

}

#endif
