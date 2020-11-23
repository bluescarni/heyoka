// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

class ConstantFolder;
class IRBuilderDefaultInserter;
template <typename, typename>
class IRBuilder;

class IRBuilderBase;

} // namespace llvm

#endif
