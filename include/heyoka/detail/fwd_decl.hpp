// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_FWD_DECL_HPP
#define HEYOKA_DETAIL_FWD_DECL_HPP

namespace heyoka
{

// Fwd declaration of heyoka classes.
class expression;
class number;
class variable;
class binary_operator;
class function;

} // namespace heyoka

namespace llvm
{

// Fwd declaration of LLVM classes.
class Value;
class Module;
class ConstantFolder;
class IRBuilderDefaultInserter;

template <typename, typename>
class IRBuilder;

namespace legacy
{

class FunctionPassManager;
class PassManager;

} // namespace legacy

} // namespace llvm

#endif
