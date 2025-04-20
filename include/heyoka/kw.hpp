// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_KW_HPP
#define HEYOKA_KW_HPP

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>

// NOTE: all keyword arguments are gathered
// in this file in order to make it easier to
// prevent name collisions.

HEYOKA_BEGIN_NAMESPACE

namespace kw
{

// llvm_state/llvm_multi_state.
IGOR_MAKE_NAMED_ARGUMENT(mname);
IGOR_MAKE_NAMED_ARGUMENT(opt_level);
IGOR_MAKE_NAMED_ARGUMENT(fast_math);
// NOTE: this flag is used to force the use of 512-bit AVX512
// registers (if the CPU supports them). At the time of this writing,
// LLVM defaults to 256-bit registers due to CPU downclocking issues
// which can lead to performance degradation. Hopefully we
// can get rid of this in the future when AVX512 implementations improve
// and LLVM learns to discriminate good and bad implementations.
IGOR_MAKE_NAMED_ARGUMENT(force_avx512);
IGOR_MAKE_NAMED_ARGUMENT(slp_vectorize);
IGOR_MAKE_NAMED_ARGUMENT(code_model);
IGOR_MAKE_NAMED_ARGUMENT(parjit);

// cfunc API.
IGOR_MAKE_NAMED_ARGUMENT(batch_size);
IGOR_MAKE_NAMED_ARGUMENT(strided);
IGOR_MAKE_NAMED_ARGUMENT(check_prec);

// taylor_adaptive and friends.
IGOR_MAKE_NAMED_ARGUMENT(tol);
IGOR_MAKE_NAMED_ARGUMENT(t_events);
IGOR_MAKE_NAMED_ARGUMENT(nt_events);
// NOTE: these are used for constructing events.
IGOR_MAKE_NAMED_ARGUMENT(callback);
IGOR_MAKE_NAMED_ARGUMENT(cooldown);
IGOR_MAKE_NAMED_ARGUMENT(direction);
// NOTE: these are used in the
// propagate_*() functions.
IGOR_MAKE_NAMED_ARGUMENT(max_steps);
IGOR_MAKE_NAMED_ARGUMENT(max_delta_t);
IGOR_MAKE_NAMED_ARGUMENT(write_tc);
IGOR_MAKE_NAMED_ARGUMENT(c_output);

// Diff tensors API.
IGOR_MAKE_NAMED_ARGUMENT(diff_order);

// Used in several APIs.
IGOR_MAKE_NAMED_ARGUMENT(time);
IGOR_MAKE_NAMED_ARGUMENT(time_expr);
IGOR_MAKE_NAMED_ARGUMENT(prec);
IGOR_MAKE_NAMED_ARGUMENT(compact_mode);
IGOR_MAKE_NAMED_ARGUMENT(high_accuracy);
IGOR_MAKE_NAMED_ARGUMENT(parallel_mode);
IGOR_MAKE_NAMED_ARGUMENT(pars);

// ffnn model.
IGOR_MAKE_NAMED_ARGUMENT(inputs);
IGOR_MAKE_NAMED_ARGUMENT(nn_hidden);
IGOR_MAKE_NAMED_ARGUMENT(n_out);
IGOR_MAKE_NAMED_ARGUMENT(activations);
IGOR_MAKE_NAMED_ARGUMENT(nn_wb);

// cartesian2geodetic
IGOR_MAKE_NAMED_ARGUMENT(ecc2);
IGOR_MAKE_NAMED_ARGUMENT(R_eq);
IGOR_MAKE_NAMED_ARGUMENT(n_iters);

// thermonets
IGOR_MAKE_NAMED_ARGUMENT(geodetic);
IGOR_MAKE_NAMED_ARGUMENT(f107a);
IGOR_MAKE_NAMED_ARGUMENT(f107);
IGOR_MAKE_NAMED_ARGUMENT(ap);
IGOR_MAKE_NAMED_ARGUMENT(s107);
IGOR_MAKE_NAMED_ARGUMENT(s107a);
IGOR_MAKE_NAMED_ARGUMENT(m107);
IGOR_MAKE_NAMED_ARGUMENT(m107a);
IGOR_MAKE_NAMED_ARGUMENT(y107);
IGOR_MAKE_NAMED_ARGUMENT(y107a);
IGOR_MAKE_NAMED_ARGUMENT(dDstdT);

// Fixed centres model.
IGOR_MAKE_NAMED_ARGUMENT(positions);

// Pendulum model.
IGOR_MAKE_NAMED_ARGUMENT(gconst);
IGOR_MAKE_NAMED_ARGUMENT(length);

// Used in multiple models.
IGOR_MAKE_NAMED_ARGUMENT(masses);
IGOR_MAKE_NAMED_ARGUMENT(omega);
IGOR_MAKE_NAMED_ARGUMENT(Gconst);
IGOR_MAKE_NAMED_ARGUMENT(mu);
IGOR_MAKE_NAMED_ARGUMENT(thresh);
IGOR_MAKE_NAMED_ARGUMENT(eop_data);
IGOR_MAKE_NAMED_ARGUMENT(sw_data);
IGOR_MAKE_NAMED_ARGUMENT(a);

} // namespace kw

HEYOKA_END_NAMESPACE

#endif
