//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/detail/RanluxppImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/random/data/RanluxppTypes.hh"

namespace celeritas
{
namespace detail
{

// Compute a + b and set overflow accordingly
CELER_FUNCTION RanluxppUInt add_overflow(RanluxppUInt a,
                                         RanluxppUInt b,
                                         unsigned int& overflow);

// Compute a + b and increment carry if there was an overflow
CELER_FUNCTION RanluxppUInt add_carry(RanluxppUInt a,
                                      RanluxppUInt b,
                                      unsigned int& carry);

// Compute a - b` and set overflow accordingly
CELER_FUNCTION RanluxppUInt sub_overflow(RanluxppUInt a,
                                         RanluxppUInt b,
                                         unsigned int& overflow);

// Compute a - b and increment `carry` if there was an overflow
CELER_FUNCTION RanluxppUInt sub_carry(RanluxppUInt a,
                                      RanluxppUInt b,
                                      unsigned int& carry);

// Update r = r - (t1 + t2) + (t3 + t2) * b ** 10
CELER_FUNCTION int64_t compute_remainder(Span<RanluxppUInt const, 9> upper,
                                         Span<RanluxppUInt, 9> r);

// Multiply two 576 bit numbers, stored as 9 numbers of 64 bits each
CELER_FUNCTION [[nodiscard]] RanluxppArray18
multiply_9x9(RanluxppArray9 const& in1, RanluxppArray9 const& in2);

// Compute a value congruent to mul modulo m less than 2 ** 576
CELER_FUNCTION [[nodiscard]] RanluxppArray9
compute_modulus(RanluxppArray18 const& mul);

// Combine multiply9x9 and mod_m with internal temporary storage
CELER_FUNCTION [[nodiscard]] RanluxppArray9
compute_mod_multiply(RanluxppArray9 const& factor1,
                     RanluxppArray9 const& factor2);

// Compute base to the n modulo m
CELER_FUNCTION [[nodiscard]] RanluxppArray9
compute_power_modulus(RanluxppArray9 base, RanluxppUInt n);

// Convert RANLUX numbers to an LCG state
CELER_FUNCTION [[nodiscard]] RanluxppArray9
to_lcg(RanluxppArray9 const& ranlux, unsigned int c);

// Convert an LCG state to RANLUX numbers
CELER_FUNCTION [[nodiscard]] RanluxppArray9
to_ranlux(RanluxppArray9 const& lcg, unsigned int& c_out);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
