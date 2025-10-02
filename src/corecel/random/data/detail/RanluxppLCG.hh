//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/detail/RanluxppLCG.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/random/data/RanluxppTypes.hh"

#include "RanluxppHelpers.hh"

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Convert RANLUX numbers to an LCG state
 *
 * Computes \f$ m = 2^{576} - 2^{240} + 1 \f$.
 *
 * \param[in]  ranlux  The RANLUX numbers as 576 bits
 * \param[in]  c       The carry bit of the RANLUX state
 * \param[out] lcg     The 576 bits of the LCG state, smaller than m
 */
CELER_FUNCTION inline void
toLCG(RanluxppArray9 const& ranlux, unsigned int c, RanluxppArray9& lcg)
{
    unsigned int carry = 0;

    // Subtract the final 240 bits.
    for (int i : celeritas::range(9))
    {
        RanluxppUInt ranlux_i = ranlux[i];
        RanluxppUInt lcg_i = subOverflow(ranlux_i, carry, carry);

        RanluxppUInt bits = 0;
        if (i < 4)
        {
            bits += ranlux[i + 5] >> 16;
            if (i < 3)
            {
                bits += ranlux[i + 6] << 48;
            }
        }
        lcg_i = subCarry(lcg_i, bits, carry);
        lcg[i] = lcg_i;
    }

    // Add and propagate the carry bit.
    for (RanluxppUInt& lcg_val : lcg)
    {
        lcg_val = addOverflow(lcg_val, c, c);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert an LCG state to RANLUX numbers
 *
 * \f$ m = 2^{576} - 2^{240} + 1 \f$
 *
 * \param[in]  lcg     The 576 bits of the LCG state, must be smaller than m
 * \param[out] ranlux  The RANLUX numbers as 576 bits
 * \param[out] c       The carry bit of the RANLUX state
 */
CELER_FUNCTION inline void
toRanlux(RanluxppArray9 const& lcg, RanluxppArray9& ranlux, unsigned int& c_out)
{
    RanluxppArray9 r = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int64_t c = computeR(celeritas::make_span(lcg), celeritas::make_span(r));

    // ranlux = t1 + t2 + c
    unsigned int carry = 0;
    for (int i : celeritas::range(9))
    {
        RanluxppUInt in_i = lcg[i];
        RanluxppUInt tmp_i = addOverflow(in_i, carry, carry);

        RanluxppUInt bits = 0;
        if (i < 4)
        {
            bits += lcg[i + 5] >> 16;
            if (i < 3)
            {
                bits += lcg[i + 6] << 48;
            }
        }
        tmp_i = addCarry(tmp_i, bits, carry);
        ranlux[i] = tmp_i;
    }

    // If c = -1, we need to add it to all components.
    int64_t c1 = c >> 1;
    ranlux[0] = addOverflow(ranlux[0], c, carry);
    for (int i : celeritas::range(1, 9))
    {
        RanluxppUInt ranlux_i = ranlux[i];
        ranlux_i = addOverflow(ranlux_i, carry, carry);
        ranlux_i = addCarry(ranlux_i, c1, carry);
    }

    c_out = carry;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
