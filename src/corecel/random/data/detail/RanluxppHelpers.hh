//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/detail/RanluxppHelpers.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <limits>

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/random/data/RanluxppTypes.hh"

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Compute `a + b` and set `overflow` accordingly.
 *
 * \param[in]  a         The first operand of the sum
 * \param[in]  b         The second operand of the sum
 * \param[out] overflow  The amount of (potential) overflow
 *
 * \return The result of the sum
 */
CELER_FUNCTION inline RanluxppUInt
addOverflow(RanluxppUInt a, RanluxppUInt b, RanluxppUInt& overflow)
{
    RanluxppUInt add = a + b;
    overflow = (add < a);
    return add;
}

//---------------------------------------------------------------------------//
/*!
 * Compute `a + b` and increment `carry` if there was an overflow
 *
 * \param[in]  a      The first operand of the sum
 * \param[in]  b      The second operand of the sum
 * \param[out] carry  Maintains sum of all overflows
 *
 * \return The result of the sum
 */
CELER_FUNCTION inline RanluxppUInt
addCarry(RanluxppUInt a, RanluxppUInt b, RanluxppUInt& carry)
{
    RanluxppUInt overflow;
    RanluxppUInt add = addOverflow(a, b, overflow);

    // Do NOT branch on overflow to avoid jumping code, just add 0 if there was
    // no overflow.
    carry += overflow;
    return add;
}

//---------------------------------------------------------------------------//
/*!
 * Compute `a - b` and set `overflow` accordingly
 *
 * \param[in] a          The first operand of the subtraction
 * \param[in] b          The second operand of the subtraction
 * \param[out] overflow  Stores the amount of any overflow
 *
 * \return The result of the subtraction
 */
CELER_FUNCTION inline RanluxppUInt
subOverflow(RanluxppUInt a, RanluxppUInt b, RanluxppUInt& overflow)
{
    RanluxppUInt sub = a - b;
    overflow = (sub > a);
    return sub;
}

//---------------------------------------------------------------------------//
/*!
 * Compute `a - b` and increment `carry` if there was an overflow
 *
 * \param[in] a       The first operand of the subtraction
 * \param[in] b       The second operand of the subtraction
 * \param[out] carry  Maintains sum of all overflows
 *
 * \return The result of the subtraction
 */
CELER_FUNCTION static inline RanluxppUInt
subCarry(RanluxppUInt a, RanluxppUInt b, RanluxppUInt& carry)
{
    RanluxppUInt overflow;
    RanluxppUInt sub = subOverflow(a, b, overflow);

    // Do NOT branch on overflow to avoid jumping code, just add 0 if there was
    // no overflow.
    carry += overflow;
    return sub;
}

//---------------------------------------------------------------------------//
/*!
 * Update r = r - (t1 + t2) + (t3 + t2) * b ** 10
 *
 * This function also yields cbar = floor(r / m) as its return value (int64_t
 * because the value can be -1). With an initial value of r = t0, this can
 * be used for computing the remainder after division by m (see the function
 * mod_m in mulmod.h). The function to_ranlux passes r = 0 and uses only the
 * return value to obtain the decimal expansion after divison by m.
 */
CELER_FUNCTION inline RanluxppUInt
computeR(Span<RanluxppUInt const, 9> upper, Span<RanluxppUInt, 9> r)
{
    // Subtract t1 (24 * 24 = 576 bits)
    RanluxppUInt carry = 0;
    for (int i : celeritas::range(9))
    {
        RanluxppUInt r_i = r[i];
        r_i = subOverflow(r_i, carry, carry);

        RanluxppUInt t1_i = upper[i];
        r_i = subCarry(r_i, t1_i, carry);
        r[i] = r_i;
    }
    RanluxppUInt c = -(carry);

    // Subtract t2 (only 240 bits, so need to extend)
    carry = 0;
    for (int i : celeritas::range(9))
    {
        RanluxppUInt r_i = r[i];
        r_i = subOverflow(r_i, carry, carry);

        RanluxppUInt t2_bits = 0;
        if (i < 4)
        {
            t2_bits += upper[i + 5] >> 16;
            if (i < 3)
            {
                t2_bits += upper[i + 6] << 48;
            }
        }
        r_i = subCarry(r_i, t2_bits, carry);
        r[i] = r_i;
    }
    c -= carry;

    // r += (t3 + t2) * 2 ** 240
    carry = 0;
    {
        RanluxppUInt r_3 = r[3];
        // 16 upper bits
        RanluxppUInt t2_bits = (upper[5] >> 16) << 48;
        RanluxppUInt t3_bits = (upper[0] << 48);

        r_3 = addCarry(r_3, t2_bits, carry);
        r_3 = addCarry(r_3, t3_bits, carry);

        r[3] = r_3;
    }
    for (int i : celeritas::range(3))
    {
        RanluxppUInt r_i = r[i + 4];
        r_i = addOverflow(r_i, carry, carry);

        RanluxppUInt t2_bits = (upper[5 + i] >> 32) + (upper[6 + i] << 32);
        RanluxppUInt t3_bits = (upper[i] >> 16) + (upper[1 + i] << 48);

        r_i = addCarry(r_i, t2_bits, carry);
        r_i = addCarry(r_i, t3_bits, carry);

        r[i + 4] = r_i;
    }
    {
        RanluxppUInt r_7 = r[7];
        r_7 = addOverflow(r_7, carry, carry);

        RanluxppUInt t2_bits = (upper[8] >> 32);
        RanluxppUInt t3_bits = (upper[3] >> 16) + (upper[4] << 48);

        r_7 = addCarry(r_7, t2_bits, carry);
        r_7 = addCarry(r_7, t3_bits, carry);

        r[7] = r_7;
    }
    {
        RanluxppUInt r_8 = r[8];
        r_8 = addOverflow(r_8, carry, carry);

        RanluxppUInt t3_bits = (upper[4] >> 16) + (upper[5] << 48);

        r_8 = addCarry(r_8, t3_bits, carry);

        r[8] = r_8;
    }
    c += carry;

    // c = floor(r / 2 ** 576) has been computed along the way via the carry
    // flags. Now if c = 0 and the value currently stored in r is greater or
    // equal to m, we need cbar = 1 and subtract m, otherwise cbar = c. The
    // value currently in r is greater or equal to m, if and only if one of
    // the last 240 bits is set and the upper bits are all set.
    bool greater_m = r[0] | r[1] | r[2] | (r[3] & 0x0000ffffffffffff);
    greater_m &= (r[3] >> 48) == 0xffff;
    for (int i : celeritas::range(4, 9))
    {
        greater_m &= (r[i] == celeritas::numeric_limits<RanluxppUInt>::max());
    }
    return c + (c == 0 && greater_m);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
