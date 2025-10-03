//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/detail/RanluxppHelpers.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/Macros.hh"
#include "corecel/random/data/RanluxppTypes.hh"

#include "RanluxppHelpers.hh"

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Multiply two 576 bit numbers, stored as 9 numbers of 64 bits each
 *
 * \param[in]  in1  first factor as 9 numbers of 64 bits each
 * \param[in]  in2  second factor as 9 numbers of 64 bits each
 * \param[out] out  result with 18 numbers of 64 bits each
 *
 * \todo We have disabled the branch using 128-bit integers. This causes
 *       about a 25% performance hit on optimized builds.  A fast portable
 *       solution would be nice.
 */
CELER_FUNCTION inline void multiply9x9(RanluxppArray9 const& in1,
                                       RanluxppArray9 const& in2,
                                       RanluxppArray18& out)
{
    RanluxppUInt next = 0;
    unsigned int nextCarry = 0;

#if defined(__clang__) || defined(__INTEL_COMPILER) || defined(__CUDA_ARCH__)
#    pragma unroll
#elif defined(__GNUC__) && __GNUC__ >= 8
// This pragma was introduced in GCC version 8.
#    pragma GCC unroll 18
#endif
    for (int i : celeritas::range(18))
    {
        RanluxppUInt current = next;
        unsigned int carry = nextCarry;

        next = 0;
        nextCarry = 0;

#if defined(__clang__) || defined(__INTEL_COMPILER) || defined(__CUDA_ARCH__)
#    pragma unroll
#elif defined(__GNUC__) && __GNUC__ >= 8
// This pragma was introduced in GCC version 8.
#    pragma GCC unroll 9
#endif
        for (int j : celeritas::range(9))
        {
            int k = i - j;
            if (k < 0 || k >= 9)
            {
                continue;
            }
            RanluxppUInt fac1 = in1[j];
            RanluxppUInt fac2 = in2[k];
#if defined(__CUDA_ARCH__)
            // In principle, we could use the "portable" code path with
            // __int128 starting from CUDA 11.5, but the math intrinsic is
            // equally easy to write and should work in older versions of CUDA.
            RanluxppUInt lower = fac1 * fac2;
            RanluxppUInt upper = __umul64hi(fac1, fac2);
#elif defined(__SIZEOF_INT128__)
#    ifdef __GNUC__
            // This block of code requires 128-bit unsigned integers, which is
            // non-standard.  If using GCC, we temporarily disable
            // "-Wpedantic".
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wpedantic"
            using uint128 = unsigned __int128;
#        pragma GCC diagnostic pop
#    else
            using uint128 = unsigned __int128;
#    endif
            uint128 prod = fac1;
            prod = prod * fac2;

            RanluxppUInt upper = prod >> 64;
            RanluxppUInt lower = static_cast<RanluxppUInt>(prod);
#else
            RanluxppUInt upper1 = fac1 >> 32;
            RanluxppUInt lower1 = static_cast<uint32_t>(fac1);

            RanluxppUInt upper2 = fac2 >> 32;
            RanluxppUInt lower2 = static_cast<uint32_t>(fac2);

            // Multiply 32-bit parts, each product has a maximum value of
            // (2 ** 32 - 1) ** 2 = 2 ** 64 - 2 * 2 ** 32 + 1.
            RanluxppUInt upper = upper1 * upper2;
            RanluxppUInt middle1 = upper1 * lower2;
            RanluxppUInt middle2 = lower1 * upper2;
            RanluxppUInt lower = lower1 * lower2;

            // When adding the two products, the maximum value for middle is
            // 2 * 2 ** 64 - 4 * 2 ** 32 + 2, which exceeds a uint64_t.
            unsigned int overflow;
            RanluxppUInt middle = addOverflow(middle1, middle2, overflow);
            // Handling the overflow by a multiplication with 0 or 1 is cheaper
            // than branching with an if statement, which the compiler does not
            // optimize to this equivalent code. Note that we could do entirely
            // without this overflow handling when summing up the intermediate
            // products differently as described in the following SO answer:
            //    https://stackoverflow.com/a/51587262
            // However, this approach takes at least the same amount of
            // thinking why a) the code gives the same results without b)
            // overflowing due to the mixture of 32 bit arithmetic. Moreover,
            // my tests show that the scheme implemented here is actually
            // slightly more performant.
            RanluxppUInt overflow_add = overflow
                                        * (static_cast<RanluxppUInt>(1) << 32);
            // This addition can never overflow because the maximum value of
            // upper is 2 ** 64 - 2 * 2 ** 32 + 1 (see above). When now adding
            // another 2 ** 32, the result is 2 ** 64 - 2 ** 32 + 1 and still
            // smaller than the maximum 2 ** 64 - 1 that can be stored in a
            // uint64_t.
            upper += overflow_add;

            RanluxppUInt middle_upper = middle >> 32;
            RanluxppUInt middle_lower = middle << 32;

            lower = addOverflow(lower, middle_lower, overflow);
            upper += overflow;

            // This still can't overflow since the maximum of middle_upper is
            //  - 2 ** 32 - 4 if there was an overflow for middle above,
            //  bringing
            //    the maximum value of upper to 2 ** 64 - 2.
            //  - otherwise upper still has the initial maximum value given
            //  above
            //    and the addition of a value smaller than 2 ** 32 brings it to
            //    a maximum value of 2 ** 64 - 2 ** 32 + 2.
            // (Both cases include the increment to handle the overflow in
            // lower.)
            //
            // All the reasoning makes perfect sense given that the product of
            // two 64 bit numbers is smaller than or equal to
            //     (2 ** 64 - 1) ** 2 = 2 ** 128 - 2 * 2 ** 64 + 1
            // with the upper bits matching the 2 ** 64 - 2 of the first case.
            upper += middle_upper;
#endif

            // Add to current, remember carry.
            current = addCarry(current, lower, carry);

            // Add to next, remember nextCarry.
            next = addCarry(next, upper, nextCarry);
        }

        next = addCarry(next, carry, nextCarry);
        out[i] = current;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Compute a value congruent to mul modulo m less than 2 ** 576
 *
 * This computes \f$ m = 2^{576} - 2^{240} + 1 \f$. The result in
 * \p out is guaranteed to be smaller than the modulus.
 *
 * \param[in] mul product from multiply9x9 with 18 numbers of 64 bits each
 * \param[out] out result with 9 numbers of 64 bits each
 */
CELER_FUNCTION inline void modM(RanluxppArray18 const& mul, RanluxppArray9& out)
{
    RanluxppArray9 r = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // Assign r = t0
    std::copy_n(mul.begin(), 9, r.begin());

    // Make a subspan of the last 9 elements of mul
    auto mul_end = celeritas::make_span(mul).subspan<9, 9>();
    CELER_ASSERT(mul_end.size() == 9);
    CELER_ASSERT(std::equal(mul_end.begin(), mul_end.end(), mul.begin() + 9));

    int64_t c = computeR(mul_end, celeritas::make_span(r));

    // To update r = r - c * m, it suffices to know c * (-2 ** 240 + 1)
    // because the 2 ** 576 will cancel out. Also note that c may be zero, but
    // the operation is still performed to avoid branching.

    // c * (-2 ** 240 + 1) in 576 bits looks as follows, depending on c:
    //  - if c = 0, the number is zero.
    //  - if c = 1: bits 576 to 240 are set,
    //              bits 239 to 1 are zero, and
    //              the last one is set
    //  - if c = -1, which corresponds to all bits set (signed int64_t):
    //              bits 576 to 240 are zero and the rest is set.
    // Note that all bits except the last are exactly complimentary (unless c =
    // 0) and the last byte is conveniently represented by c already. Now
    // construct the three bit patterns from c, their names correspond to the
    // assembly implementation by Alexei Sibidanov.

    // c = 0 -> t0 = 0; c = 1 -> t0 = 0; c = -1 -> all bits set (sign
    // extension) (The assembly implementation shifts by 63, which gives the
    // same result.)
    int64_t t0 = c >> 1;

    // c = 0 -> t2 = 0; c = 1 -> upper 16 bits set; c = -1 -> lower 48 bits set
    int64_t t2 = t0 - (c << 48);

    // c = 0 -> t1 = 0; c = 1 -> all bits set; c = -1 -> t1 = 0
    // (The assembly implementation shifts by 63, which gives the same result.)
    int64_t t1 = t2 >> 48;

    unsigned int carry = 0;
    {
        RanluxppUInt r_0 = r[0];

        RanluxppUInt out_0 = subCarry(r_0, c, carry);
        out[0] = out_0;
    }
    for (int i : celeritas::range(1, 3))
    {
        RanluxppUInt r_i = r[i];
        r_i = subOverflow(r_i, carry, carry);

        RanluxppUInt out_i = subCarry(r_i, t0, carry);
        out[i] = out_i;
    }
    {
        RanluxppUInt r_3 = r[3];
        r_3 = subOverflow(r_3, carry, carry);

        RanluxppUInt out_3 = subCarry(r_3, t2, carry);
        out[3] = out_3;
    }
    for (int i : celeritas::range(4, 9))
    {
        RanluxppUInt r_i = r[i];
        r_i = subOverflow(r_i, carry, carry);

        RanluxppUInt out_i = subCarry(r_i, t1, carry);
        out[i] = out_i;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Combine multiply9x9 and mod_m with internal temporary storage
 *
 * The result in \p fac_result is guaranteed to be smaller than the modulus.
 *
 * \param[in]      factor      first factor with 9 numbers of 64 bits each
 * \param[in, out] fac_result  second factor and also the output of the same
 *                             size
 */
CELER_FUNCTION inline void
mulmod(RanluxppArray9 const& factor, RanluxppArray9& fac_result)
{
    RanluxppArray18 mul;
    multiply9x9(factor, fac_result, mul);
    modM(mul, fac_result);
}

//---------------------------------------------------------------------------//
/*!
 * Compute base to the n modulo m
 *
 * The arguments \p base and \p res may point to the same location.
 *
 * \param[in]  base  with 9 numbers of 64 bits each
 * \param[out] res   output with 9 numbers of 64 bits each
 * \param[in]  n     exponent
 */
CELER_FUNCTION inline void
powermod(RanluxppArray9 const& base, RanluxppArray9& res, RanluxppUInt n)
{
    RanluxppArray9 fac = base;
    res = {1, 0, 0, 0, 0, 0, 0, 0};

    RanluxppArray18 mul;
    while (n)
    {
        if (n & 1)
        {
            multiply9x9(res, fac, mul);
            modM(mul, res);
        }
        n >>= 1;
        if (!n)
        {
            break;
        }
        multiply9x9(fac, fac, mul);
        modM(mul, fac);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
