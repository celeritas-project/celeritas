//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-FileCopyrightText: 2025 Celeritas contributors
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/engine/detail/RanluxppImpl.hh
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/ranluxpp/helpers.h
 * and
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/ranluxpp/mulmod.h
 * and
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/ranluxpp/ranlux_lcg.h
 */
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/random/data/RanluxppTypes.hh"

namespace celeritas
{
namespace detail
{

// Compute a + b and set overflow accordingly
inline CELER_FUNCTION RanluxppUInt add_overflow(RanluxppUInt a,
                                                RanluxppUInt b,
                                                unsigned int& overflow);

// Compute a + b and increment carry if there was an overflow
inline CELER_FUNCTION RanluxppUInt add_carry(RanluxppUInt a,
                                             RanluxppUInt b,
                                             unsigned int& carry);

// Compute a - b` and set overflow accordingly
inline CELER_FUNCTION RanluxppUInt sub_overflow(RanluxppUInt a,
                                                RanluxppUInt b,
                                                unsigned int& overflow);

// Compute a - b and increment `carry` if there was an overflow
inline CELER_FUNCTION RanluxppUInt sub_carry(RanluxppUInt a,
                                             RanluxppUInt b,
                                             unsigned int& carry);

// Update r = r - (t1 + t2) + (t3 + t2) * b ** 10
inline CELER_FUNCTION int64_t
compute_remainder(Span<RanluxppUInt const, 9> upper, Span<RanluxppUInt, 9> r);

// Multiply two 576 bit numbers, stored as 9 numbers of 64 bits each
[[nodiscard]] inline CELER_FUNCTION RanluxppArray18
multiply_9x9(RanluxppArray9 const& in1, RanluxppArray9 const& in2);

// Compute a value congruent to mul modulo m less than 2 ** 576
[[nodiscard]] inline CELER_FUNCTION RanluxppArray9
compute_modulus(RanluxppArray18 const& mul);

// Combine multiply9x9 and mod_m with internal temporary storage
[[nodiscard]] inline CELER_FUNCTION RanluxppArray9 compute_mod_multiply(
    RanluxppArray9 const& factor1, RanluxppArray9 const& factor2);

// Compute base to the n modulo m
[[nodiscard]] inline CELER_FUNCTION RanluxppArray9
compute_power_modulus(RanluxppArray9 base, RanluxppUInt n);

// Convert RANLUX numbers to an LCG state
[[nodiscard]] inline CELER_FUNCTION RanluxppArray9
to_lcg(RanluxppNumber const& ranlux);

// Convert an LCG state to RANLUX numbers
[[nodiscard]] inline CELER_FUNCTION RanluxppNumber
to_ranlux(RanluxppArray9 const& lcg);

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Compute sum of \p a and \p b and set \p overflow accordingly.
 *
 * \param[in]  a         The first operand of the sum
 * \param[in]  b         The second operand of the sum
 * \param[out] overflow  The amount of (potential) overflow
 *
 * \return The result of the sum
 */
CELER_FUNCTION RanluxppUInt add_overflow(RanluxppUInt a,
                                         RanluxppUInt b,
                                         unsigned int& overflow)
{
    RanluxppUInt add = a + b;
    overflow = (add < a);
    return add;
}

//---------------------------------------------------------------------------//
/*!
 * Compute sum of \p a and \p b and increment \p carry if there was an
 * overflow.
 *
 * \param[in]  a      The first operand of the sum
 * \param[in]  b      The second operand of the sum
 * \param[out] carry  Maintains sum of all overflows
 *
 * \return The result of the sum
 */
CELER_FUNCTION RanluxppUInt add_carry(RanluxppUInt a,
                                      RanluxppUInt b,
                                      unsigned int& carry)
{
    unsigned int overflow;
    RanluxppUInt add = add_overflow(a, b, overflow);

    // Do NOT branch on overflow to avoid jumping code, just add 0 if there was
    // no overflow.
    carry += overflow;
    return add;
}

//---------------------------------------------------------------------------//
/*!
 * Compute difference of \p a and \p b and set \p overflow accordingly.
 *
 * \param[in] a          The first operand of the subtraction
 * \param[in] b          The second operand of the subtraction
 * \param[out] overflow  Stores the amount of any overflow
 *
 * \return The result of the subtraction
 */
CELER_FUNCTION RanluxppUInt sub_overflow(RanluxppUInt a,
                                         RanluxppUInt b,
                                         unsigned int& overflow)
{
    RanluxppUInt sub = a - b;
    overflow = (sub > a);
    return sub;
}

//---------------------------------------------------------------------------//
/*!
 * Compute difference of \p a and \p b and increment \p carry if there was an
 * overflow.
 *
 * \param[in] a       The first operand of the subtraction
 * \param[in] b       The second operand of the subtraction
 * \param[out] carry  Maintains sum of all overflows
 *
 * \return The result of the subtraction
 */
CELER_FUNCTION RanluxppUInt sub_carry(RanluxppUInt a,
                                      RanluxppUInt b,
                                      unsigned int& carry)
{
    unsigned int overflow;
    RanluxppUInt sub = sub_overflow(a, b, overflow);

    // Do NOT branch on overflow to avoid jumping code, just add 0 if there was
    // no overflow.
    carry += overflow;
    return sub;
}

//---------------------------------------------------------------------------//
/*!
 * Update r = r - (t1 + t2) + (t3 + t2) * b ** 10.
 *
 * This function also yields cbar = floor(r / m) as its return value (int64_t
 * because the value can be -1). With an initial value of r = t0, this can
 * be used for computing the remainder after division by m (see the function
 * mod_m in mulmod.h). The function to_ranlux passes r = 0 and uses only the
 * return value to obtain the decimal expansion after division by m.
 */
CELER_FUNCTION int64_t compute_remainder(Span<RanluxppUInt const, 9> upper,
                                         Span<RanluxppUInt, 9> r)
{
    // Subtract t1 (24 * 24 = 576 bits)
    unsigned int carry = 0;
    for (int i : celeritas::range(9))
    {
        RanluxppUInt r_i = r[i];
        r_i = sub_overflow(r_i, carry, carry);

        RanluxppUInt t1_i = upper[i];
        r_i = sub_carry(r_i, t1_i, carry);
        r[i] = r_i;
    }
    int64_t c = -(static_cast<int64_t>(carry));

    // Subtract t2 (only 240 bits, so need to extend)
    carry = 0;
    for (int i : celeritas::range(9))
    {
        RanluxppUInt r_i = r[i];
        r_i = sub_overflow(r_i, carry, carry);

        RanluxppUInt t2_bits = 0;
        if (i < 4)
        {
            t2_bits += upper[i + 5] >> 16;
            if (i < 3)
            {
                t2_bits += upper[i + 6] << 48;
            }
        }
        r_i = sub_carry(r_i, t2_bits, carry);
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

        r_3 = add_carry(r_3, t2_bits, carry);
        r_3 = add_carry(r_3, t3_bits, carry);

        r[3] = r_3;
    }
    for (int i : celeritas::range(3))
    {
        RanluxppUInt r_i = r[i + 4];
        r_i = add_overflow(r_i, carry, carry);

        RanluxppUInt t2_bits = (upper[5 + i] >> 32) + (upper[6 + i] << 32);
        RanluxppUInt t3_bits = (upper[i] >> 16) + (upper[1 + i] << 48);

        r_i = add_carry(r_i, t2_bits, carry);
        r_i = add_carry(r_i, t3_bits, carry);

        r[i + 4] = r_i;
    }
    {
        RanluxppUInt r_7 = r[7];
        r_7 = add_overflow(r_7, carry, carry);

        RanluxppUInt t2_bits = (upper[8] >> 32);
        RanluxppUInt t3_bits = (upper[3] >> 16) + (upper[4] << 48);

        r_7 = add_carry(r_7, t2_bits, carry);
        r_7 = add_carry(r_7, t3_bits, carry);

        r[7] = r_7;
    }
    {
        RanluxppUInt r_8 = r[8];
        r_8 = add_overflow(r_8, carry, carry);

        RanluxppUInt t3_bits = (upper[4] >> 16) + (upper[5] << 48);

        r_8 = add_carry(r_8, t3_bits, carry);

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
/*!
 * Multiply two 576 bit numbers, stored as 9 numbers of 64 bits each.
 *
 * \param[in]  in1  first factor as 9 numbers of 64 bits each
 * \param[in]  in2  second factor as 9 numbers of 64 bits each
 *
 * \return  result with 18 numbers of 64 bits each
 */
CELER_FUNCTION RanluxppArray18 multiply_9x9(RanluxppArray9 const& in1,
                                            RanluxppArray9 const& in2)
{
    RanluxppArray18 result;

    RanluxppUInt next = 0;
    unsigned int nextCarry = 0;

#if defined(__clang__) || defined(__INTEL_COMPILER) || defined(__CUDA_ARCH__)
#    pragma unroll
#elif defined(__GNUC__) && __GNUC__ >= 8 && !defined(__CUDACC__)
// This pragma was introduced in GCC version 8.
#    pragma GCC unroll 18
#endif
    for (int i = 0; i < 18; ++i)
    {
        RanluxppUInt current = next;
        unsigned int carry = nextCarry;

        next = 0;
        nextCarry = 0;

#if defined(__clang__) || defined(__INTEL_COMPILER) || defined(__CUDA_ARCH__)
#    pragma unroll
#elif defined(__GNUC__) && __GNUC__ >= 8 && !defined(__CUDACC__)
// This pragma was introduced in GCC version 8.
#    pragma GCC unroll 9
#endif
        for (int j = 0; j < 9; ++j)
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
            RanluxppUInt middle = add_overflow(middle1, middle2, overflow);
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

            lower = add_overflow(lower, middle_lower, overflow);
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
            current = add_carry(current, lower, carry);

            // Add to next, remember nextCarry.
            next = add_carry(next, upper, nextCarry);
        }

        next = add_carry(next, carry, nextCarry);
        result[i] = current;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Compute a value congruent to mul modulo m less than 2 ** 576.
 *
 * This computes \f$ m = 2^{576} - 2^{240} + 1 \f$. The result in
 * \p out is guaranteed to be smaller than the modulus.
 *
 * \param[in] mul product from multiply_9x9 with 18 numbers of 64 bits each
 * \param[out] out result with 9 numbers of 64 bits each
 */
CELER_FUNCTION RanluxppArray9 compute_modulus(RanluxppArray18 const& mul)
{
    RanluxppArray9 result;

    RanluxppArray9 r;
    for (auto i : celeritas::range(0, 9))
    {
        r[i] = mul[i];
    }

    // Make a subspan of the last 9 elements of mul
    auto mul_end = celeritas::make_span(mul).subspan<9, 9>();
    CELER_ASSERT(mul_end.size() == 9);

    int64_t c = compute_remainder(mul_end, celeritas::make_span(r));

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

        RanluxppUInt out_0 = sub_carry(r_0, c, carry);
        result[0] = out_0;
    }
    for (int i : celeritas::range(1, 3))
    {
        RanluxppUInt r_i = r[i];
        r_i = sub_overflow(r_i, carry, carry);

        RanluxppUInt out_i = sub_carry(r_i, t0, carry);
        result[i] = out_i;
    }
    {
        RanluxppUInt r_3 = r[3];
        r_3 = sub_overflow(r_3, carry, carry);

        RanluxppUInt out_3 = sub_carry(r_3, t2, carry);
        result[3] = out_3;
    }
    for (int i : celeritas::range(4, 9))
    {
        RanluxppUInt r_i = r[i];
        r_i = sub_overflow(r_i, carry, carry);

        RanluxppUInt out_i = sub_carry(r_i, t1, carry);
        result[i] = out_i;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Combine multiply_9x9 and compute_modulus with internal temporary storage.
 *
 * The result in \p fac_result is guaranteed to be smaller than the modulus.
 *
 * \param[in] factor1 first factor with 9 numbers of 64 bits each
 * \param[in] factor2 second factor with 9 numbers of 64 bits each
 */
CELER_FUNCTION RanluxppArray9 compute_mod_multiply(
    RanluxppArray9 const& factor1, RanluxppArray9 const& factor2)
{
    RanluxppArray18 mul = multiply_9x9(factor1, factor2);
    return compute_modulus(mul);
}

//---------------------------------------------------------------------------//
/*!
 * Compute \p base to the \p n modulo m.
 *
 * The arguments \p base and \p res may point to the same location.
 *
 * \param[in]  base  with 9 numbers of 64 bits each
 * \param[out] res   output with 9 numbers of 64 bits each
 * \param[in]  n     exponent
 */
CELER_FUNCTION RanluxppArray9 compute_power_modulus(RanluxppArray9 base,
                                                    RanluxppUInt n)
{
    RanluxppArray9 res = {1, 0, 0, 0, 0, 0, 0, 0, 0};

    RanluxppArray18 mul;
    while (n)
    {
        if (n & 1)
        {
            mul = multiply_9x9(res, base);
            res = compute_modulus(mul);
        }
        n >>= 1;
        if (!n)
        {
            break;
        }
        mul = multiply_9x9(base, base);
        base = compute_modulus(mul);
    }

    return res;
}

//---------------------------------------------------------------------------//
/*!
 * Convert RANLUX numbers to an LCG state.
 *
 * Computes \f$ m = 2^{576} - 2^{240} + 1 \f$.
 *
 * \param[in] ranluxpp_carry_state  A struct containing the state and carry
 *                                  numbers
 * \return     The 576 bits of the LCG state, smaller than m
 */
CELER_FUNCTION RanluxppArray9 to_lcg(RanluxppNumber const& ranlux)
{
    RanluxppArray9 result;

    RanluxppArray9 const& number = ranlux.number;
    unsigned int c = ranlux.carry;

    // Subtract the final 240 bits.
    unsigned int carry = 0;
    for (int i : celeritas::range(9))
    {
        RanluxppUInt ranlux_i = number[i];
        RanluxppUInt lcg_i = sub_overflow(ranlux_i, carry, carry);

        RanluxppUInt bits = 0;
        if (i < 4)
        {
            bits += number[i + 5] >> 16;
            if (i < 3)
            {
                bits += number[i + 6] << 48;
            }
        }
        lcg_i = sub_carry(lcg_i, bits, carry);
        result[i] = lcg_i;
    }

    // Add and propagate the carry bit.
    for (RanluxppUInt& lcg_val : result)
    {
        lcg_val = add_overflow(lcg_val, c, c);
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an LCG state to RANLUX numbers.
 *
 * \f$ m = 2^{576} - 2^{240} + 1 \f$
 *
 * \param[in]  lcg     The 576 bits of the LCG state, must be smaller than m
 * \result  A struct containing the ranlux numbers (as 576 bits) and the carry
 *          bit
 */
CELER_FUNCTION RanluxppNumber to_ranlux(RanluxppArray9 const& lcg)
{
    RanluxppNumber result;
    int64_t c = compute_remainder(celeritas::make_span(lcg),
                                  celeritas::make_span(result.number));

    // ranlux = t1 + t2 + c
    unsigned int carry = 0;
    for (int i : celeritas::range(9))
    {
        RanluxppUInt in_i = lcg[i];
        RanluxppUInt tmp_i = add_overflow(in_i, carry, carry);

        RanluxppUInt bits = 0;
        if (i < 4)
        {
            bits += lcg[i + 5] >> 16;
            if (i < 3)
            {
                bits += lcg[i + 6] << 48;
            }
        }
        tmp_i = add_carry(tmp_i, bits, carry);
        result.number[i] = tmp_i;
    }

    // If c = -1, we need to add it to all components.
    int64_t c1 = c >> 1;
    result.number[0] = add_overflow(result.number[0], c, carry);
    for (int i : celeritas::range(1, 9))
    {
        RanluxppUInt ranlux_i = result.number[i];
        add_overflow(ranlux_i, carry, carry);
        add_carry(ranlux_i, c1, carry);
    }

    result.carry = carry;

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
