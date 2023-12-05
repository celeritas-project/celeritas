//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"

#include "XorwowRngData.hh"
#include "detail/GenerateCanonical32.hh"
#include "distribution/GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate random data using the XORWOW algorithm.
 *
 * The XorwowRngEngine uses a C++11-like interface to generate random data. The
 * sampling of uniform floating point data is done with specializations to the
 * GenerateCanonical class.
 *
 * The \c resize function for \c XorwowRngStateData will fully randomize the
 * state at initialization. Alternatively, the state can be initialized with a
 * seed, subsequence, and offset.
 *
 * See Marsaglia (2003) for the theory underlying the algorithm and the the
 * "example" \c xorwow that combines an \em xorshift output with a Weyl
 * sequence (https://www.jstatsoft.org/index.php/jss/article/view/v008i14/916).
 *
 * For a description of the jump ahead method using the polynomial
 * representation of the recurrance, see: Haramoto, H., Matsumoto, M.,
 * Nishimura, T., Panneton, F., L’Ecuyer, P.  2008. "Efficient jump ahead for
 * F2-linear random number generators". INFORMS Journal on Computing.
 * https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0251
 */
class XorwowRngEngine
{
  public:
    //!@{
    //! \name Type aliases
    using uint_t = unsigned int;
    using result_type = uint_t;
    using Initializer_t = XorwowRngInitializer;
    using ParamsRef = NativeCRef<XorwowRngParamsData>;
    using StateRef = NativeRef<XorwowRngStateData>;
    //!@}

  public:
    //! Lowest value potentially generated
    static CELER_CONSTEXPR_FUNCTION result_type min() { return 0u; }
    //! Highest value potentially generated
    static CELER_CONSTEXPR_FUNCTION result_type max() { return 0xffffffffu; }

    // Construct from state and persistent data
    inline CELER_FUNCTION XorwowRngEngine(ParamsRef const& params,
                                          StateRef const& state,
                                          TrackSlotId tid);

    // Initialize state
    inline CELER_FUNCTION XorwowRngEngine& operator=(Initializer_t const&);

    // Generate a 32-bit pseudorandom number
    inline CELER_FUNCTION result_type operator()();

    // Apply the transformation to the state
    inline CELER_FUNCTION void next();

    // Jump ahead \c n steps
    inline CELER_FUNCTION void discard(ull_int n);

    // Jump ahead \c n subsequences (\c n * 2^67 steps)
    inline CELER_FUNCTION void discard_subsequence(ull_int n);

  private:
    /// TYPES ///

    using JumpPoly = Array<uint_t, 5>;
    using ArrayJumpPoly = Array<JumpPoly, 10>;

    /// DATA ///

    ParamsRef const& params_;
    XorwowState* state_;

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION void jump(ull_int, ArrayJumpPoly const&);
    inline CELER_FUNCTION void jump(JumpPoly const&);

    // Helper RNG for initializing the state
    struct SplitMix64
    {
        uint64_t state;
        inline CELER_FUNCTION uint64_t operator()();
    };
};

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for XorwowRngEngine.
 */
template<class RealType>
class GenerateCanonical<XorwowRngEngine, RealType>
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = RealType;
    //!@}

  public:
    //! Sample a random number on [0, 1)
    CELER_FORCEINLINE_FUNCTION result_type operator()(XorwowRngEngine& rng)
    {
        return detail::GenerateCanonical32<RealType>()(rng);
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from state and persistent data.
 */
CELER_FUNCTION
XorwowRngEngine::XorwowRngEngine(ParamsRef const& params,
                                 StateRef const& state,
                                 TrackSlotId tid)
    : params_(params)
{
    CELER_EXPECT(tid < state.state.size());
    state_ = &state.state[tid];
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG engine.
 *
 * This moves the state ahead to the given subsequence (a subsequence has size
 * 2^67) and skips \c offset random numbers.
 *
 * It is recommended to initialize the state using a very different generator
 * from the one being initialized to avoid correlations. Here the 64-bit
 * SplitMis64 generator is used for initialization (See Matsumoto, Wada,
 * Kuramoto, and Ashihara, "Common defects in initialization of pseudorandom
 * number generators". https://dl.acm.org/doi/10.1145/1276927.1276928.)
 */
CELER_FUNCTION XorwowRngEngine&
XorwowRngEngine::operator=(Initializer_t const& init)
{
    auto& s = state_->xorstate;

    // Initialize the state from the seed
    SplitMix64 rng{init.seed};
    uint64_t seed = rng();
    s[0] = static_cast<uint_t>(seed);
    s[1] = static_cast<uint_t>(seed >> 32);
    seed = rng();
    s[2] = static_cast<uint_t>(seed);
    s[3] = static_cast<uint_t>(seed >> 32);
    seed = rng();
    s[4] = static_cast<uint_t>(seed);
    state_->weylstate = static_cast<uint_t>(seed >> 32);

    // Skip ahead
    this->discard_subsequence(init.subsequence);
    this->discard(init.offset);

    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Generate a 32-bit pseudorandom number using the 'xorwow' engine.
 */
CELER_FUNCTION auto XorwowRngEngine::operator()() -> result_type
{
    this->next();
    return state_->weylstate + state_->xorstate[4];
}

//---------------------------------------------------------------------------//
/*!
 * Apply the transformation to the state.
 */
CELER_FUNCTION void XorwowRngEngine::next()
{
    auto& s = state_->xorstate;
    auto const t = (s[0] ^ (s[0] >> 2u));

    s[0] = s[1];
    s[1] = s[2];
    s[2] = s[3];
    s[3] = s[4];
    s[4] = (s[4] ^ (s[4] << 4u)) ^ (t ^ (t << 1u));

    state_->weylstate += 362437u;
}

//---------------------------------------------------------------------------//
/*!
 * Advance the state \c n steps.
 */
CELER_FUNCTION void XorwowRngEngine::discard(ull_int n)
{
    this->jump(n, params_.jump);
    state_->weylstate += static_cast<unsigned int>(n) * 362437u;
}

//---------------------------------------------------------------------------//
/*!
 * Advance the state \c n subsequences (\c n * 2^67 steps).
 *
 * Note that the Weyl sequence value remains the same since it has period 2^32
 * which divides evenly into 2^67.
 */
CELER_FUNCTION void XorwowRngEngine::discard_subsequence(ull_int n)
{
    this->jump(n, params_.jump_subsequence);
}

//---------------------------------------------------------------------------//
/*!
 * Jump ahead \c n steps or subsequences.
 *
 * This applies the jump polynomials until the given number of steps or
 * subsequences have been skipped.
 */
CELER_FUNCTION void
XorwowRngEngine::jump(ull_int n, ArrayJumpPoly const& jump_poly_arr)
{
    // Maximum number of times to apply any jump polynomial. Since the jump
    // sizes are 4^i for i = [0, 10), the max is 3.
    constexpr size_type max_num_jump = 3;

    // Start with the smallest jump (either one step or one subsequence)
    size_type jp_idx = 0;
    while (n > 0)
    {
        // Number of times to apply this jump polynomial
        uint_t num_jump = static_cast<uint_t>(n) & max_num_jump;
        for (size_type i = 0; i < num_jump; ++i)
        {
            this->jump(jump_poly_arr[jp_idx]);
        }
        ++jp_idx;
        n >>= 2;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Jump ahead using the given jump polynomial.
 *
 * This uses the polynomial representation to apply the recurrance to the
 * state.  The theory is described in
 * https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0251. Let
 * \f[
   g(z) = z^d \mod p(z) = a_1 z^{k-1} + ... + a_{k-1} z + a_k,
 * \f]
 * where \f$ p(z) = det(zI + T) \f$ is the characteristic polynomial and \f$ T
 * \f$ is the transformation matrix. Observing that \f$ g(z) = z^d q(z)p(z) \f$
 * for some polynomial \f$ q(z) \f$ and that \f$ p(T) = 0 \f$ (a fundamental
 * property of the characteristic polynomial, it follows that
 * \f[
   g(T) = T^d = a_1 A^{k-1} + ... + a_{k-1} A + a_k I.
 * \f]
 * Therefore, using the precalculated coefficients of the jump polynomial \f$
 * g(z) \f$ and Horner's method for polynomial evaluation, the state after \f$
 * d \f$ steps can be computed as
 * \f[
   T^d x = T(...T(T(T a_1 x + a_2 x) + a_3 x) + ... + a_{k-1} x) + a_k x.
 * \f]
 * Note that applying \f$ T \f$ to \f$ x \f$ is equivalent to calling \c
 * next(), and that in \f$ F_2 \f$, the finite field with two elements,
 * addition is the same as subtraction and equivalent to bitwise exclusive or.
 * Multiplication is bitwise and.
 */
CELER_FUNCTION void XorwowRngEngine::jump(JumpPoly const& jump_poly)
{
    constexpr size_type num_bits = 32;
    constexpr size_type num_words = 5;

    Array<uint_t, 5> s = {0};
    for (size_type i : range(num_words))
    {
        for (size_type j : range(num_bits))
        {
            if (jump_poly[i] & (1 << j))
            {
                for (size_type k : range(num_words))
                {
                    s[k] ^= state_->xorstate[k];
                }
            }
            this->next();
        }
    }
    state_->xorstate = s;
}

//---------------------------------------------------------------------------//
/*!
 * Generate a 64-bit pseudorandom number using the SplitMix64 engine.
 *
 * This is used to initialize the XORWOW state. See https://prng.di.unimi.it.
 */
CELER_FUNCTION uint64_t XorwowRngEngine::SplitMix64::operator()()
{
    uint64_t z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
