//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/XorwowRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/random/data/XorwowRngData.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/detail/GenerateCanonical32.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate random data using the XORWOW algorithm.
 *
 * The XorwowRngEngine uses a C++11-like interface to generate random data
 * using Marasglia's modified xorshift generator
 * \citep{marsaglia-xorshift-2003,
 * https://www.jstatsoft.org/index.php/jss/article/view/v008i14/916}.
 * The sampling of uniform floating point data is done with specializations to
 * the \c GenerateCanonical class.
 *
 * The \c resize function for \c XorwowRngStateData will fully randomize the
 * state at initialization. Alternatively, the state can be initialized with a
 * seed, subsequence, and offset.
 *
 * Initialization moves the state ahead to the given subsequence (a subsequence
 * has size \f$2^{67}\f$) and skips \c offset random numbers.
 * It is recommended to initialize the state using a very different generator
 * from the one being initialized to avoid correlations. Here, the 64-bit
 * SplitMix64 generator is used for initialization
 * \citep{matsumoto-defects-2007,
 * https://dl.acm.org/doi/10.1145/1276927.1276928}.
 *
 * For a description of the jump ahead method using the polynomial
 * representation of the recurrence, see
 * \citet{haramato-jump-2008,
 * https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0251}.
 *
 * The jump polynomials were precomputed using [a python script](
 * https://github.com/celeritas-project/utils/blob/main/prng/xorwow-jump.py).
 * For a more detailed description of how to calculate the jump polynomials
 * using Knuth's square-and-multiply algorithm in \f$ O(k^2 log d) \f$ time
 * (where \em k is the number of bits in the state and \em d is the number of
 * steps to skip ahead), see \citet{collins-rng-2008,
 * http://www.dtic.mil/docs/citations/ADA486379}.
 */
class XorwowRngEngine
{
  public:
    //!@{
    //! \name Type aliases
    using uint_t = XorwowUInt;
    using result_type = uint_t;
    using Initializer_t = XorwowRngInitializer;
    using ParamsRef = NativeCRef<XorwowRngParamsData>;
    using StateRef = NativeRef<XorwowRngStateData>;
    using RngStateInitializer_t = XorwowRngStateInitializer;
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

    // Initialize state with an RNG initializer
    inline CELER_FUNCTION XorwowRngEngine& operator=(Initializer_t const&);

    // Initialize state with a state initializer
    inline CELER_FUNCTION XorwowRngEngine&
    operator=(RngStateInitializer_t const&);

    // Generate a 32-bit pseudorandom number
    inline CELER_FUNCTION result_type operator()();

    // Advance the state \c count times
    inline CELER_FUNCTION void discard(ull_int count);

    // Initialize a state for a new spawned RNG
    inline CELER_FUNCTION RngStateInitializer_t branch();

  private:
    /// TYPES ///

    using JumpPoly = Array<uint_t, 5>;
    using ArrayJumpPoly = Array<JumpPoly, 32>;

    /// DATA ///

    ParamsRef const& params_;
    XorwowState* state_;

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION void discard_subsequence(ull_int);
    inline CELER_FUNCTION void next(XorwowState&);
    inline CELER_FUNCTION void
    jump(ull_int, ArrayJumpPoly const&, XorwowState&);
    inline CELER_FUNCTION void jump(JumpPoly const&, XorwowState&);

    // Helper RNG for initializing the state
    struct SplitMix64
    {
        std::uint64_t state;
        inline CELER_FUNCTION std::uint64_t operator()();
    };
};

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for XorwowRngEngine.
 */
template<class RealType>
struct GenerateCanonical<XorwowRngEngine, RealType>
{
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = RealType;
    //!@}

    //! Declare that we use the 32-bit canonical generator
    static constexpr auto policy = GenerateCanonicalPolicy::builtin32;

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
 */
CELER_FUNCTION XorwowRngEngine&
XorwowRngEngine::operator=(Initializer_t const& init)
{
    auto& s = state_->xorstate;

    // Initialize the state from the seed
    SplitMix64 rng{init.seed[0]};
    std::uint64_t seed = rng();
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
 * Initialize the RNG engine with a state initializer.
 */
CELER_FUNCTION XorwowRngEngine&
XorwowRngEngine::operator=(RngStateInitializer_t const& state_init)
{
    state_->xorstate = state_init.xorstate;
    state_->weylstate = state_init.weylstate;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Generate a 32-bit pseudorandom number using the 'xorwow' engine.
 */
CELER_FUNCTION auto XorwowRngEngine::operator()() -> result_type
{
    this->next(*state_);
    state_->weylstate += 362437u;
    return state_->weylstate + state_->xorstate[4];
}

//---------------------------------------------------------------------------//
/*!
 * Advance the state \c count times.
 */
CELER_FUNCTION void XorwowRngEngine::discard(ull_int count)
{
    this->jump(count, params_.jump, *state_);
    state_->weylstate += static_cast<unsigned int>(count) * 362437u;
}

//---------------------------------------------------------------------------//
/*!
 * Generate a branched and (hopefully) decorreleated RNG
 *
 * \todo There are good reasons to believe that an RNG that is based on XOR
 *       operations may not be decorrelated when XORing the state.  Work on an
 *       improved methodology may be warranted.
 */
CELER_FUNCTION XorwowRngEngine::RngStateInitializer_t XorwowRngEngine::branch()
{
    XorwowState new_state;

    // Copy the state into the new state
    new_state.xorstate = state_->xorstate;

    // Advance this RNG 4 times to get new state
    this->jump(4, params_.jump, *state_);

    // XOR the state with the new state
    for (auto i : celeritas::range(new_state.xorstate.size()))
    {
        new_state.xorstate[i] ^= state_->xorstate[i];
    }

    // Advance the new state 4 times to (hopefully) decorrelate
    this->jump(4, params_.jump, new_state);

    // Create and return the state initializer
    return RngStateInitializer_t{new_state.xorstate, state_->weylstate};
}

//---------------------------------------------------------------------------//
/*!
 * Advance the state \c count subsequences (\c count * 2^67 times).
 *
 * Note that the Weyl sequence value remains the same since it has period 2^32
 * which divides evenly into 2^67.
 */
CELER_FUNCTION void XorwowRngEngine::discard_subsequence(ull_int count)
{
    this->jump(count, params_.jump_subsequence, *state_);
}

//---------------------------------------------------------------------------//
/*!
 * Apply the transformation to the state.
 *
 * This does not update the Weyl sequence value.
 */
CELER_FUNCTION void XorwowRngEngine::next(XorwowState& state)
{
    auto& s = state.xorstate;
    auto const t = (s[0] ^ (s[0] >> 2u));

    s[0] = s[1];
    s[1] = s[2];
    s[2] = s[3];
    s[3] = s[4];
    s[4] = (s[4] ^ (s[4] << 4u)) ^ (t ^ (t << 1u));
}

//---------------------------------------------------------------------------//
/*!
 * Jump ahead \c count steps or subsequences.
 *
 * This applies the jump polynomials until the given number of steps or
 * subsequences have been skipped.
 */
CELER_FUNCTION void XorwowRngEngine::jump(ull_int count,
                                          ArrayJumpPoly const& jump_poly_arr,
                                          XorwowState& state)
{
    // Maximum number of times to apply any jump polynomial. Since the jump
    // sizes are 4^i for i = [0, 32), the max is 3.
    constexpr size_type max_num_jump = 3;

    // Start with the smallest jump (either one step or one subsequence)
    size_type jump_idx = 0;
    while (count > 0)
    {
        // Number of times to apply this jump polynomial
        uint_t num_jump = static_cast<uint_t>(count) & max_num_jump;
        for (size_type i = 0; i < num_jump; ++i)
        {
            CELER_ASSERT(jump_idx < jump_poly_arr.size());
            this->jump(jump_poly_arr[jump_idx], state);
        }
        ++jump_idx;
        count >>= 2;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Jump ahead using the given jump polynomial.
 *
 * This uses the polynomial representation to apply the recurrence to the
 * state. The theory is described in
 * https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0251. Let
 * \f[
   g(z) = z^d \mod p(z) = a_1 z^{k-1} + ... + a_{k-1} z + a_k,
 * \f]
 * where \f$ p(z) = det(zI + T) \f$ is the characteristic polynomial and \f$ T
 * \f$ is the transformation matrix. Observing that \f$ g(z) = z^d q(z)p(z) \f$
 * for some polynomial \f$ q(z) \f$ and that \f$ p(T) = 0 \f$ (a fundamental
 * property of the characteristic polynomial), it follows that
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
 * addition is the same as subtraction and equivalent to bitwise exclusive or,
 * and multiplication is bitwise and.
 */
CELER_FUNCTION void
XorwowRngEngine::jump(JumpPoly const& jump_poly, XorwowState& state)
{
    Array<uint_t, 5> s = {0};
    for (size_type i : range(params_.num_words()))
    {
        for (size_type j : range(params_.num_bits()))
        {
            if (jump_poly[i] & (1 << j))
            {
                for (size_type k : range(params_.num_words()))
                {
                    s[k] ^= state.xorstate[k];
                }
            }
            this->next(state);
        }
    }
    state.xorstate = s;
}

//---------------------------------------------------------------------------//
/*!
 * Generate a 64-bit pseudorandom number using the SplitMix64 engine.
 *
 * This is used to initialize the XORWOW state. See https://prng.di.unimi.it.
 */
CELER_FUNCTION std::uint64_t XorwowRngEngine::SplitMix64::operator()()
{
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
