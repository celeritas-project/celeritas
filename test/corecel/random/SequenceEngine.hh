//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/SequenceEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * "Random" number generator that returns a sequence of values.
 *
 * This is useful for testing algorithms that expect a random stream on the
 * host: you can specify the sequence of values to return. When the sequence is
 * used up, the next operator call will raise an exception.
 *
 * The factory function SequenceEngine::from_reals will create an integer
 * sequence that exactly reproduces the given real numbers (which must be in
 * the half-open interval \f$ [0, 1) \f$) using the \c generate_canonical
 * function.
 */
class SequenceEngine
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = std::uint32_t;
    using VecResult = std::vector<result_type>;
    using LimitsT = std::numeric_limits<result_type>;
    using size_type = VecResult::size_type;
    //!@}

  public:
    // Nearly reproduce the given stream of reals with generate_canonical
    inline static SequenceEngine from_reals(Span<double const> values);

    // Nearly reproduce the given stream of reals with generate_canonical
    inline static SequenceEngine
    from_reals(std::initializer_list<double> values);

    // Construct from a sequence of integers (the sequence to reproduce)
    explicit inline SequenceEngine(VecResult values);

    // Get the next random number in the sequence; throw if past the end
    inline result_type operator()();

    size_type count() const { return i_; }
    size_type max_count() const { return values_.size(); }

    //!@{
    //! \name Engine limits
    static constexpr result_type min() { return LimitsT::min(); }
    static constexpr result_type max() { return LimitsT::max(); }
    //!@}

  private:
    VecResult values_;
    size_type i_;
};
//---------------------------------------------------------------------------//
}  // namespace test

//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for SequenceEngine.
 */
template<class T>
class GenerateCanonical<test::SequenceEngine, T>
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = T;
    using result_type = real_type;
    //!@}

  public:
    // Sample a random number
    inline result_type operator()(test::SequenceEngine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct a sequence to nearly reproduce the given stream of reals.
 *
 * Values below 1e-64 will not be represented. Furthermore, some other values
 * (see the RngEngine unit test) will not be exactly reproduced. All input
 * values must be bounded in  \f$ [0, 1) \f$ .
 */
SequenceEngine SequenceEngine::from_reals(Span<double const> values)
{
    using real_type = double;

    CELER_EXPECT(!values.empty());
    real_type const range = SequenceEngine::max() + real_type(1);

    SequenceEngine::VecResult elements(values.size() * 2);
    auto dst = elements.begin();
    for (double v : values)
    {
        CELER_EXPECT(v >= 0 && v < 1);
        // Calculate first (big end) value
        v *= range;
        auto second = static_cast<result_type>(std::floor(v));

        // Calculate second (little end) value
        v -= second;
        v *= range;

        // Store values
        *dst++ = static_cast<result_type>(std::floor(v));
        *dst++ = second;
    }
    CELER_ENSURE(dst == elements.end());

    return SequenceEngine(std::move(elements));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a sequence to nearly reproduce the given stream of reals.
 */
SequenceEngine SequenceEngine::from_reals(std::initializer_list<double> values)
{
    return SequenceEngine::from_reals({values.begin(), values.end()});
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a sequence of integers (the sequence to reproduce).
 */
SequenceEngine::SequenceEngine(VecResult values)
    : values_(std::move(values)), i_{0}
{
    CELER_EXPECT(!values_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Get the next random value in the sequence.
 *
 * This is designed to throw an error rather than dereference an off-the-end
 * iterator even when debugging is disabled, since this class is only used
 * inside of tests.
 */
auto SequenceEngine::operator()() -> result_type
{
    if (CELER_UNLIKELY(i_ == values_.size()))
    {
        // Always throw a debug error rather than letting the test crash
        throw DebugError{{DebugErrorType::precondition,
                          "SequenceEngine RNG stream exceeded",
                          __FILE__,
                          __LINE__}};
    }
    return values_[i_++];
}
//---------------------------------------------------------------------------//
}  // namespace test

//---------------------------------------------------------------------------//
/*!
 * Specialization for sequence RNG.
 *
 * This should reproduce double exactly, and give the same "single precision"
 * approximation for floats. It will be inaccurate for higher-precision
 * numbers.
 *
 * Some RNGs always consume 64 bits from the RNG, and can thus provide
 * single-precision values that are lower than 2^{-32} -- this reproduces that
 * behavior for small values.
 */
template<class T>
T GenerateCanonical<test::SequenceEngine, T>::operator()(
    test::SequenceEngine& rng)
{
    // Range for sequence engine should be [0, 2^32 - 1) = 2^32
    real_type const range = static_cast<real_type>(test::SequenceEngine::max())
                            + real_type(1);
    real_type result = static_cast<real_type>(rng());
    result += rng() * range;
    result *= 1 / ipow<2>(range);
    if (CELER_UNLIKELY(result == real_type(1)))
    {
        // Change to nearest point value closer to zero
        result = std::nextafter(result, real_type(0));
    }
    CELER_ENSURE(result >= 0 && result < 1);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
