//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SequenceEngine.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Algorithms.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Construct a sequence to nearly reproduce the given stream of reals.
 *
 * Values below 1e-64 will not be represented. Furthermore, some other values
 * (see the RngEngine unit test) will not be exactly reproduced. All input
 * values must be bounded in  \f$ [0, 1) \f$ .
 */
SequenceEngine SequenceEngine::from_reals(celeritas::Span<const double> values)
{
    using real_type = double;

    CELER_EXPECT(!values.empty());
    const real_type range = SequenceEngine::max() + real_type(1);

    SequenceEngine::VecResult elements(values.size() * 2);
    auto                      dst = elements.begin();
    for (double v : values)
    {
        CELER_EXPECT(v >= 0 && v < 1);
        // Calculate first (big end) value
        v *= range;
        result_type second = std::floor(v);

        // Calculate second (little end) value
        v -= second;
        v *= range;

        // Store values
        *dst++ = std::floor(v);
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
    : values_(std::move(values)), iter_(values_.cbegin())
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
    CELER_EXPECT(iter_ != values_.cend());
    if (CELER_UNLIKELY(iter_ == values_.cend()))
    {
        // Always throw a debug error rather than letting the test crash
        celeritas::throw_debug_error(celeritas::DebugErrorType::precondition,
                                     "SequenceEngine RNG stream exceeded",
                                     __FILE__,
                                     __LINE__);
    }
    return *iter_++;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test

namespace celeritas
{
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
T GenerateCanonical<celeritas_test::SequenceEngine, T>::operator()(
    celeritas_test::SequenceEngine& rng)
{
    // Range for sequence engine should be [0, 2^32 - 1) = 2^32
    const real_type range = celeritas_test::SequenceEngine::max()
                            + real_type(1);
    real_type result = rng();
    result += rng() * range;
    result *= 1 / celeritas::ipow<2>(range);
    if (CELER_UNLIKELY(result == real_type(1)))
    {
        // Change to nearest point value closer to zero
        result = std::nextafter(result, real_type(0));
    }
    CELER_ENSURE(result >= 0 && result < 1);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
