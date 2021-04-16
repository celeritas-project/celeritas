//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SequenceEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include "random/distributions/GenerateCanonical.hh"
#include "base/Span.hh"

namespace celeritas_test
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
    //@{
    //! Typedefs
    using result_type = std::uint32_t;
    using VecResult   = std::vector<result_type>;
    using LimitsT     = std::numeric_limits<result_type>;
    using size_type   = VecResult::size_type;
    //!@}

  public:
    // Nearly reproduce the given stream of reals with generate_canonical
    inline static SequenceEngine
    from_reals(celeritas::Span<const double> values);

    // Nearly reproduce the given stream of reals with generate_canonical
    inline static SequenceEngine
    from_reals(std::initializer_list<double> values);

    // Construct from a sequence of integers (the sequence to reproduce)
    explicit inline SequenceEngine(VecResult values);

    // Get the next random number in the sequence; throw if past the end
    inline result_type operator()();

    size_type count() const { return iter_ - values_.begin(); }
    size_type max_count() const { return values_.size(); }

    //!@{
    //! Engine limits
    static constexpr result_type min() { return LimitsT::min(); }
    static constexpr result_type max() { return LimitsT::max(); }
    //!@}

  private:
    VecResult                 values_;
    VecResult::const_iterator iter_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for SequenceEngine.
 */
template<class T>
class GenerateCanonical<celeritas_test::SequenceEngine, T>
{
  public:
    //!@{
    //! Type aliases
    using real_type   = T;
    using result_type = real_type;
    //!@}

  public:
    // Sample a random number
    inline result_type operator()(celeritas_test::SequenceEngine& rng);
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "SequenceEngine.i.hh"
