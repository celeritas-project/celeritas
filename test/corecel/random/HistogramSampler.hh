//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/HistogramSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/Types.hh"

#include "DiagnosticRngEngine.hh"
#include "Histogram.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Sampled distribution
struct SampledHistogram
{
    //! Sampled distribution
    std::vector<double> distribution;
    //! Average number of RNG samples
    double rng_count{};

    void print_expected() const;
};

//---------------------------------------------------------------------------//
/*!
 * Sample one or more distributions, returning a histogram.
 *
 * \code
    constexpr size_type num_samples = 1000;
    HistogramSampler calc_histogram(8, {-1, 1}, num_samples);
    std::vector<SampledHistogram> actual;


    for (real_type inc_e : {0.1, 1.0, 1e2, 1e3, 1e6})
    {
        for (real_type eps : {0.001, 0.01, 0.1})
        {
            MuAngularDistribution sample_mu{
                Energy{inc_e}, muon_mass, Energy{eps * inc_e}};
            actual.push_back(calc_histogram(sample_mu));
        }
    }
    SampledHistogram const expected[] = {
        {{0, 0, 0, 0, 0.484, 0.604, 0.96, 1.952}, 2},
        // ...
    };
    EXPECT_REF_EQ(expected, actual);
 * \endcode
 */
class HistogramSampler
{
  public:
    //!@{
    //! \name Type aliases
    using Dbl2 = Histogram::Dbl2;
    //!@}

    // Construct with number of samples per operator
    explicit inline HistogramSampler(size_type num_bins,
                                     Dbl2 domain,
                                     size_type num_samples);

    // Sample one distribution
    template<class DistributionT>
    inline SampledHistogram operator()(DistributionT&& sample_from);

    // Sample one distribution, transforming the result to a single real number
    template<class TransformT, class DistributionT>
    inline SampledHistogram
    operator()(TransformT&& transform, DistributionT&& sample_from);

  private:
    size_type num_bins_;
    Dbl2 domain_;
    size_type num_samples_;
    DiagnosticRngEngine<std::mt19937> rng_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& os, SampledHistogram const& sh);

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   SampledHistogram const& val1,
                                   SampledHistogram const& val2);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sample from and accumulate a distribution \c count times.
 *
 * \tparam AccumulatorT Accumulate a single estimator (usually Histogram)
 * \tparam DistributionT Distribution to sample
 * \tparam RandomEngineT PRNG
 */
template<class AccumulatorT, class DistributionT, class RandomEngineT>
void accumulate_n(AccumulatorT&& accumulate,
                  DistributionT&& sample_from,
                  RandomEngineT& engine,
                  size_type count)
{
    for (size_type i = 0; i < count; ++i)
    {
        accumulate(sample_from(engine));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct with number of samples.
 */
HistogramSampler::HistogramSampler(size_type num_bins,
                                   Dbl2 domain,
                                   size_type num_samples)
    : num_bins_(num_bins), domain_(domain), num_samples_(num_samples)
{
    CELER_EXPECT(num_bins_ > 0);
    CELER_EXPECT(num_samples_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample from and accumulate the given distribution.
 */
template<class DistributionT>
SampledHistogram HistogramSampler::operator()(DistributionT&& sample_from)
{
    // Transform
    return (*this)([](auto&& v) { return v; }, sample_from);
}

//---------------------------------------------------------------------------//
/*!
 * Sample from and accumulate the given distribution using a transform.
 */
template<class TransformT, class DistributionT>
SampledHistogram HistogramSampler::operator()(TransformT&& transform,
                                              DistributionT&& sample_from)
{
    Histogram hist{num_bins_, domain_};
    accumulate_n([&hist, &transform](auto&& v) { hist(transform(v)); },
                 sample_from,
                 rng_,
                 num_samples_);
    EXPECT_EQ(0, hist.underflow())
        << "Encountered values as low as " << hist.min();
    EXPECT_EQ(0, hist.overflow())
        << "Encountered values as high as " << hist.max();

    SampledHistogram result;
    result.distribution = hist.calc_density();
    result.rng_count = static_cast<double>(rng_.exchange_count())
                       / static_cast<double>(num_samples_);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
