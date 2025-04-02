//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SampleStats.cc
//---------------------------------------------------------------------------//
#include "SampleStats.hh"

#include <cmath>

#include "corecel/io/Repr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Construct from accumulator
SampleStats::SampleStats(SampleStatsAccumulator acc)
    : min_(acc.min())
    , max_(acc.max())
    , mean_(acc.moment(1) / acc.moment(0))
    , count_(acc.count())
{
    real_type n = acc.moment(0);
    stdev_ = std::sqrt((acc.moment(2) - ipow<2>(acc.moment(1) / n)) / n);
}

//---------------------------------------------------------------------------//
//! Construct manually
SampleStats::SampleStats(real_type min,
                         real_type max,
                         real_type mean,
                         real_type stdev,
                         size_type count)
    : min_(min), max_(max), mean_(mean), stdev_(stdev), count_(count)
{
    CELER_EXPECT(min <= max);
    CELER_EXPECT(count > 0);
    CELER_EXPECT(stdev >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the z-score for the difference in means between two sample
 * statistics.
 */
real_type SampleStats::calc_z_score(SampleStats const& other) const
{
    // Calculate standard error of the difference between means
    real_type se_diff = std::sqrt(
        (stdev_ * stdev_ / static_cast<real_type>(count_))
        + (other.stdev_ * other.stdev_ / static_cast<real_type>(other.count_)));

    // Calculate and return z-score for the difference in means
    return std::fabs(mean_ - other.mean_) / se_diff;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the F-statistic for the ratio of variances between two sample
 * statistics.
 */
real_type SampleStats::calc_f_statistic(SampleStats const& other) const
{
    // F-statistic is the ratio of variances (square of standard deviations)
    real_type var1 = stdev_ * stdev_;
    real_type var2 = other.stdev_ * other.stdev_;

    // Always return F >= 1 by putting larger variance in numerator
    if (var1 < var2)
    {
        std::swap(var1, var2);
    }
    return var1 / var2;
}

//---------------------------------------------------------------------------//
/*!
 * Compare if two sample statistics are probably equivalent at given
 * confidence.
 */
bool SampleStats::probably_equal(SampleStats const& other,
                                 real_type confidence) const
{
    // Calculate z-score
    real_type z_score = this->calc_z_score(other);

    // Calculate critical value based on confidence level with simple map
    real_type critical_value = 1.0;
    if (confidence >= 0.99)
        critical_value = 2.576;
    else if (confidence >= 0.95)
        critical_value = 1.96;
    else if (confidence >= 0.90)
        critical_value = 1.645;

    // Return true if difference is not statistically significant
    return z_score <= critical_value;
}

//---------------------------------------------------------------------------//
/*!
 * Print "expected" code.
 */
void SampleStats::print_expected() const
{
    std::cout << "SampleStats const expected" << *this << ";\n";
}

//---------------------------------------------------------------------------//
/*!
 * Print to a stream.
 */
std::ostream& operator<<(std::ostream& os, SampleStats const& ss)
{
    os << "{" << repr(ss.min()) << ", " << repr(ss.max()) << ", "
       << repr(ss.mean()) << ", " << repr(ss.stdev()) << ", "
       << repr(ss.count()) << "}";
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Test distribution equivalence.
 */
::testing::AssertionResult IsRefEqual(char const* expected_expr,
                                      char const* actual_expr,
                                      char const* conf_expr,
                                      SampleStats const& e,
                                      SampleStats const& a,
                                      SampleStatsTolerance tol)
{
    CELER_EXPECT(tol.mean_confidence > 0 && tol.mean_confidence < 1);
    CELER_EXPECT(tol.stdev_tolerance > 0 && tol.stdev_tolerance < 1);
    auto result = ::testing::AssertionSuccess();
    auto fail = [&]() -> ::testing::AssertionResult& {
        if (result)
        {
            result = ::testing::AssertionFailure();
            result << "Expected: (" << expected_expr << ") == (" << actual_expr
                   << "):\n";
        }
        else
        {
            result << '\n';
        }
        return result;
    };

    SoftEqual<> soft_eq{tol.stdev_tolerance};

    if (a.min() < e.min() || a.max() > e.max()
        || !e.probably_equal(a, tol.mean_confidence)
        || !soft_eq(a.stdev(), e.stdev()))
    {
        fail() << "Mismatch in stats outside of confidence " << conf_expr
               << " (" << tol.mean_confidence << "):\n"
               << "  Expected: {min=" << e.min() << ", max=" << e.max()
               << ", mean=" << e.mean() << ", stdev=" << e.stdev()
               << ", count=" << e.count() << "}\n"
               << "  Actual:   {min=" << a.min() << ", max=" << a.max()
               << ", mean=" << a.mean() << ", stdev=" << a.stdev()
               << ", count=" << a.count()
               << "}: z-score: " << e.calc_z_score(a);
    }

    return result;
}
}  // namespace test
}  // namespace celeritas
