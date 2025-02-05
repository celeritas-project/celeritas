//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/distribution/NormalDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Constants.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Constants.hh"

#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from a normal distribution.
 *
 * This uses the Box-Muller transform to generate pairs of independent,
 * normally distributed random numbers, returning them one at a time. Two
 * random numbers uniformly distributed on [0, 1] are mapped to two
 * independent, standard, normally distributed samples using the relations:
 * \f[
  x_1 = \sqrt{-2 \ln \xi_1} \cos(2 \pi \xi_2)
  x_2 = \sqrt{-2 \ln \xi_1} \sin(2 \pi \xi_2)
   \f]
 */
template<class RealType = ::celeritas::real_type>
class NormalDistribution
{
    static_assert(std::is_floating_point_v<RealType>);

  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Construct with mean and standard deviation
    inline CELER_FUNCTION NormalDistribution(real_type mean, real_type stddev);

    //! Construct with unit deviation
    explicit CELER_FUNCTION NormalDistribution(real_type mean)
        : NormalDistribution{mean, 1}
    {
    }

    //! Construct with unit deviation and zero mean
    CELER_FUNCTION NormalDistribution() : NormalDistribution{0, 1} {}

    // Initialize with parameters but not spare values
    inline CELER_FUNCTION NormalDistribution(NormalDistribution const& other);

    // Reset spare value of other distribution
    inline CELER_FUNCTION
    NormalDistribution(NormalDistribution&& other) noexcept;

    // Keep spare value but change distribution
    inline CELER_FUNCTION NormalDistribution&
    operator=(NormalDistribution const&);

    // Possibly use spare value, change distribution
    inline CELER_FUNCTION NormalDistribution&
    operator=(NormalDistribution&&) noexcept;

    // Default destructor (rule of 5)
    ~NormalDistribution() = default;

    // Sample a random number according to the distribution
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);

  private:
    // Distribution properties
    real_type mean_;
    real_type stddev_;

    // Intermediate samples
    real_type spare_{};
    bool has_spare_{false};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with mean and standard deviation.
 */
template<class RealType>
CELER_FUNCTION
NormalDistribution<RealType>::NormalDistribution(real_type mean,
                                                 real_type stddev)
    : mean_(mean), stddev_(stddev)
{
    CELER_EXPECT(stddev > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize with parameters but not spare values.
 */
template<class RealType>
CELER_FUNCTION
NormalDistribution<RealType>::NormalDistribution(NormalDistribution const& other)
    : mean_{other.mean_}, stddev_{other.stddev_}
{
}

//---------------------------------------------------------------------------//
/*!
 * Reset spare value of other distribution.
 */
template<class RealType>
CELER_FUNCTION NormalDistribution<RealType>::NormalDistribution(
    NormalDistribution&& other) noexcept
    : mean_{other.mean_}
    , stddev_{other.stddev_}
    , spare_{other.spare_}
    , has_spare_{other.has_spare_}
{
    other.has_spare_ = false;
}

//---------------------------------------------------------------------------//
/*!
 * Keep spare value but change distribution.
 */
template<class RealType>
CELER_FUNCTION NormalDistribution<RealType>&
NormalDistribution<RealType>::operator=(NormalDistribution const& other)
{
    mean_ = other.mean_;
    stddev_ = other.stddev_;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Possibly use spare value, change distribution.
 */
template<class RealType>
CELER_FUNCTION NormalDistribution<RealType>&
NormalDistribution<RealType>::operator=(NormalDistribution&& other) noexcept
{
    mean_ = other.mean_;
    stddev_ = other.stddev_;
    if (!has_spare_ && other.has_spare_)
    {
        spare_ = other.spare_;
        has_spare_ = other.has_spare_;
        other.has_spare_ = false;
    }
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class RealType>
template<class Generator>
CELER_FUNCTION auto
NormalDistribution<RealType>::operator()(Generator& rng) -> result_type
{
    if (has_spare_)
    {
        has_spare_ = false;
        return std::fma(spare_, stddev_, mean_);
    }

    // TODO: use sincospi here
    constexpr auto twopi = static_cast<RealType>(2 * constants::pi);
    real_type theta = twopi * generate_canonical<RealType>(rng);
    real_type r = std::sqrt(-2 * std::log(generate_canonical<RealType>(rng)));
    spare_ = r * std::cos(theta);
    has_spare_ = true;
    return std::fma(r * std::sin(theta), stddev_, mean_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
