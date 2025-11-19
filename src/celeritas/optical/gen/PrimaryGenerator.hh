//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "geocel/random/IsotropicDistribution.hh"

#include "../TrackInitializer.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical photons from user-configurable distributions.
 */
class PrimaryGenerator
{
  public:
    // Construct from distribution data
    inline CELER_FUNCTION PrimaryGenerator(PrimaryDistributionData const& data);

    // Sample an optical photon from the distributions
    template<class Generator>
    inline CELER_FUNCTION optical::TrackInitializer operator()(Generator& rng);

  private:
    //// DATA ////

    PrimaryDistributionData const& data_;
    IsotropicDistribution<real_type> sample_polarization_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical materials and distribution parameters.
 */
CELER_FUNCTION
PrimaryGenerator::PrimaryGenerator(PrimaryDistributionData const& data)
    : data_(data)
{
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample an optical photon from the distributions.
 *
 * \todo There are a couple places in the code where we resample the
 * polarization if orthogonality fails: possibly add a helper function to
 * reduce duplication
 */
template<class Generator>
CELER_FUNCTION optical::TrackInitializer
PrimaryGenerator::operator()(Generator& rng)
{
    optical::TrackInitializer result;
    result.energy = data_.sample_energy(rng);
    result.position = data_.sample_position(rng);
    result.direction = data_.sample_direction(rng);
    do
    {
        result.polarization = make_unit_vector(
            make_orthogonal(sample_polarization_(rng), result.direction));
    } while (CELER_UNLIKELY(
        !is_soft_orthogonal(result.polarization, result.direction)));

    CELER_ENSURE(result.energy > zero_quantity());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
