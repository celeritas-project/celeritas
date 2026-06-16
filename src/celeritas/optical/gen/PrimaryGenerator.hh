//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/random/data/DistributionData.hh"
#include "corecel/random/distribution/DistributionVisitor.hh"
#include "celeritas/phys/InteractionUtils.hh"

#include "PrimaryGeneratorData.hh"
#include "../TrackInitializer.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical photons from user-configurable distributions.
 *
 * This samples a user-specified number of photons from user-configurable
 * distributions specified in \c celeritas::inp::OpticalPrimaryGenerator .
 *
 * \internal The \c DistributionVisitor is responsible for managing the
 * std::variant-like behavior of this class by mapping distribution IDs to
 * type-deleted data.
 */
class PrimaryGenerator
{
  public:
    // Construct from distribution data
    inline CELER_FUNCTION
    PrimaryGenerator(NativeCRef<DistributionParamsData> const& params,
                     PrimaryDistributionData const& data);

    // Sample an optical photon from the distributions
    template<class Generator>
    inline CELER_FUNCTION optical::TrackInitializer operator()(Generator& rng);

  private:
    NativeCRef<DistributionParamsData> const& params_;
    PrimaryDistributionData const& data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical materials and distribution parameters.
 */
CELER_FUNCTION
PrimaryGenerator::PrimaryGenerator(
    NativeCRef<DistributionParamsData> const& params,
    PrimaryDistributionData const& data)
    : params_(params), data_(data)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample an optical photon from the energy, angular and spatial distributions.
 */
template<class Generator>
CELER_FUNCTION optical::TrackInitializer
PrimaryGenerator::operator()(Generator& rng)
{
    DistributionVisitor visit{params_};

    optical::TrackInitializer result;
    result.energy = units::MevEnergy{sample_with(visit, data_.energy, rng)};
    CELER_ASSERT(result.energy > zero_quantity());
    result.position = sample_with(visit, data_.shape, rng);
    result.direction = sample_with(visit, data_.angle, rng);
    CELER_ASSERT(is_soft_unit_vector(result.direction));
    result.primary = {};  // No associated Geant4 primary
    result.polarization = TransversePolarizationSampler{result.direction}(rng);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
