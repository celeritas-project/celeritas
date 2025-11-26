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
 *
 * \todo There are a couple places in the code where we resample the
 * polarization if orthogonality fails: possibly add a helper function to
 * reduce duplication
 */
template<class Generator>
CELER_FUNCTION optical::TrackInitializer
PrimaryGenerator::operator()(Generator& rng)
{
    DistributionVisitor visit{params_};

    optical::TrackInitializer result;
    result.energy = units::MevEnergy{sample_with(visit, data_.energy, rng)};
    result.position = sample_with(visit, data_.shape, rng);
    result.direction = sample_with(visit, data_.angle, rng);
    do
    {
        result.polarization = ExitingDirectionSampler{0, result.direction}(rng);
    } while (CELER_UNLIKELY(
        !is_soft_orthogonal(result.polarization, result.direction)));

    CELER_ENSURE(result.energy > zero_quantity());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
