//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/interactor/WavelengthShiftInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/random/distribution/PoissonDistribution.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"
#include "celeritas/optical/WavelengthShiftData.hh"
#include "celeritas/phys/InteractionUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample state change and number of secondaries from a WLS interaction.
 *
 * The number of photons is sampled from a Poisson distribution. The secondary
 * photons are sampled later by the \c WavelengthShiftGenerator.
 *
 * \todo Initialize the first secondary directly in the parent's track slot.
 */
class WavelengthShiftInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    WavelengthShiftInteractor(NativeCRef<WavelengthShiftData> const& shared,
                              ParticleTrackView const& particle,
                              SimTrackView const& sim,
                              Real3 const& pos,
                              OptMatId const& mat_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    PoissonDistribution<real_type> sample_num_photons_;
    WlsDistributionData distribution_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
WavelengthShiftInteractor::WavelengthShiftInteractor(
    NativeCRef<WavelengthShiftData> const& shared,
    ParticleTrackView const& particle,
    SimTrackView const& sim,
    Real3 const& pos,
    OptMatId const& mat_id)
    : sample_num_photons_(shared.wls_record[mat_id].mean_num_photons)
{
    CELER_EXPECT(mat_id);

    distribution_.energy = particle.energy();
    distribution_.time = sim.time();
    distribution_.position = pos;
    distribution_.primary = sim.primary_id();
    distribution_.material = mat_id;
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sampling the wavelength shift (WLS) photons.
 */
template<class Engine>
CELER_FUNCTION Interaction WavelengthShiftInteractor::operator()(Engine& rng)
{
    // Sample the number of photons generated from WLS.
    Interaction result = Interaction::from_absorption();
    size_type num_photons = sample_num_photons_(rng);
    if (num_photons > 0)
    {
        result.distribution = distribution_;
        result.distribution.num_photons = num_photons;
        CELER_ASSERT(result.distribution);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
