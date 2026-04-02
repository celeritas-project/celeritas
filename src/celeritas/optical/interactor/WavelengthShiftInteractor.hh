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
 * \todo See if initializing the first photon directly in this track slot
 * improves performance
 */
class WavelengthShiftInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using DistId = ItemId<WlsDistributionData>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    WavelengthShiftInteractor(NativeCRef<WavelengthShiftData> const& shared,
                              NativeRef<WlsGeneratorStateData> data,
                              ParticleTrackView const& particle,
                              SimTrackView const& sim,
                              Real3 const& pos,
                              OptMatId const& mat_id,
                              DistId distribution_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    NativeRef<WlsGeneratorStateData> data_;
    PoissonDistribution<real_type> sample_num_photons_;
    DistId distribution_id_;
    units::MevEnergy emission_threshold_;
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
    NativeRef<WlsGeneratorStateData> data,
    ParticleTrackView const& particle,
    SimTrackView const& sim,
    Real3 const& pos,
    OptMatId const& mat_id,
    DistId distribution_id)
    : data_(data)
    , sample_num_photons_(shared.wls_record[mat_id].mean_num_photons)
    , distribution_id_(distribution_id)
    , emission_threshold_(shared.reals[shared.energy_cdf[mat_id].grid.front()])
{
    CELER_EXPECT(data_);
    CELER_EXPECT(mat_id);
    CELER_EXPECT(distribution_id_ < data_.distributions.size());
    CELER_EXPECT(!data_.distributions[distribution_id_]);

    distribution_.type = shared.type;
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
    Interaction result = Interaction::from_absorption();

    if (distribution_.energy <= emission_threshold_)
    {
        // If the incident particle energy is below the lower bound of the
        // emitted energy sampling grid, don't emit any photons
        distribution_.num_photons = 0;
    }
    else
    {
        // Sample the number of photons generated from WLS.
        distribution_.num_photons = sample_num_photons_(rng);
    }
    CELER_ASSERT(distribution_ || distribution_.num_photons == 0);
    data_.distributions[distribution_id_] = distribution_;

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
