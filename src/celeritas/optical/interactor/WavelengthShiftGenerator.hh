//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/interactor/WavelengthShiftGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/random/distribution/ExponentialDistribution.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/WavelengthShiftData.hh"
#include "celeritas/phys/InteractionUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical photons from the wavelength shift process.
 *
 * A wavelength shifter absorbs an incident light and reemits secondary lights
 * isotropically at longer wavelengths. It usually shifts the ultraviolet
 * region of the radiation spectrum to the visible region, which enhances the
 * light collection or reduces the self-absorption of the optical production.
 * The number of the reemitted photons follows the Poisson distribution with
 * the mean number of the characteristic light production, which depends on the
 * optical property of wavelength shifters. The polarization of the reemitted
 * lights is assumed to be incoherent with respect to the polarization of the
 * primary optical photon.
 *
 * \note This performs the same sampling routine as in the G4OpWLS class of
 * the Geant4 release 11.2.
 */
class WavelengthShiftGenerator
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    WavelengthShiftGenerator(NativeCRef<WavelengthShiftData> const& shared,
                             WlsDistributionData const& distribution);

    // Sample a WLS photon
    template<class Engine>
    inline CELER_FUNCTION TrackInitializer operator()(Engine& rng);

  private:
    //// TYPES ////

    using Energy = units::MevEnergy;

    //// DATA ////

    WlsDistributionData const& distribution_;
    real_type time_constant_;
    WlsTimeProfile time_profile_;
    NonuniformGridCalculator calc_cdf_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
WavelengthShiftGenerator::WavelengthShiftGenerator(
    NativeCRef<WavelengthShiftData> const& shared,
    WlsDistributionData const& distribution)
    : distribution_(distribution)
    , time_constant_(shared.wls_record[distribution_.material].time_constant)
    , time_profile_(shared.time_profile)
    , calc_cdf_(shared.energy_cdf[distribution_.material], shared.reals)
{
    CELER_EXPECT(distribution_);
    CELER_EXPECT(distribution_.energy.value() > calc_cdf_.grid().front());
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sampling the wavelength shift (WLS) photons.
 */
template<class Engine>
CELER_FUNCTION TrackInitializer WavelengthShiftGenerator::operator()(Engine& rng)
{
    // Sample wavelength shifted optical photon
    TrackInitializer result;

    // Sample the emitted energy from the inverse cumulative distribution
    // TODO: add CDF sampler; see
    // https://github.com/celeritas-project/celeritas/pull/1507/files#r1844973621
    NonuniformGridCalculator calc_energy = calc_cdf_.make_inverse();
    real_type energy = calc_energy(generate_canonical(rng));
    if (CELER_UNLIKELY(energy > value_as<Energy>(distribution_.energy)))
    {
        // Sample a restricted energy below the incident photon energy
        real_type cdf_max = calc_cdf_(value_as<Energy>(distribution_.energy));
        UniformRealDistribution<real_type> sample_cdf(0, cdf_max);
        energy = calc_energy(sample_cdf(rng));
    }
    CELER_ENSURE(energy < value_as<Energy>(distribution_.energy));
    result.energy = Energy{energy};

    // Use the post-step position
    result.position = distribution_.position;

    // Sample the emitted photon (incoherent) direction and polarization
    result.direction = IsotropicDistribution{}(rng);
    result.polarization = ExitingDirectionSampler{0, result.direction}(rng);

    // Sample the delta time (based on the exponential relaxation)
    result.time
        = distribution_.time
          + (time_profile_ == WlsTimeProfile::delta
                 ? time_constant_
                 : ExponentialDistribution(real_type{1} / time_constant_)(rng));
    result.primary = distribution_.primary;

    CELER_ENSURE(is_soft_unit_vector(result.polarization));
    CELER_ENSURE(is_soft_orthogonal(result.direction, result.polarization));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
