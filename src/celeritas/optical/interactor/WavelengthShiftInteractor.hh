//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/interactor/WavelengthShiftInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/random/distribution/ExponentialDistribution.hh"
#include "corecel/random/distribution/PoissonDistribution.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "geocel/random/IsotropicDistribution.hh"
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
 * Apply the wavelength shift (WLS) to optical photons.
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
class WavelengthShiftInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using ParamsRef = NativeCRef<WavelengthShiftData>;
    using SecondaryAllocator = StackAllocator<TrackInitializer>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    WavelengthShiftInteractor(ParamsRef const& shared,
                              ParticleTrackView const& particle,
                              OpticalMaterialId const& mat_id,
                              SecondaryAllocator& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Incident photon energy
    Energy const inc_energy_;
    // Sample distributions
    PoissonDistribution<real_type> sample_num_photons_;
    ExponentialDistribution<real_type> sample_time_;
    // Grid calculators
    NonuniformGridCalculator calc_cdf_;
    // Allocate space for secondary particles
    SecondaryAllocator& allocate_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
WavelengthShiftInteractor::WavelengthShiftInteractor(
    ParamsRef const& shared,
    ParticleTrackView const& particle,
    OpticalMaterialId const& mat_id,
    SecondaryAllocator& allocate)
    : inc_energy_(particle.energy())
    , sample_num_photons_(shared.wls_record[mat_id].mean_num_photons)
    , sample_time_(real_type{1} / shared.wls_record[mat_id].time_constant)
    , calc_cdf_(shared.energy_cdf[mat_id], shared.reals)
    , allocate_(allocate)
{
    CELER_EXPECT(inc_energy_.value() > calc_cdf_.grid().front());
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
    /*!
     * Sample the number of photons generated from WLS.
     *
     * \todo if this is nonzero and allocation fails, we lose reproducibility.
     */
    unsigned int num_photons = sample_num_photons_(rng);

    if (num_photons == 0)
    {
        // Return absorbed photon without reemitted optical photons
        return Interaction::from_absorption();
    }

    // Allocate space for reemitted optical photons - Note: the reproducibility
    // is not guaranteed in the case of an out-of-memory error
    TrackInitializer* secondaries = allocate_(num_photons);

    if (secondaries == nullptr)
    {
        // Failed to allocate space
        return Interaction::from_failure();
    }

    // Sample wavelength shifted optical photons
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, num_photons};

    IsotropicDistribution sample_direction{};
    NonuniformGridCalculator calc_energy = calc_cdf_.make_inverse();
    for (size_type i : range(num_photons))
    {
        // Sample the emitted energy from the inverse cumulative distribution
        // TODO: add CDF sampler; see
        // https://github.com/celeritas-project/celeritas/pull/1507/files#r1844973621
        real_type energy = calc_energy(generate_canonical(rng));
        if (CELER_UNLIKELY(energy > inc_energy_.value()))
        {
            // Sample a restricted energy below the incident photon energy
            real_type cdf_max = calc_cdf_(inc_energy_.value());
            UniformRealDistribution<real_type> sample_cdf(0, cdf_max);
            energy = calc_energy(sample_cdf(rng));
        }
        CELER_ENSURE(energy < inc_energy_.value());
        secondaries[i].energy = Energy{energy};

        // Sample the emitted photon (incoherent) direction and polarization
        secondaries[i].direction = sample_direction(rng);
        secondaries[i].polarization
            = ExitingDirectionSampler{0, secondaries[i].direction}(rng);

        CELER_ENSURE(is_soft_unit_vector(secondaries[i].direction));
        CELER_ENSURE(is_soft_unit_vector(secondaries[i].polarization));
        CELER_ENSURE(soft_zero(dot_product(secondaries[i].direction,
                                           secondaries[i].polarization)));

        // Sample the delta time (based on the exponential relaxation)
        secondaries[i].time = sample_time_(rng);
    }

    CELER_ENSURE(result.action == Interaction::Action::absorbed);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
