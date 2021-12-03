//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "PhysicsConstants.hh"
#include "random/distributions/UniformRealDistribution.hh"
#include "SBEnergyDistHelper.hh"
#include "SBEnergyDistribution.hh"
#include "SBPositronXsCorrector.hh"
#include "TsaiUrbanDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared/device and state data.
 *
 * The incident particle must be within the model's valid energy range. this
 * must be handled in code *before* the interactor is constructed.
 */
SeltzerBergerInteractor::SeltzerBergerInteractor(
    const SeltzerBergerNativeRef& shared,
    const ParticleTrackView&      particle,
    const Real3&                  inc_direction,
    const CutoffView&             cutoffs,
    StackAllocator<Secondary>&    allocate,
    const MaterialView&           material,
    const ElementComponentId&     elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(inc_direction)
    , inc_particle_is_electron_(particle.particle_id() == shared_.ids.electron)
    , gamma_cutoff_(cutoffs.energy(shared.ids.gamma))
    , allocate_(allocate)
    , material_(material)
    , elcomp_id_(elcomp_id)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.electron
                 || particle.particle_id() == shared_.ids.positron);
    CELER_EXPECT(gamma_cutoff_ > zero_quantity());
}

//---------------------------------------------------------------------------//
/*!
 * Bremsstrahlung using the Seltzer-Berger model.
 *
 * See section 10.2.1 of the Geant physics reference 10.6.
 */
template<class Engine>
CELER_FUNCTION Interaction SeltzerBergerInteractor::operator()(Engine& rng)
{
    // Check if secondary can be produced. If not, this interaction cannot
    // happen and the incident particle must undergo an energy loss process
    // instead.
    if (gamma_cutoff_ > inc_energy_)
    {
        return Interaction::from_unchanged(inc_energy_, inc_direction_);
    }

    // Allocate space for the brems photon
    Secondary* secondaries = this->allocate_(1);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for the secondary
        return Interaction::from_failure();
    }

    // Density correction
    real_type density_factor = material_.electron_density() * migdal_constant;
    real_type total_energy_val = inc_energy_.value()
                                 + shared_.electron_mass.value();
    real_type density_correction = density_factor * ipow<2>(total_energy_val);

    // Outgoing photon secondary energy sampler
    Energy gamma_exit_energy;
    {
        // Helper class preprocesses cross section bounds and calculates
        // distribution
        SBEnergyDistHelper sb_helper(
            shared_.differential_xs,
            inc_energy_,
            material_.element_id(elcomp_id_),
            SBEnergyDistHelper::EnergySq{density_correction},
            gamma_cutoff_);

        if (inc_particle_is_electron_)
        {
            // Rejection sample without modifying cross section
            SBEnergyDistribution<SBElectronXsCorrector> sample_gamma_energy(
                sb_helper, {});
            gamma_exit_energy = sample_gamma_energy(rng);
        }
        else
        {
            SBEnergyDistribution<SBPositronXsCorrector> sample_gamma_energy(
                sb_helper,
                {shared_.electron_mass,
                 material_.element_view(elcomp_id_),
                 gamma_cutoff_,
                 inc_energy_});
            gamma_exit_energy = sample_gamma_energy(rng);
        }
    }

    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.action = Action::scattered;
    result.energy
        = units::MevEnergy{inc_energy_.value() - gamma_exit_energy.value()};
    result.direction           = inc_direction_;
    result.secondaries         = {secondaries, 1};
    secondaries[0].particle_id = shared_.ids.gamma;
    secondaries[0].energy      = gamma_exit_energy;

    // Generate exiting gamma direction from isotropic azimuthal
    // angle and TsaiUrbanDistribution for polar angle
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    TsaiUrbanDistribution              sample_gamma_angle(inc_energy_,
                                             shared_.electron_mass);
    real_type                          cost = sample_gamma_angle(rng);
    secondaries[0].direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    // Update parent particle direction
    for (unsigned int i : range(3))
    {
        real_type inc_momentum_i   = inc_momentum_.value() * inc_direction_[i];
        real_type gamma_momentum_i = result.secondaries[0].energy.value()
                                     * result.secondaries[0].direction[i];
        result.direction[i] = inc_momentum_i - gamma_momentum_i;
    }
    normalize_direction(&result.direction);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
