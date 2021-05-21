//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "TsaiUrbanDistribution.hh"
#include "random/distributions/UniformRealDistribution.hh"

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
    const SeltzerBergerDeviceRef& device_pointers,
    const ParticleTrackView&      particle,
    const Real3&                  inc_direction,
    const CutoffView&             cutoffs,
    StackAllocator<Secondary>&    allocate,
    const MaterialView&           material,
    const ElementId&              element_id)
    : device_pointers_(device_pointers)
    , particle_id_(particle.particle_id())
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(inc_direction)
    , cutoffs_(cutoffs)
    , allocate_(allocate)
    , material_(material)
    , element_id_(element_id)
{
    CELER_EXPECT(particle_id_ == device_pointers_.ids.electron
                 || particle_id_ == device_pointers_.ids.positron);
}

//---------------------------------------------------------------------------//
/*!
 * Pair-production using the Seltzer-Berger model.
 *
 * See section 10.2.1 of the Geant physics reference 10.6.
 */
template<class Engine>
CELER_FUNCTION Interaction SeltzerBergerInteractor::operator()(Engine& rng)
{
    // Check if secondary can be produced. If not, this interaction cannot
    // happen and the incident particle must undergo an energy loss process
    // instead.
    if (cutoffs_.energy(device_pointers_.ids.gamma).value()
        > inc_energy_.value())
    {
        return Interaction::from_unchanged(inc_energy_, inc_direction_);
    }

    // Allocate space for the bremss photon
    Secondary* gamma_secondary = this->allocate_(1);
    if (gamma_secondary == nullptr)
    {
        // Failed to allocate space for the secondary
        return Interaction::from_failure();
    }

    // Density correction
    constexpr auto migdal = 4 * constants::pi * constants::r_electron
                            * ipow<2>(constants::lambdabar_electron);
    real_type density_factor   = material_.electron_density() * migdal;
    real_type total_energy_val = inc_energy_.value()
                                 + device_pointers_.electron_mass;
    real_type density_correction = density_factor * ipow<2>(total_energy_val);

    // Outgoing photon secondary energy sampler
    SBEnergyDistribution sample_gamma_energy{
        device_pointers_,
        inc_energy_,
        element_id_,
        EnergySq{density_correction},
        cutoffs_.energy(device_pointers_.ids.gamma)};
    Energy gamma_exit_energy = sample_gamma_energy(rng);

    // Cross-section scaling for positrons
    if (particle_id_ == device_pointers_.ids.positron)
    {
        SBPositronXsCorrector scale_xs(
            units::MevMass{device_pointers_.electron_mass},
            material_.element_view(ElementComponentId{0}),
            cutoffs_.energy(device_pointers_.ids.gamma),
            inc_energy_);
        /// TODO: integrate XSCorrector into energy distribution sampler
        /// For now, just keep gamma energy unmodified (i.e. do nothing).
    }

    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.action = Action::spawned;
    result.energy
        = units::MevEnergy{inc_energy_.value() - gamma_exit_energy.value()};
    result.direction               = inc_direction_;
    result.secondaries             = {gamma_secondary, 1};
    gamma_secondary[0].particle_id = device_pointers_.ids.gamma;

    // Sample exiting gamma direction
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    real_type                          phi = sample_phi(rng);
    TsaiUrbanDistribution              sample_gamma_angle(
        gamma_exit_energy, units::MevMass{1 / device_pointers_.electron_mass});
    real_type cost = sample_gamma_angle(rng);
    gamma_secondary[0].direction
        = rotate(from_spherical(cost, phi), inc_direction_);

    // Update parent particle direction
    for (unsigned int i : range(3))
    {
        real_type inc_momentum_i   = inc_momentum_.value() * inc_direction_[i];
        real_type gamma_momentum_i = result.secondaries[0].energy.value()
                                     * gamma_secondary[0].direction[i];
        result.secondaries[0].direction[i] = inc_momentum_i - gamma_momentum_i;
    }

    return result;
}

template<class Engine>
CELER_FUNCTION real_type SeltzerBergerInteractor::sample_cos_theta(
    const Energy kinetic_energy, Engine& rng)
{
    real_type umax
        = 2.0 * (1.0 + kinetic_energy.value() / device_pointers_.electron_mass);
    real_type u;
    do
    {
        u = -std::log(generate_canonical(rng) * generate_canonical(rng));
        if (0.25 > generate_canonical(rng))
            u /= 0.625;
        else
            u /= 1.875;
    } while (u > umax);

    return 1.0 - 2.0 * ipow<2>(u / umax);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
