//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Algorithms.hh"
#include "base/Constants.hh"
#include "random/distributions/GenerateCanonical.hh"
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
 * Construct with shared and state data.
 */
CELER_FUNCTION
CombinedBremInteractor::CombinedBremInteractor(
    const CombinedBremNativeRef& shared,
    const ParticleTrackView&     particle,
    const Real3&                 direction,
    const CutoffView&            cutoffs,
    StackAllocator<Secondary>&   allocate,
    const MaterialView&          material,
    const ElementComponentId&    elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(direction)
    , gamma_cutoff_(cutoffs.energy(shared.rb_data.ids.gamma))
    , allocate_(allocate)
    , material_(material)
    , elcomp_id_(elcomp_id)
    , rb_dxsec_(shared.rb_data, particle, material, elcomp_id)
    , is_electron_(particle.particle_id() == shared.rb_data.ids.electron)
    , is_relativistic_(particle.energy() > shared.rb_data.low_energy_limit())
{
    CELER_EXPECT(is_electron_
                 || particle.particle_id() == shared_.rb_data.ids.positron);
    CELER_EXPECT(gamma_cutoff_.value() > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the production of photons using the combined model.
 */
template<class Engine>
CELER_FUNCTION Interaction CombinedBremInteractor::operator()(Engine& rng)
{
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

    // Sample the bremsstrahlung energy
    Energy energy = (!is_relativistic_) ? sample_energy_sb<Engine>(rng)
                                        : sample_energy_rb<Engine>(rng);

    // Update kinematics of the final state
    Interaction result = update_state<Engine>(rng, energy, secondaries);

    return result;
}

template<class Engine>
CELER_FUNCTION auto CombinedBremInteractor::sample_energy_sb(Engine& rng)
    -> Energy
{
    // Outgoing photon secondary energy sampler
    Energy gamma_energy;
    {
        // Helper class preprocesses cross section bounds and calculates
        // distribution
        SBEnergyDistHelper sb_helper(
            shared_.sb_differential_xs,
            inc_energy_,
            material_.element_id(elcomp_id_),
            SBEnergyDistHelper::EnergySq{rb_dxsec_.density_correction()},
            gamma_cutoff_);

        if (is_electron_)
        {
            // Rejection sample without modifying cross section
            SBEnergyDistribution<SBElectronXsCorrector> sample_gamma_energy(
                sb_helper, {});
            gamma_energy = sample_gamma_energy(rng);
        }
        else
        {
            SBEnergyDistribution<SBPositronXsCorrector> sample_gamma_energy(
                sb_helper,
                {shared_.rb_data.electron_mass,
                 material_.element_view(elcomp_id_),
                 gamma_cutoff_,
                 inc_energy_});
            gamma_energy = sample_gamma_energy(rng);
        }
    }

    return gamma_energy;
}

template<class Engine>
CELER_FUNCTION auto CombinedBremInteractor::sample_energy_rb(Engine& rng)
    -> Energy
{
    // Min and max kinetic energy limits for sampling the secondary photon
    Energy tmin = min(gamma_cutoff_, inc_energy_);
    Energy tmax = min(shared_.rb_data.high_energy_limit(), inc_energy_);

    real_type density_corr = rb_dxsec_.density_correction();
    real_type xmin         = std::log(ipow<2>(tmin.value()) + density_corr);
    real_type xrange = std::log(ipow<2>(tmax.value()) + density_corr) - xmin;

    real_type gamma_energy{0};
    real_type dsigma{0};

    do
    {
        gamma_energy = std::sqrt(max(
            real_type(0),
            std::exp(xmin + generate_canonical(rng) * xrange) - density_corr));
        dsigma       = rb_dxsec_(gamma_energy);
    } while (dsigma < rb_dxsec_.maximum_value() * generate_canonical(rng));

    return units::MevEnergy{gamma_energy};
}

template<class Engine>
CELER_FUNCTION Interaction CombinedBremInteractor::update_state(
    Engine& rng, const Energy gamma_energy, Secondary* secondaries)
{
    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.action = Action::scattered;
    result.energy
        = units::MevEnergy{inc_energy_.value() - gamma_energy.value()};
    result.secondaries         = {secondaries, 1};
    secondaries[0].particle_id = shared_.rb_data.ids.gamma;
    secondaries[0].energy      = gamma_energy;

    // Generate exiting gamma direction from isotropic azimuthal angle and
    // TsaiUrbanDistribution for polar angle (based on G4ModifiedTsai)
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    TsaiUrbanDistribution              sample_gamma_angle(inc_energy_,
                                             shared_.rb_data.electron_mass);
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
