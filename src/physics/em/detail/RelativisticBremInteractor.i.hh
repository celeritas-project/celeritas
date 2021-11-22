//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Algorithms.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/UniformRealDistribution.hh"
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
RelativisticBremInteractor::RelativisticBremInteractor(
    const RelativisticBremNativeRef& shared,
    const ParticleTrackView&         particle,
    const Real3&                     direction,
    const CutoffView&                cutoffs,
    StackAllocator<Secondary>&       allocate,
    const MaterialView&              material,
    const ElementComponentId&        elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_direction_(direction)
    , gamma_cutoff_(cutoffs.energy(shared.ids.gamma))
    , allocate_(allocate)
    , dxsec_(shared, particle, material, elcomp_id)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.electron
                 || particle.particle_id() == shared_.ids.positron);
    CELER_EXPECT(gamma_cutoff_.value() > 0);

    // Valid energy region of the relativistic e-/e+ Bremsstrahlung model
    CELER_EXPECT(inc_energy_ > shared_.low_energy_limit());
}

//---------------------------------------------------------------------------//
/*!
 * Sample the production of photons using the G4eBremsstrahlungRelModel
 * of Geant4 6.10.
 */
template<class Engine>
CELER_FUNCTION Interaction RelativisticBremInteractor::operator()(Engine& rng)
{
    // Min and max kinetic energy limits for sampling the secondary photon
    Energy tmin = min(gamma_cutoff_, inc_energy_);
    Energy tmax = min(shared_.high_energy_limit(), inc_energy_);
    if (tmin >= tmax)
    {
        return Interaction::from_unchanged(inc_energy_, inc_direction_);
    }

    real_type density_corr = dxsec_.density_correction();
    real_type xmin         = std::log(ipow<2>(tmin.value()) + density_corr);
    real_type xrange = std::log(ipow<2>(tmax.value()) + density_corr) - xmin;

    real_type gamma_energy{0};
    real_type dsigma{0};

    do
    {
        gamma_energy = std::sqrt(max(
            0.0,
            std::exp(xmin + generate_canonical(rng) * xrange) - density_corr));
        dsigma       = dxsec_(gamma_energy);
    } while (dsigma < dxsec_.maximum_value() * generate_canonical(rng));

    // Allocate space for the brems photon
    Secondary* secondaries = this->allocate_(1);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for the secondary
        return Interaction::from_failure();
    }

    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.action      = Action::scattered;
    result.energy      = units::MevEnergy{inc_energy_.value() - gamma_energy};
    result.secondaries = {secondaries, 1};
    secondaries[0].particle_id = shared_.ids.gamma;
    secondaries[0].energy      = units::MevEnergy{gamma_energy};

    // angular distribution: G4ModifiedTsai

    // Generate exiting gamma direction from isotropic azimuthal
    // angle and TsaiUrbanDistribution for polar angle
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    //    TsaiUrbanDistribution sample_gamma_angle(secondaries[0].energy,
    TsaiUrbanDistribution sample_gamma_angle(inc_energy_,
                                             shared_.electron_mass);
    real_type             cost = sample_gamma_angle(rng);
    secondaries[0].direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    // Update parent particle direction
    real_type inc_momentum = std::sqrt(
        inc_energy_.value()
        * (inc_energy_.value() + 2 * shared_.electron_mass.value()));
    for (unsigned int i : range(3))
    {
        real_type inc_momentum_i   = inc_momentum * inc_direction_[i];
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
