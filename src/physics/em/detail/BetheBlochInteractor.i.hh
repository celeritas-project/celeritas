//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheBlochInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "random/distributions/UniformRealDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION BetheBlochInteractor::BetheBlochInteractor(
    const BetheBlochInteractorPointers& shared,
    const ParticleTrackView&            particle,
    const Real3&                        inc_direction,
    StackAllocator<Secondary>&          allocate,
    MaterialView&                       material)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , material_(material)
    , inc_mass_(particle.mass().value())
{
    CELER_EXPECT(inc_energy_ >= this->min_incident_energy()
                 && inc_energy_ <= this->max_incident_energy());
    CELER_EXPECT(particle.particle_id() == shared_.mu_minus_id 
                 || particle.particle_id() == shared_.mu_plus_id); 
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the Bethe-Bloch model.
 */
template<class Engine>
CELER_FUNCTION Interaction BetheBlochInteractor::operator()(Engine& rng)
{
    // Allocate space for gamma
    Secondary* secondaries = this->allocate_(1); 
    if (secondaries == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    real_type kinetic_energy = inc_energy_.value();
    
    real_type tmin = min(kinetic_energy, this->min_incident_energy().value());

    ElementView element = material_.element_view(celeritas::ElementComponentId{0});

    real_type func1 = tmin * this->differential_cross_section(tmin, element);

    real_type ln_epsilon, epsilon;
    real_type func2;

    real_type xmin = std::log(tmin);
    real_type xmax = std::log(kinetic_energy / tmin);
    UniformRealDistribution<real_type> p;

    do
    {
        ln_epsilon = xmin + p(rng) * xmax;
        epsilon = std::exp(ln_epsilon);
        func2 = epsilon * this->differential_cross_section(epsilon, element);
    } while(func2 < func1 * p(rng));

    real_type gamma_energy = epsilon;

    // Sample secondary direction.
    UniformRealDistribution<real_type> phi(0, 2 * constants::pi);

    real_type cost = this->sample_cos_theta(gamma_energy, rng);
    Real3 gamma_dir = rotate(from_spherical(cost, phi(rng)), inc_direction_);

    real_type tot_momentum = std::sqrt(gamma_energy *
                                    (gamma_energy + 2.0 * inc_mass_.value()));
    Real3 inc_direction;
    for (int i = 0; i < 3; i++)
    {
        inc_direction[i] = tot_momentum * inc_direction_[i]
                                        - gamma_energy * gamma_dir[i];
    }
    normalize_direction(&inc_direction);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action      = Action::scattered;
    result.energy      = units::MevEnergy{kinetic_energy - gamma_energy};
    result.direction   = inc_direction;
    result.secondaries = {secondaries, 1};

    // Save outgoing secondary data
    secondaries[0].particle_id = shared_.gamma_id;
    secondaries[0].energy      = units::MevEnergy{gamma_energy};
    secondaries[0].direction   = gamma_dir;

    return result;
}

template<class Engine>
CELER_FUNCTION real_type
BetheBlochInteractor::sample_cos_theta(real_type gamma_energy,
                                       Engine& rng)
{
    real_type mass = inc_mass_.value();
    real_type gam  = 1.0 + inc_energy_.value() / mass;
    real_type rmax = gam * constants::pi * 0.5
                         * min(1.0, gam * mass / gamma_energy - 1);

    UniformRealDistribution<real_type> xi;
    real_type x = xi(rng) * ipow<2>(rmax) / (1 + ipow<2>(rmax));

    return std::cos(std::sqrt(x / (1.0 - x)) / gam);
}

CELER_FUNCTION real_type
BetheBlochInteractor::differential_cross_section(real_type gamma_energy,
                                                 ElementView element)
{

    real_type dxsection = 0.0;
    real_type ep_max = inc_energy_.value();

    if (gamma_energy >= ep_max)
    {
        return dxsection;
    }

    int Z = element.atomic_number();
    real_type A = element.atomic_mass().value();
    real_type mass = inc_mass_.value();
    real_type sqrte = 1.648721271;
    real_type E = mass + ep_max;
    real_type v = gamma_energy / E;
    real_type delta = ipow<2>(mass) * v / (2.0 * E - gamma_energy);

    real_type dn, dnstar, b, b1;
    dn = 1.54 * std::pow(A,0.27);

    if (Z == 1)
    {
        dnstar = dn;
        b = 202.4;
        b1 = 446.0;
    }
    else
    {
        dnstar = dn / std::pow(dn,1.0 / Z);
        b = 183.0;
        b1 = 1429.0;
    }

    real_type phi_n;
    real_type Z13 = 1.0 / element.cbrt_z();
    real_type electron_m = constants::electron_mass;

    phi_n = std::log(b * Z13 * (mass + delta * (dnstar * sqrte - 2.0))
                     / (dnstar * (electron_m + delta * sqrte* b * Z13)));

    if (phi_n < 0)
    {
        phi_n = 0.0;
    }

    real_type phi_e = 0.0;
    real_type Z23 = 1 / Z * Z13;
    real_type ep_max1 = E / (1.0 + 0.5 * ipow<2>(mass) / electron_m * E);

    if (gamma_energy < ep_max1)
    {
        phi_e = std::log(b1 * Z23 * mass / (1.0 + delta * mass
                         / (ipow<2>(electron_m) * sqrte))
                         * (electron_m + delta * sqrte * b1 * Z23));

        if (phi_e < 0.0)
        {
            phi_e = 0.0;
        }
    }

    real_type alpha = constants::alpha_fine_structure;
    real_type na = constants::na_avogadro;
    real_type re = constants::r_electron;

    dxsection = 16.0 * alpha * na * ipow<2>(electron_m) * ipow<2>(re) * Z
                    * (Z * phi_n + phi_e) * (1 - v * (1 - 0.75 * v))
                    / (3.0 * ipow<2>(mass) * gamma_energy * A);
    return dxsection;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
