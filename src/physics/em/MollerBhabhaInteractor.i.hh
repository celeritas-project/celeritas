//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/Range.hh"
#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MollerBhabhaInteractor::MollerBhabhaInteractor(
    const MollerBhabhaInteractorPointers& shared,
    const ParticleTrackView&              particle,
    const Real3&                          inc_direction,
    SecondaryAllocatorView&               allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_momentum_(particle.momentum().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{
    CELER_EXPECT(inc_energy_ >= this->min_incident_energy()
                 && inc_energy_ <= this->max_incident_energy());
   //CELER_EXPECT(particle.def_id() == shared_.gamma_id); // XXX
}

//---------------------------------------------------------------------------//
/*!
 * Sample e-e- or e+e- scattering using Moller or Bhabha models, depending on
 * the incident particle
 *
 * See section 10.1.4 of the Geant4 physics reference manual (release 10.6).
 */
template<class Engine>
CELER_FUNCTION Interaction MollerBhabhaInteractor::operator()(Engine& rng)
{
    // Incident particle scatters an electron
    Secondary* electron_secondary = this->allocate_(1);
    if (electron_secondary == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // ??
    (void)sizeof(rng);

    // Set up a commonly used constant. NOTE: c = 1
    real_type electron_mass_c_sq = shared_.electron_mass_.value()
                                   * units::CLightSq().value();

    // Total XS formulas are valid below this threshold
    real_type max_kinetic_energy;

    // Moller scattering (e-e-)
    if (inc_particle_is_electron_)
    {
        max_kinetic_energy = 0.5 * inc_energy_.value();
    }
    // Bhabha scattering (e+e-)
    else
    {
        max_kinetic_energy = inc_energy_.value();
    }

    // TODO: TO BE DOUBLE CHECKED
    if (max_incident_energy().value() < max_kinetic_energy)
    {
        max_kinetic_energy = max_incident_energy().value();
    }

    // If the model's minimum energy is >= the max_kinetic_energy, stop
    if (min_incident_energy().value() >= max_kinetic_energy)
    {
        return Interaction::from_failure();
    }

    // Set up sampling parameters
    real_type total_energy = inc_energy_.value() + electron_mass_c_sq;
    real_type x_min    = min_incident_energy().value() / inc_energy_.value();
    real_type x_max    = max_kinetic_energy / inc_energy_.value();
    real_type gamma    = total_energy / electron_mass_c_sq;
    real_type gamma_sq = gamma * gamma;
    real_type beta_sq  = 1.0 - (1.0 / gamma_sq);
    real_type x, z, rejection_function_g;
    real_type random[2];

    // Moller scattering (e-e-)
    if (inc_particle_is_electron_)
    {
        real_type gg         = (2.0 * gamma - 1.0) / gamma_sq;
        real_type y          = 1.0 - x_max;
        rejection_function_g = 1.0 - gg * x_max
                               + x_max * x_max
                                     * (1.0 - gg + (1.0 - gg * y) / (y * y));

        do
        {
            random[0] = generate_canonical(rng);
            random[1] = generate_canonical(rng);

            x = x_min * x_max / (x_min * (1.0 - random[0]) + x_max * random[0]);
            y = 1.0 - x;
            z = 1.0 - gg * x + x * x * (1.0 - gg + (1.0 - gg * y) / (y * y));

        } while (rejection_function_g * random[1] > z);
    }

    // Bhabha scattering (e+e-)
    else
    {
        real_type y    = 1.0 / (1.0 + gamma);
        real_type y2   = y * y;
        real_type y12  = 1.0 - 2.0 * y;
        real_type b1   = 2.0 - y2;
        real_type b2   = y12 * (3.0 + y2);
        real_type y122 = y12 * y12;
        real_type b4   = y122 * y12;
        real_type b3   = b4 + y122;

        y = x_max * x_max;

        rejection_function_g
            = 1.0
              + (y * y * b4 - x_min * x_min * x_min * b3 + y * b2 - x_min * b1)
                    * beta_sq;
        do
        {
            random[0] = generate_canonical(rng);
            random[1] = generate_canonical(rng);

            x = x_min * x_max / (x_min * (1.0 - random[0]) + x_max * random[0]);
            y = x * x;
            z = 1.0 + (y * y * b4 - x * y * b3 + y * b2 - x * b1) * beta_sq;

        } while (rejection_function_g * random[1] > z);
    }

    // Change in the primary kinetic energy
    real_type delta_kinetic_energy = x * inc_energy_.value();
    real_type final_primary_energy = inc_energy_.value() - delta_kinetic_energy;

    // Change in the primary direction
    Real3 delta_direction;

    real_type delta_momentum
        = sqrt(delta_kinetic_energy
               * (delta_kinetic_energy + 2.0 * electron_mass_c_sq));

    // Theta comes from energy-momentum conservation, phi is isotropic
    real_type cos_theta = delta_kinetic_energy
                          * (total_energy + electron_mass_c_sq)
                          / (delta_momentum * inc_momentum_.value());

    // Geant says if (cos_theta > 1) { cos_theta = 1; }
    CELER_ASSERT(cos_theta >= -1.0 && cos_theta <= 1.0);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action      = Action::scattered;
    result.energy      = units::MevEnergy{final_primary_energy};
    result.direction   = inc_direction_;
    result.secondaries = {electron_secondary, 1};

    // Sample phi isotropically
    real_type sin_theta = sqrt((1.0 - cos_theta) * (1.0 + cos_theta));
    UniformRealDistribution<real_type> random_phi(0, 2 * constants::pi);
    real_type                          phi = random_phi(rng);

    // Rotate outgoing direction
    result.direction = rotate(from_spherical(cos_theta, phi), result.direction);

    delta_direction
        = {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};

    // Save outgoing secondary data
    electron_secondary[0].def_id = shared_.electron_id;
    electron_secondary[0].energy = units::MevEnergy{delta_kinetic_energy};

    for (auto i : range(3))
    {
        electron_secondary->direction[i]
            = inc_direction_[i] * inc_energy_.value()
              - result.direction[i] * result.energy.value();
    }
    normalize_direction(&electron_secondary->direction);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
