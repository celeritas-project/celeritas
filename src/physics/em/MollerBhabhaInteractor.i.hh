//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaInteractor.i.hh
//---------------------------------------------------------------------------//

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
    const bool&                           is_electron,
    SecondaryAllocatorView&               allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_momentum_(particle.momentum().value())
    , inc_direction_(inc_direction)
    , inc_particle_is_electron_(is_electron)
    , allocate_(allocate)
{
    CELER_EXPECT(inc_energy_ >= this->min_incident_energy()
                 && inc_energy_ <= this->max_incident_energy());
   //CELER_EXPECT(particle.def_id() == shared_.gamma_id); // XXX
}

//---------------------------------------------------------------------------//
/*!
 * Sample e-e- or e+e- scattering using Moller and Bhabha models.
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

    //-----------------------------------------------------
    // Double check the c2. Here c2 is not 1.
    real_type electron_mass_c_sq = shared_.electron_mass_.value()
                                   * units::CLightSq().value();
    real_type max_kinetic_energy;

    // Total XS formulas are valid above these thresholds
    // Moller scattering
    if (inc_particle_is_electron_)
        max_kinetic_energy = 0.5 * inc_energy_.value();

    // Bhabha scattering
    else
        max_kinetic_energy = inc_energy_.value();

    // TO BE DOUBLE CHECKED
    if (max_incident_energy().value() < max_kinetic_energy)
    {
        max_kinetic_energy = max_incident_energy().value();
    }

    // If the model's minimum energy is >= the max_kinetic_energy, stop
    if (min_incident_energy().value() >= max_kinetic_energy)
    {  
        return Interaction::from_failure();
    }

    // Set up parameters
    real_type total_energy = inc_energy_.value() + electron_mass_c_sq;
    real_type x_min    = min_incident_energy().value() / inc_energy_.value();
    real_type x_max    = max_kinetic_energy / inc_energy_.value();
    real_type gamma    = total_energy / electron_mass_c_sq;
    real_type gamma_sq = gamma * gamma;
    real_type beta_sq  = 1.0 - (1.0 / gamma_sq);
    real_type x, z, rejection_function_g;
    real_type random[2];

    // Moller (e-e-) scattering
    if (inc_particle_is_electron_)
    {
        real_type gg = (2.0 * gamma - 1.0) / gamma_sq;

        real_type y = 1.0 - x_max;

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

    // Bhabha (e+e-) scattering
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

    // Selecting a final state independent of material.
    // Geant4 has a flag that calls a G4DeltaAngle::SampleDirection(...)
    // that samples the final direction based on the material if flag
    // UseAngularGeneratorFlag is set to true

    // Change in the primary kinetic energy
    real_type delta_kinetic_energy = x * inc_energy_.value();

    // Change in the primary direction
    Real3 delta_direction;

    real_type delta_momentum
        = sqrt(delta_kinetic_energy
               * (delta_kinetic_energy + 2.0 * electron_mass_c_sq));
    real_type cos_theta = delta_kinetic_energy
                          * (total_energy + electron_mass_c_sq)
                          / (delta_momentum * inc_momentum_.value());

    // WTF Geant?
    if (cos_theta > 1.0)
    {
        cos_theta = 1.0;
    }

    real_type sin_theta = sqrt((1.0 - cos_theta) * (1.0 + cos_theta));
    UniformRealDistribution<real_type> phi(0, 2 * constants::pi);

    delta_direction
        = {sin_theta * std::cos(phi(rng)), sin_theta * std::sin(phi(rng)), cos_theta};
    // delta_direction.rotateUz(dp->GetMomentumDirection());

    // create G4DynamicParticle object for delta ray
    /*
    G4DynamicParticle* delta = new G4DynamicParticle(
        theElectron, delta_direction, delta_kinetic_energy);
    vdp->push_back(delta);

    // primary change
    kineticEnergy -= delta_kinetic_energy;
    G4ThreeVector finalP = dp->GetMomentum() - delta->GetMomentum();
    finalP               = finalP.unit();

    fParticleChange->SetProposedKineticEnergy(kineticEnergy);
    fParticleChange->SetProposedMomentumDirection(finalP);
    */
    //-----------------------------------------------------

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action = Action::scattered;
    result.energy
        = units::MevEnergy{inc_energy_.value() - delta_kinetic_energy};
    for (int i = 0; i < 3; ++i)
    {
        result.direction[i] = inc_direction_[i] + delta_direction[i];
    }

    result.secondaries = {electron_secondary, 1};

    // Save outgoing secondary data
    electron_secondary[0].def_id    = shared_.electron_id;
    electron_secondary[0].energy    = units::MevEnergy{delta_kinetic_energy};
    electron_secondary[0].direction = delta_direction;

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
