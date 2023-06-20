//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/CoulombScatteringInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform Coulomb scattering.
 */
class CoulombScatteringInteractor
{
  public:
    // Construct from shared and state data
    inline CELER_FUNCTION
    CoulombScatteringInteractor(CoulombScatteringData const& shared,
                                ParticleTrackView const& particle,
                                StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// TYPES ////

    //// DATA ////
    CoulombScatteringData const& shared_;
    StackAllocator<Secondary>& allocate_;

    // Incident kinetic energy
    const real_type inc_energy_;

    // Incident direction
    Real3 const& inc_direction_;

    // Incident mass
    const real_type inc_mass_;

    // ?
    real_type lowEnergyThreshold;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION CoulombScatteringInteractor::CoulombScatteringInteractor(
    CoulombScatteringData const& shared,
    ParticleTrackView const& particle,
    StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_()
    , inc_direction_()
    , inc_mass_()
    , allocate_(allocate)
{
    // TODO: CELER_EXPECT the preconditions
}

//---------------------------------------------------------------------------//
/*!
 * Sample Coulomb scattering
 */
template<class Engine>
CELER_FUNCTION Interaction CoulombScatteringInteractor::operator()(Engine& rng)
{
    // std::vector<G4DynamicParticle*>* fvect
    // const G4MaterialCutsCouple* couple
    // const G4DynamicParticle* dp
    // G4double cutEnergy
    // G4double /*maxEnergy*/

    // Absorb particle below low-energy limit
    if (inc_energy_ < lowEnergyThreshold)
    {
        Interaction result = Interaction::from_absorption();
        result.energy = 0.;
        result.energy_deposition = inc_energy;
        return result;
    }

    Interaction result;  // scattered interaction

    // Setup the particle from the definition
    wokvi->SetupParticle(dp->GetDefinition());

    // Setup the material from the cuts
    DefineMaterial(couple);

    // Choose nucleus
    G4Element const* currentElement = SelectRandomAtom(
        couple, particle, inc_energy_, cutEnergy, inc_energy_);
    G4double Z = currentElement->GetZ();

    // If there's no cross section for this element, don't interact
    if (ComputeCrossSectionPerAtom(
            particle, inc_energy_, Z, -1.0, cutEnergy, -1.0)
        == 0.0)
    {
        return;
    }

    // Select nuclide
    G4int iz = G4int(Z);
    G4int ia = SelectIsotopeNumber(currentElement);
    G4double targetMass = G4NucleiProperties::GetNuclearMass(ia, iz);
    wokvi->SetTargetMass(targetMass);

    // Sample new direction from wokvi
    G4ThreeVector newDirection
        = wokvi->SampleSingleScattering(cosTetMinNuc, cosThetaMax, elecRatio);
    G4double cost = newDirection.z();

    // Align new direction
    newDirection.rotateUz(inc_direction_);
    result.direction = newDirection;

    // Incident momentum squared
    const real_type mom2 = inc_energy_ * (inc_energy_ * 2.0 * inc_mass_);

    // Recoil sampling (first order correction to primary)
    G4double trec = mom2 * (1.0 - cost)
                    / (targetMass + (inc_mass_ + inc_energy_) * (1.0 - cost));
    G4double finalT = inc_energy_ - trec;
    // Absorb particle if recoil is below threshold
    if (finalT <= lowEnergyThreshold)
    {
        trec = inc_energy_;
        finalT = 0.0;
    }

    // Set proposed primary kinetic energy
    result.energy = finalT;

    // Determine recoil threshold cut
    G4double tcut = recoilThreshold;
    if (pcuts)
    {
        tcut = std::max(tcut, (*pCuts)[currentMaterialIndex]);
    }

    if (trec > tcut)
    {
        // If recoil energy is above recoil threshold, emit an ion particle
        G4ParticleDefinition* ion = theParticleTable->GetIon(iz, ia, 0.0);
        G4ThreeVector dir
            = (inc_direction_ * sqrt(mom2)
               - newDirection * sqrt(finalT * (2.0 * inc_mass_ + finalT)))
                  .unit();
        G4DynamicParticle* newdp = new G4DynamicParticle(ion, dir, trec);
        fvect->push_back(newdp);

        Secondary* secondaries = this->allocate_(1);
        if (secondaries == nullptr)
        {
            // Failed to allocate space for ejected ion
            return Interaction::from_failure();
        }
        // Can we create ion tracks?
    }
    else
    {
        // Otherwise energy deposit
        result.energy_deposition = trec;
        result.secondaries = {secondaries, 0};  // BAD
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
