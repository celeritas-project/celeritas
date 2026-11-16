//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantParticleView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant particle data from Geant4 with exact quantities.
 *
 * This annotates the Geant4 view with the correct Quantities for seamless
 * integration with Celeritas.
 */
class GeantParticleView
{
  public:
    //!@{
    //! \name Type aliases
    using real_type = double;
    using Charge = Quantity<units::EElectron, double>;
    using Mass = Quantity<units::MevPerCsq, double>;
    using Energy = units::ClhepEnergy;
    using InvTime = Quantity<UnitInverse<units::Nanosecond>, double>;
    //!@}

  public:
    // Find the Geant4 particle definition for a PDG number
    static inline G4ParticleDefinition const* find(PDGNumber pdg);

    // Construct from G4ParticleDefinition
    explicit GeantParticleView(G4ParticleDefinition const& def) : pd_(def) {}

    //! Name
    std::string const& name() const { return pd_.GetParticleName(); }

    // PDG number
    inline PDGNumber pdg() const;

    //! Rest mass [MeV / c^2]
    Mass mass() const { return Mass{pd_.GetPDGMass()}; }

    //! Charge [elemental charge e+]
    Charge charge() const { return Charge{pd_.GetPDGCharge()}; }

    // Decay constant
    inline InvTime decay_constant() const;

    // Whether it is antimatter
    inline bool is_antiparticle() const;

    // Whether it is an optical photon
    inline bool is_optical_photon() const;

  private:
    G4ParticleDefinition const& pd_;
};

//---------------------------------------------------------------------------//
/*!
 * Find the Geant4 particle definition for a PDG number.
 *
 * The result is null if no particle is found.
 */
G4ParticleDefinition const* GeantParticleView::find(PDGNumber pdg)
{
    CELER_EXPECT(pdg);

    auto& table = *G4ParticleTable::GetParticleTable();
    if constexpr (G4VERSION_NUMBER < 1070)
    {
        // Geant4 adopted PDG number -22 for optical photons in version 10.7.
        // Older versions use PDG 0 plus a unique name
        if (pdg == PDGNumber{-22})
        {
            return table.FindParticle("opticalphoton");
        }
    }
    return table.FindParticle(pdg.get());
}

//---------------------------------------------------------------------------//
/*!
 * Get the PDG number.
 */
PDGNumber GeantParticleView::pdg() const
{
    if constexpr (G4VERSION_NUMBER < 1070)
    {
        // Geant4 adopted PDG number -22 for optical photons in version 10.7.
        // Older versions use PDG 0 plus a unique name
        if (pd_.GetPDGEncoding() == 0 && this->name() == "opticalphoton")
        {
            return PDGNumber{-22};
        }
    }
    return PDGNumber{pd_.GetPDGEncoding()};
}

//---------------------------------------------------------------------------//
/*!
 * Decay constant.
 */
auto GeantParticleView::decay_constant() const -> InvTime
{
    if (pd_.GetPDGStable())
    {
        // Decay constant of zero is an infinite half-life, i.e., stable
        return zero_quantity();
    }

    return InvTime{1 / pd_.GetPDGLifeTime()};
}

//---------------------------------------------------------------------------//
/*!
 * Whether it is an antiparticle.
 */
bool GeantParticleView::is_antiparticle() const
{
    return this->pdg() && this->pdg().get() < 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether it is an optical photon.
 */
bool GeantParticleView::is_optical_photon() const
{
    // Special Geant4 internal terminology for optical photon
    return this->pdg() == PDGNumber{-22};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
