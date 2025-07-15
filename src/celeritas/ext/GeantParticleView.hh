//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantParticleView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ParticleDefinition.hh>

#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant particle data from Geant4 with Celeritas units.
 */
class GeantParticleView
{
  public:
    //!@{
    //! \name Type aliases
    using Charge = units::ElementaryCharge;
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct from G4ParticleDefinition
    explicit GeantParticleView(G4ParticleDefinition const& def) : pd_(def) {}

    //! Name
    std::string const& name() const { return pd_.GetParticleName(); }

    //! PDG number
    PDGNumber pdg() const { return PDGNumber{pd_.GetPDGEncoding()}; }

    //! Rest mass [MeV / c^2]
    Mass mass() const { return Mass{pd_.GetPDGMass()}; }

    //! Charge [elemental charge e+]
    Charge charge() const { return Charge{pd_.GetPDGCharge()}; }

    // Decay constant [1/s]
    inline real_type decay_constant() const;

  private:
    G4ParticleDefinition const& pd_;
};

//---------------------------------------------------------------------------//
/*!
 * Decay constant [1/s].
 */
real_type GeantParticleView::decay_constant() const
{
    if (pd_.GetPDGStable())
    {
        return 0;
    }

    // CLHEP time unit system
    using Time = Quantity<units::ClhepTraits::Time, double>;

    // Decay constant is 1/lifetime
    return 1 / native_value_from(Time{pd_.GetPDGLifeTime()});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
