//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantParticleView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ParticleDefinition.hh>
#include <G4Version.hh>

#include "corecel/math/Quantity.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access invariant particle data from Geant4 with Celeritas units.
 *
 * Geant4 data are all in double precision.
 */
class GeantParticleView
{
  public:
    //!@{
    //! \name Type aliases
    using Charge = Quantity<units::EElectron, double>;
    using Mass = Quantity<units::MevPerCsq, double>;
    using real_type = double;
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

    // Whether it is antimatter
    inline bool is_antiparticle() const;

    // Whether it is an optical photon
    inline bool is_optical_photon() const;

  private:
    G4ParticleDefinition const& pd_;
};

//---------------------------------------------------------------------------//
/*!
 * Decay constant [1/s].
 */
auto GeantParticleView::decay_constant() const -> real_type
{
    if (pd_.GetPDGStable())
    {
        // Decay constant of zero is an infinite half-life, i.e., stable
        return 0;
    }

    // CLHEP time unit system
    using Time = Quantity<units::ClhepTraits::Time, double>;

    // Decay constant is 1/lifetime
    return 1 / native_value_from(Time{pd_.GetPDGLifeTime()});
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
 *
 * Newer versions of Geant4 have a special PDG number.
 */
bool GeantParticleView::is_optical_photon() const
{
    if constexpr (G4VERSION_NUMBER >= 1070)
    {
        // Special Geant4 internal terminology for optical photon
        return this->pdg() == PDGNumber{-22};
    }
    else
    {
        return this->pdg() == PDGNumber{0} && this->name() == "opticalphoton";
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
