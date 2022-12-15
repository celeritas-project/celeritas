//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/ActionInitialization.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VUserActionInitialization.hh>

#include "accel/SharedParams.hh"

#include "HepMC3Reader.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Set up demo-specific action initializations.
 */
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    //!@{
    //! \name Type aliases
    using SPParams       = std::shared_ptr<celeritas::SharedParams>;
    using SPHepMC3Reader = std::shared_ptr<HepMC3Reader>;
    //!@}

  public:
    ActionInitialization();
    void BuildForMaster() const final {}
    void Build() const final;

  private:
    SPParams       params_;
    SPHepMC3Reader hepmc3_reader_{nullptr}; // Shared among worker threads
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
