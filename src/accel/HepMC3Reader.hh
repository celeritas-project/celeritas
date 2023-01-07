//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3Reader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <G4Event.hh>
#include <G4VPrimaryGenerator.hh>

// Forward declarations
namespace HepMC3
{
class Reader;
} // namespace HepMC3

class G4VSolid;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Singleton HepMC3 reader class.
 *
 * This singleton is shared among threads so that events can be correctly split
 * up between them, being constructed the first time `instance()` is invoked.
 * As this is a derived `G4VPrimaryGenerator` class, the HepMC3Reader
 * must be used by a concrete implementation of the
 * `G4VUserPrimaryGeneratorAction` class:
 * \code
   void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
   {
       HepMC3Reader::Instance()->GeneratePrimaryVertex(event);
   }
 * \endcode
 */
class HepMC3Reader final : public G4VPrimaryGenerator
{
  public:
    //! Return non-owning pointer to a singleton
    static HepMC3Reader* Instance();

    //! Add primaries to Geant4 event
    void GeneratePrimaryVertex(G4Event* g4_event) final;

    //! Get total number of events
    int num_events() { return num_events_; }

  private:
    G4VSolid*                       world_solid_; // World volume solid
    std::shared_ptr<HepMC3::Reader> input_file_;  // HepMC3 input file
    int                             num_events_;  // Total number of events

  private:
    // Construct singleton with HepMC3 filename; called by Instance()
    HepMC3Reader();
    // Default destructor in .cc
    ~HepMC3Reader();
};

//---------------------------------------------------------------------------//
} // namespace celeritas
