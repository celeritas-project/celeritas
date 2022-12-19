//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HepMC3Reader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <G4Event.hh>
#include <G4VPrimaryGenerator.hh>
#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle_fwd.h>
#include <HepMC3/GenVertex_fwd.h>

// Forward declarations
namespace HepMC3
{
class Reader;
} // namespace HepMC3

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * HepMC3 reader class.
 *
 * Singleton constructed the first time `instance()` is invoked by any given
 * thread. As is it a concrete implementation of `G4VPrimaryGenerator`, the
 * reader should be used by a concrete implementation of
 * `G4VUserPrimaryGeneratorAction`:
 * \code
   void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
   {
       HepMC3Reader::instance()->GeneratePrimaryVertex(event);
   }
 * \endcode
 */
class HepMC3Reader final : public G4VPrimaryGenerator
{
  public:
    //! Return non-owning pointer to a singleton
    static HepMC3Reader* instance();

    //! Add primaries to Geant4 event
    void GeneratePrimaryVertex(G4Event* g4_event) final;

    //! Get total number of events
    std::size_t num_events() { return num_events_; }

  private:
    // Construct singleton with HepMC3 filename; called by instance()
    HepMC3Reader();
    // Default destructor in .cc
    ~HepMC3Reader();

    // Read event and load list of primaries into event_primaries_
    bool store_primaries();

  private:
    struct Primary
    {
        int           pdg;
        double        energy;
        G4ThreeVector vertex;
        G4ThreeVector momentum;
    };

    std::shared_ptr<HepMC3::Reader> input_file_; // HepMC3 input file
    std::size_t                     num_events_; // Total number of events
    std::vector<Primary> event_primaries_;       // Primaries of current event
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
