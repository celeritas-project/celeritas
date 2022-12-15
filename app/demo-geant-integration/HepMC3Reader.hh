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
class GenEvent;
class Reader;
} // namespace HepMC3

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Minimal HepMC3 reader class. This should be shared among threads
 */
class HepMC3Reader final : public G4VPrimaryGenerator
{
  public:
    struct Primary
    {
        int           pdg;
        double        energy;
        G4ThreeVector vertex;
        G4ThreeVector momentum;
    };

    //!@{
    //! \name Type aliases
    using Primaries = std::vector<Primary>;
    //!@}

  public:
    // Construct with input file
    HepMC3Reader(std::string hepmc3_filename);

    // Add primaries to Geant4 event
    void GeneratePrimaryVertex(G4Event* g4_event) final;

    // Get total number of events (for beamOn)
    std::size_t num_events() { return num_events_; }

  private:
    // Read event and load list of primaries into event_primaries_
    bool read_event();

  private:
    std::shared_ptr<HepMC3::Reader> input_file_; // HepMC3 input file
    HepMC3::GenEvent                gen_event_;  // Current event
    Primaries   event_primaries_;                // Primaries of current event
    std::size_t num_events_;                     // Total number of events
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
