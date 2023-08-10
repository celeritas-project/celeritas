//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <mutex>
#include <G4Event.hh>
#include <G4VPrimaryGenerator.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

class G4VSolid;

namespace HepMC3
{
class Reader;
}  // namespace HepMC3

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * HepMC3 reader class for sharing across threads.
 *
 * This singleton is shared among threads so that events can be correctly split
 * up between them, being constructed the first time `instance()` is invoked.
 * As this is a derived `G4VPrimaryGenerator` class, the HepMC3PrimaryGenerator
 * must be used by a concrete implementation of the
 * `G4VUserPrimaryGeneratorAction` class:
 * \code
   void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
   {
       HepMC3PrimaryGenerator::Instance()->GeneratePrimaryVertex(event);
   }
 * \endcode
 */
class HepMC3PrimaryGenerator final : public G4VPrimaryGenerator
{
  public:
    // Construct with HepMC3 filename
    explicit HepMC3PrimaryGenerator(std::string const& filename);

    //! Add primaries to Geant4 event
    void GeneratePrimaryVertex(G4Event* g4_event) final;

    //! Get total number of events
    int NumEvents() { return num_events_; }

    //! Dump list of primaries to ROOT
    static void dump_to_root(std::string const& hepmc3_input_filename,
                             std::string const& root_output_filename);

  private:
    using SPReader = std::shared_ptr<HepMC3::Reader>;

    int num_events_;  // Total number of events
    G4VSolid* world_solid_{nullptr};  // World volume solid
    SPReader reader_;  // HepMC3 input reader
    std::mutex read_mutex_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_HEPMC3
inline HepMC3PrimaryGenerator::HepMC3PrimaryGenerator(std::string const&)
{
    CELER_NOT_CONFIGURED("HepMC3");
    (void)sizeof(world_solid_);
    (void)sizeof(reader_);
    (void)sizeof(read_mutex_);
}

inline void HepMC3PrimaryGenerator::GeneratePrimaryVertex(G4Event*) {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
