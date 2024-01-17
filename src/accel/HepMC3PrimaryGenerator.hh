//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3PrimaryGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <G4Event.hh>
#include <G4VPrimaryGenerator.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

class G4VSolid;

namespace HepMC3
{
class Reader;
class GenEvent;
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

    CELER_DELETE_COPY_MOVE(HepMC3PrimaryGenerator);

    //! Add primaries to Geant4 event
    void GeneratePrimaryVertex(G4Event* g4_event) final;

    //! Get total number of events
    int NumEvents() { return static_cast<int>(num_events_); }

  private:
    using SPReader = std::shared_ptr<HepMC3::Reader>;
    using SPHepEvt = std::shared_ptr<HepMC3::GenEvent>;
    using size_type = std::size_t;

    size_type num_events_{0};  // Total number of events
    G4VSolid* world_solid_{nullptr};  // World volume solid

    SPReader reader_;  // HepMC3 input reader
    std::mutex read_mutex_;
    std::deque<SPHepEvt> event_buffer_;
    size_type start_event_{0};

    // Read
    SPHepEvt read_event(size_type event_id);
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_HEPMC3
inline HepMC3PrimaryGenerator::HepMC3PrimaryGenerator(std::string const&)
{
    CELER_NOT_CONFIGURED("HepMC3");
    CELER_DISCARD(world_solid_);
    CELER_DISCARD(reader_);
    CELER_DISCARD(read_mutex_);
}

inline void HepMC3PrimaryGenerator::GeneratePrimaryVertex(G4Event*) {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
