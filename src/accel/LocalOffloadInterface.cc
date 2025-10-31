//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOffloadInterface.cc
//---------------------------------------------------------------------------//
#include "LocalOffloadInterface.hh"

#include <G4EventManager.hh>
#include <G4MTRunManager.hh>
#include <G4Threading.hh>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "geocel/GeantUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set the event ID and reseed the Celeritas RNG at the start of an event.
 */
void LocalOffloadBase::InitializeEvent(int id)
{
    CELER_EXPECT(id >= 0);

    event_id_ = id_cast<UniqueEventId>(id);

    if (!(G4Threading::IsMultithreadedApplication()
          && G4MTRunManager::SeedOncePerCommunication()))
    {
        // Since Geant4 schedules events dynamically, reseed the Celeritas RNGs
        // using the Geant4 event ID for reproducibility. This guarantees that
        // an event can be reproduced given the event ID.
        this->Reseed(event_id_.get());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Ensure the event ID is correctly set.
 */
void LocalOffloadBase::check_event_id()
{
    if (event_manager_ || !event_id_)
    {
        if (CELER_UNLIKELY(!event_manager_))
        {
            // Save the event manager pointer, thereby marking that
            // *subsequent* events need to have their IDs checked as well
            event_manager_ = G4EventManager::GetEventManager();
            CELER_ASSERT(event_manager_);
        }

        G4Event const* event = event_manager_->GetConstCurrentEvent();
        CELER_ASSERT(event);
        if (event_id_ != id_cast<UniqueEventId>(event->GetEventID()))
        {
            // The event ID has changed: reseed it
            this->InitializeEvent(event->GetEventID());
        }
    }
    CELER_ENSURE(event_id_);
};

//---------------------------------------------------------------------------//
/*!
 * Validate the thread ID and threading model.
 */
void LocalOffloadBase::validate_threading(size_type num_streams) const
{
    auto thread_id = get_geant_thread_id();
    CELER_VALIDATE(thread_id >= 0,
                   << "Geant4 ThreadID (" << thread_id
                   << ") is invalid (perhaps local offload is being built "
                      "on a non-worker thread?)");
    CELER_VALIDATE(static_cast<size_type>(thread_id) < num_streams,
                   << "Geant4 ThreadID (" << thread_id
                   << ") is out of range for the reported number of worker "
                      "threads ("
                   << num_streams << ")");

    // Check that OpenMP and Geant4 threading models don't collide
    if (CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK && !celeritas::device()
        && G4Threading::IsMultithreadedApplication())
    {
        auto msg = CELER_LOG(warning);
        msg << "Using multithreaded Geant4 with Celeritas track-level OpenMP "
               "parallelism";
        if (std::string const& nt_str = celeritas::getenv("OMP_NUM_THREADS");
            !nt_str.empty())
        {
            msg << "(OMP_NUM_THREADS=" << nt_str
                << "): CPU threads may be oversubscribed";
        }
        else
        {
            msg << ": forcing 1 Celeritas thread to Geant4 thread";
#ifdef _OPENMP
            omp_set_num_threads(1);
#else
            CELER_ASSERT_UNREACHABLE();
#endif
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
