//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalOffload.cc
//---------------------------------------------------------------------------//
#include "LocalOpticalOffload.hh"

#include <G4EventManager.hh>
#include <G4MTRunManager.hh>

#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/ActionRegistryOutput.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/gen/GeneratorAction.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "SetupOptions.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with options and shared data.
 */
LocalOpticalOffload::LocalOpticalOffload(SetupOptions const& options,
                                         SharedParams& params)
{
    CELER_EXPECT(options.optical_capacity);
    CELER_EXPECT(params.optical_params());

    CELER_VALIDATE(params.mode() == SharedParams::Mode::enabled,
                   << "cannot create local optical offload when Celeritas "
                      "offloading is disabled");
    CELER_VALIDATE(options.optical_generator
                       && std::holds_alternative<inp::OpticalOffloadGenerator>(
                           *options.optical_generator),
                   << "invalid optical photon generation mechanism for local "
                      "optical offload");

    // Check the thread ID and MT model
    validate_geant_threading(params.Params()->max_streams());

    // Save a pointer to the generator action
    CELER_ASSERT(params.optical_params()->gen_reg()->size() == 1);
    generate_ = std::dynamic_pointer_cast<optical::GeneratorAction const>(
        params.optical_params()->gen_reg()->at(GeneratorId(0)));
    CELER_ASSERT(generate_);

    // Number of optical photons to buffer before offloading
    auto const& capacity = *options.optical_capacity;
    auto_flush_ = capacity.primaries;

    auto stream_id = id_cast<StreamId>(get_geant_thread_id());

    // Allocate thread-local state data
    auto memspace = celeritas::device() ? MemSpace::device : MemSpace::host;
    if (memspace == MemSpace::device)
    {
        state_ = std::make_shared<optical::CoreState<MemSpace::device>>(
            *params.optical_params(), stream_id, capacity.tracks);
    }
    else
    {
        state_ = std::make_shared<optical::CoreState<MemSpace::host>>(
            *params.optical_params(), stream_id, capacity.tracks);
    }

    // Allocate auxiliary data
    if (params.Params()->aux_reg())
    {
        state_->aux() = std::make_shared<AuxStateVec>(
            *params.Params()->aux_reg(), memspace, stream_id, capacity.tracks);
    }

    // Build the optical transporter
    optical::Transporter::Input inp;
    inp.params = params.optical_params();
    inp.max_step_iters = options.max_optical_step_iters;
    transport_ = std::make_shared<optical::Transporter>(std::move(inp));

    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize with options and shared data.
 */
void LocalOpticalOffload::Initialize(SetupOptions const& options,
                                     SharedParams& params)
{
    *this = LocalOpticalOffload(options, params);
}

//---------------------------------------------------------------------------//
/*!
 * Set the event ID and reseed the Celeritas RNG at the start of an event.
 */
void LocalOpticalOffload::InitializeEvent(int id)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(id >= 0);

    event_id_ = id_cast<UniqueEventId>(id);

    if (!(G4Threading::IsMultithreadedApplication()
          && G4MTRunManager::SeedOncePerCommunication()))
    {
        // Since Geant4 schedules events dynamically, reseed the Celeritas RNGs
        // using the Geant4 event ID for reproducibility. This guarantees that
        // an event can be reproduced given the event ID.
        state_->reseed(transport_->params()->rng(), id_cast<UniqueEventId>(id));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Buffer distribution data for generating optical photons.
 */
void LocalOpticalOffload::Push(optical::GeneratorDistributionData const& data)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(data);

    ScopedProfiling profile_this{"push"};

    buffer_.push_back(data);
    num_photons_ += data.num_photons;

    if (num_photons_ >= auto_flush_)
    {
        this->Flush();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate and transport optical photons from the buffered distribution data.
 */
void LocalOpticalOffload::Flush()
{
    CELER_EXPECT(*this);

    if (buffer_.empty())
    {
        return;
    }

    ScopedProfiling profile_this("flush");

    //! \todo Duplicated in \c LocalTransporter
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
    CELER_ASSERT(event_id_);

    if (celeritas::device())
    {
        CELER_LOG_LOCAL(debug)
            << "Transporting " << num_photons_
            << " optical photons from event " << event_id_.unchecked_get()
            << " with Celeritas";
    }

    // Copy the buffered distributions to device
    generate_->insert(*state_, make_span(buffer_));

    state_->counters().num_pending += num_photons_;
    num_photons_ = 0;
    buffer_.clear();

    // Generate optical photons and transport to completion
    (*transport_)(*state_);
}

//---------------------------------------------------------------------------//
/*!
 * Clear local data.
 */
void LocalOpticalOffload::Finalize()
{
    CELER_EXPECT(*this);

    CELER_VALIDATE(buffer_.empty(),
                   << "offloaded photons (" << num_photons_
                   << " in buffer) were not flushed");

    //! \todo Output optical stats

    // Reset all data
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
