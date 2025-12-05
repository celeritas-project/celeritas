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
    CELER_VALIDATE(params.mode() == SharedParams::Mode::enabled,
                   << "cannot create local optical offload when Celeritas "
                      "offloading is disabled");
    CELER_VALIDATE(options.optical
                       && std::holds_alternative<inp::OpticalOffloadGenerator>(
                           options.optical->generator),
                   << "invalid optical photon generation mechanism for local "
                      "optical offload");

    // Check the thread ID and MT model
    validate_geant_threading(params.Params()->max_streams());

    // Save a pointer to the optical transporter
    transport_ = params.optical_transporter();
    CELER_ASSERT(transport_);

    CELER_ASSERT(transport_->params());
    auto const& optical_params = *transport_->params();

    // Save a pointer to the generator action
    auto const& gen_reg = *optical_params.gen_reg();
    for (auto gen_id : range(GeneratorId(gen_reg.size())))
    {
        if (auto gen = std::dynamic_pointer_cast<optical::GeneratorAction const>(
                gen_reg.at(gen_id)))
        {
            CELER_VALIDATE(!generate_,
                           << "more than one optical GeneratorAction found");
            generate_ = gen;
        }
    }
    CELER_VALIDATE(generate_, << "no optical GeneratorAction found");

    // Number of optical photons to buffer before offloading
    auto const& capacity = options.optical->capacity;
    auto_flush_ = capacity.primaries;

    auto stream_id = id_cast<StreamId>(get_geant_thread_id());

    // Allocate thread-local state data
    auto memspace = celeritas::device() ? MemSpace::device : MemSpace::host;
    if (memspace == MemSpace::device)
    {
        state_ = std::make_shared<optical::CoreState<MemSpace::device>>(
            optical_params, stream_id, capacity.tracks);
    }
    else
    {
        state_ = std::make_shared<optical::CoreState<MemSpace::host>>(
            optical_params, stream_id, capacity.tracks);
    }

    // Allocate auxiliary data
    if (params.Params()->aux_reg())
    {
        state_->aux() = std::make_shared<AuxStateVec>(
            *params.Params()->aux_reg(), memspace, stream_id, capacity.tracks);
    }

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
 * Get the accumulated action times.
 */
auto LocalOpticalOffload::GetActionTime() const -> MapStrDbl
{
    CELER_EXPECT(*this);
    return transport_->get_action_times(*state_->aux());
}

//---------------------------------------------------------------------------//
/*!
 * Clear local data.
 */
void LocalOpticalOffload::Finalize()
{
    CELER_EXPECT(*this);

    CELER_VALIDATE(buffer_.empty(),
                   << "offloaded photons (" << num_photons_ << " in buffer of "
                   << buffer_.size() << " distributions) were not flushed");

    auto const& accum = state_->accum();
    CELER_ASSERT(state_->aux());
    auto const& gen = generate_->counters(*state_->aux());
    CELER_LOG_LOCAL(info)
        << "Finalizing Celeritas after " << accum.steps
        << " optical steps (over " << accum.step_iters << " step iterations)"
        << " from " << gen.accum.num_generated
        << " optical photons generated from " << gen.accum.buffer_size
        << " distributions";

    if (!gen.counters.empty())
    {
        CELER_LOG_LOCAL(warning)
            << "Not all optical photons were tracked "
               "at the end of the stepping loop: "
            << gen.counters.num_pending << " queued photons from "
            << gen.counters.buffer_size << " distributions";
    }

    // Reset all data
    *this = {};

    CELER_ENSURE(!*this);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
