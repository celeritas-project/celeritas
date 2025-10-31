//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOpticalOffload.cc
//---------------------------------------------------------------------------//
#include "LocalOpticalOffload.hh"

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
#include "celeritas/optical/OpticalSizes.json.hh"
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
    : auto_flush_((*options.optical_capacity).primaries)
{
    CELER_VALIDATE(params.mode() == SharedParams::Mode::enabled,
                   << "cannot create local optical offload when Celeritas "
                      "offloading is disabled");
    CELER_VALIDATE(options.optical_generator
                       && std::holds_alternative<inp::OpticalOffloadGenerator>(
                           *options.optical_generator),
                   << "invalid optical photon generation mechanism for local "
                      "optical offload");

    // Check the thread ID and MT model
    this->validate_threading(params.Params()->max_streams());

    StreamId stream_id{static_cast<size_type>(get_geant_thread_id())};

    // Currently the device buffer has a fixed capacity
    auto const& capacity = *options.optical_capacity;
    buffer_.reserve(capacity.generators);

    // Save a pointer to the generator
    auto const& gen_reg = *params.optical_params()->gen_reg();
    CELER_ASSERT(gen_reg.size() == 1);
    generate_ = std::dynamic_pointer_cast<optical::GeneratorAction const>(
        gen_reg.at(GeneratorId(0)));
    CELER_ASSERT(generate_);

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
 * Set the event ID and reseed the Celeritas RNG at the start of an event.
 */
void LocalOpticalOffload::Initialize(SetupOptions const& options,
                                     SharedParams& params)
{
    *this = LocalOpticalOffload(options, params);
}

//---------------------------------------------------------------------------//
/*!
 * Reseed the RNG states.
 */
void LocalOpticalOffload::Reseed(size_type id)
{
    CELER_EXPECT(*this);
    state_->reseed(transport_->params()->rng(), UniqueEventId{id});
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
 * Transport the buffered tracks and all secondaries produced.
 */
void LocalOpticalOffload::Flush()
{
    CELER_EXPECT(*this);

    if (buffer_.empty())
    {
        return;
    }

    ScopedProfiling profile_this("flush");

    this->check_event_id();
    CELER_ASSERT(this->event_id());

    if (celeritas::device())
    {
        CELER_LOG_LOCAL(debug)
            << "Transporting " << num_photons_ << " photons from event "
            << this->event_id().unchecked_get() << " with Celeritas";
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
