//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Filler.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"

#include "TrackInitData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage persistent track initializer data.
 *
 * \todo \c max_events could potentially be per thread, not global? And we
 * should differentiate between user events and events in flight.
 */
class TrackInitParams final : public ParamsDataInterface<TrackInitParamsData>
{
  public:
    //! Track initializer construction arguments
    struct Input
    {
        size_type capacity{};  //!< Max number of initializers
        size_type max_events{};  //!< Max simultaneous events
        TrackOrder track_order{TrackOrder::none};  //!< How to sort tracks
    };

  public:
    // Construct with capacity and number of events
    explicit TrackInitParams(Input const&);

    //! Maximum number of initializers
    size_type capacity() const { return host_ref().capacity; }

    //! Event number cannot exceed this value
    size_type max_events() const { return host_ref().max_events; }

    //! Track sorting strategy
    TrackOrder track_order() const { return host_ref().track_order; }

    //! Access primaries for constructing track initializer states
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    // Reset track counters in single-event mode
    template<MemSpace M>
    inline void
    reset_track_ids(StreamId stream,
                    TrackInitStateData<Ownership::reference, M>* state) const;

  private:
    // Host/device storage and reference
    ParamsDataStore<TrackInitParamsData> data_;
};

//---------------------------------------------------------------------------//
/*!
 * Reset track counters in single-event mode.
 */
template<MemSpace M>
void TrackInitParams::reset_track_ids(
    StreamId stream, TrackInitStateData<Ownership::reference, M>* state) const
{
    CELER_EXPECT(state);

    Filler<size_type, M> fill_zero{0, stream};
    fill_zero(state->track_counters[AllItems<size_type, M>{}]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
