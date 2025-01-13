//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Tuning.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "celeritas/Types.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Set up per-process state capacities.
 *
 * Capacities are defined as the number per application process: this means
 * that in a multithreaded context it implies "strong scaling" (i.e., the
 * allocations are divided among threads), and in a multiprocess context it
 * implies "weak scaling" (the problem size grows with the number of
 * processes).
 *
 * In other words, if used in a multithread "event-parallel" context, each
 * state gets the specified \c tracks divided by the number of threads.  When
 * used in MPI parallel (e.g., one process per GPU), each process \em rank has
 * \c tracks total threads.
 *
 * \note The \c primaries is a minimum tuning parameter, not a maximum. It was
 * previously named \c auto_flush .
 *
 * Defaults:
 * - \c secondary: twice the number of track slots.
 */
struct StateCapacity
{
    //! Maximum number of track slots to be simultaneously stepped
    size_type tracks{};
    //! Maximum number of queued primaries+secondaries
    size_type initializers{};
    //! Maximum number of secondaries (automatic if zero)
    size_type secondaries{};

    //! Maximum number of simultaneous events (zero for Geant4 integration)
    size_type events{0};

    //! Minimum number of primaries before generating and advancing a step
    size_type primaries{};
};

//---------------------------------------------------------------------------//
/*!
 * When using GPU, change execution options that make it easier to debug.
 */
struct DeviceDebug
{
    //! Launch all kernels on the default stream
    bool default_stream{false};
    //! Synchronize the stream after every kernel launch
    bool sync_stream{false};
};

//---------------------------------------------------------------------------//
/*!
 * Set up system/tuning parameters that don't affect physics.
 *
 * Defaults:
 * - \c track_order: \c init_charge on GPU, \c none on CPU
 *
 * \todo 'seed' doesn't really belong here, not sure where to put it though
 */
struct Tuning
{
    //! Per-process state sizes
    StateCapacity capacity;

    //! Per-process state sizes for *optical* tracking loop
    std::optional<StateCapacity> optical_capacity;

    //! TO BE REMOVED: number of streams
    size_type num_streams{};

    //! Track sorting and initialization
    std::optional<TrackOrder> track_order;

    //! Debug options for device
    std::optional<DeviceDebug> device_debug;

    //! Perform a no-op step at the beginning to improve timing measurements
    bool warm_up{false};

    //! Random number generator seed
    size_type seed{};
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
