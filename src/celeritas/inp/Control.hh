//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Control.hh
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
 * Set up per-process state/buffer capacities.
 *
 * Increasing these values increases resource requirements with the trade-off
 * of (usually!) improving performance. A larger number of \c tracks in flight
 * means improved performance on GPU because the standard kernel size
 * increases, but it also means higher memory usage because of the larger
 * number of full states. More \c initializers are necessary for more (and
 * higher-energy) tracks when lots of particles are in flight and producing new
 * child particles. More \c secondaries may be necessary if physical processes
 * that produce many daughters (e.g., atomic relaxation or Bertini cascade) are
 * active. The number of \c events in flight primarily increases the number of
 * active tracks, possible initializers, and produced secondaries (NOTE: see
 * [#1233](https://github.com/celeritas-project/celeritas/issues/1233) ).
 * Finally, the number of \c primaries is the maximum number of pending tracks
 * from an external application before running a kernel to construct \c
 * initializers and execute the stpeping loop.
 *
 * Capacities are defined as the number per application process (task): this
 * means that in a multithreaded context it implies "strong scaling" (i.e., the
 * allocations are divided among threads), and in a multiprocess context it
 * implies "weak scaling" (the problem size grows with the number of
 * processes).
 * In other words, if used in a multithread "event-parallel" context, each
 * state gets the specified \c tracks divided by the number of threads.  When
 * used in MPI parallel (e.g., one process per GPU), each process \em rank has
 * \c tracks total threads.
 *
 * \note The \c primaries was previously named \c auto_flush .
 * \note Previously, \c SetupOptions and \c celer-g4 treated these quantities
 * as "per stream" whereas \c celer-sim used "per process".
 *
 * \todo Some of these parameters will be more automated in the future.
 */
struct StateCapacity
{
    //! Maximum number of primaries that can be buffered before stepping
    size_type primaries{};
    //! Maximum number of queued primaries+secondaries
    size_type initializers{};
    //! Maximum number of track slots to be simultaneously stepped
    size_type tracks{};
};

//---------------------------------------------------------------------------//
/*!
 * Set up per-process state/buffer capacities for the main tracking loop.
 *
 * \note The \c primaries was previously named \c auto_flush .
 * \note Previously, \c SetupOptions and \c celer-g4 treated these quantities
 * as "per stream" whereas \c celer-sim used "per process".
 *
 * Defaults:
 * - \c secondaries: twice the number of track slots
 * - \c events: single event runs at a time
 */
struct CoreStateCapacity : StateCapacity
{
    //! Maximum number of secondaries created per step
    std::optional<size_type> secondaries;

    //! Maximum number of simultaneous events (zero for doing one event at a
    //! time)
    std::optional<size_type> events;
};

//---------------------------------------------------------------------------//
/*!
 * Set up per-process state/buffer capacities for the optical tracking loop.
 *
 * \note \c generators was previously named \c buffer_capacity .
 */
struct OpticalStateCapacity : StateCapacity
{
    //! Maximum number of queued photon-generating steps
    size_type generators{};
};

//---------------------------------------------------------------------------//
/*!
 * When using GPU, change execution options that make it easier to debug.
 *
 * Defaults:
 * - \c sync_stream: \c false unless \c timers.diagnostics.action is \c true.
 */
struct DeviceDebug
{
    //! Synchronize the stream after every kernel launch
    std::optional<bool> sync_stream;
};

//---------------------------------------------------------------------------//
/*!
 * Set up control/tuning parameters that do not affect physics.
 *
 * Defaults:
 * - \c device_debug: absent unless device is enabled
 * - \c optical_capacity: absent unless optical physics is enabled
 * - \c track_order: \c init_charge on GPU, \c none on CPU
 */
struct Control
{
    //! Per-process state sizes
    CoreStateCapacity capacity;

    //! Per-process state sizes for *optical* tracking loop
    std::optional<OpticalStateCapacity> optical_capacity;

    //! Number of streams
    size_type num_streams{};

    //! Track sorting and initialization
    std::optional<TrackOrder> track_order;

    //! Debug options for device
    std::optional<DeviceDebug> device_debug;

    //! Perform a no-op step at the beginning to improve timing measurements
    bool warm_up{false};

    //! Random number generator seed
    unsigned int seed{};
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
