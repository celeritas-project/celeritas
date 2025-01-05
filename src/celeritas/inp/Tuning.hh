//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Tuning.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
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
 * If used in a multithread "event-parallel" context, each state gets a
 * fraction divided by the number of threads.
 *
 * When used in MPI parallel (e.g., one process per GPU), each rank has this
 * many.
 *
 * \note The \c primaries is a minimum tuning parameter, not a maximum.
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
 * Set up GPU capabilities and debugging options.
 *
 * Stream sharing and synchronization might be helpful for debugging potential
 * race conditions or improving timing accuracy (at the cost of reducing
 * performance).
 *
 * The CUDA heap and stack sizes may be needed for VecGeom, which has dynamic
 * resource requirements.
 */
struct Device
{
    //! Launch all kernels on the default stream
    bool default_stream{false};
    //! Synchronize the stream after every kernel launch
    bool sync_stream{false};

    //! Per-thread CUDA stack size (ignored if zero) [B]
    size_type stack_size{};
    //! Global dynamic CUDA heap size (ignored if zero) [B]
    size_type heap_size{};

    // TODO: could add preferred device ID, etc.
};

//---------------------------------------------------------------------------//
/*!
 * Set up system/tuning parameters that don't affect physics.
 *
 * Defaults:
 * - \c track_order: \c init_charge on GPU, \c none on CPU
 */
struct Tuning
{
    //! Per-process state sizes
    StateCapacity capacity;

    //! Per-process state sizes for *optical* tracking loop
    std::optional<StateCapacity> optical_capacity;

    //! REMOVE: number of streams
    size_type num_streams{};

    //! Optional: activate GPU
    std::optional<Device> device;

    //! Track sorting and initialization
    std::optional<TrackOrder> track_order;

    //! Perform a no-op step at the beginning to improve timing measurements
    bool warm_up{false};

    //! Random number generator seed
    size_type seed{};

    //! Environment variables used for program setup/diagnostic
    std::map<std::string, std::string> environ;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
