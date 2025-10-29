//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOffloadInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Types.hh"
#include "celeritas/Types.hh"

class G4EventManager;

namespace celeritas
{
struct SetupOptions;
class SharedParams;

//---------------------------------------------------------------------------//
/*!
 * Abstract base class for offloading tracks to Celeritas.
 *
 * This class \em must be constructed locally on each worker
 * thread/task/stream,
 *
 * \warning Due to Geant4 thread-local allocators, this class \em must be
 * finalized or destroyed on the same CPU thread in which is created and used!
 */
class LocalOffloadInterface
{
  public:
    // Initialize with options and core shared data
    virtual void Initialize(SetupOptions const&, SharedParams&) = 0;

    // Set the event ID and reseed the Celeritas RNG at the start of an event
    virtual void InitializeEvent(int) = 0;

    // Reseed the RNG states
    virtual void Reseed(size_type) = 0;

    // Transport all buffered tracks to completion
    virtual void Flush() = 0;

    // Clear local data and return to an invalid state
    virtual void Finalize() = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    LocalOffloadInterface() = default;
    CELER_DEFAULT_COPY_MOVE(LocalOffloadInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Manage common functionality for offloading EM tracks and optical data.
 */
class LocalOffloadBase : public LocalOffloadInterface
{
  public:
    // Set the event ID and reseed the Celeritas RNG at the start of an event
    void InitializeEvent(int) final;

    // Ensure the event ID is correctly set
    void check_event_id();

    // Validate the thread ID and threading model
    void validate_threading(size_type num_streams) const;

    //! Get the current event ID
    UniqueEventId event_id() const { return event_id_; }

  private:
    // Current event ID or manager for obtaining it
    UniqueEventId event_id_;
    G4EventManager* event_manager_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
