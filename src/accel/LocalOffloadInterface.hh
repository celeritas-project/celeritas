//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LocalOffloadInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Types.hh"

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
    virtual ~LocalOffloadInterface() = default;

    // Initialize with options and core shared data
    virtual void Initialize(SetupOptions const&, SharedParams&) = 0;

    // Set the event ID and reseed the Celeritas RNG at the start of an event
    virtual void InitializeEvent(int) = 0;

    // Transport all buffered tracks to completion
    virtual void Flush() = 0;

    // Clear local data and return to an invalid state
    virtual void Finalize() = 0;

    // Whether the class instance is initialized
    virtual bool Initialized() const = 0;

    // Get the number of buffered tracks
    virtual size_type GetBufferSize() const = 0;

    //! Whether the class instance is initialized
    explicit operator bool() const { return this->Initialized(); }

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    LocalOffloadInterface() = default;
    CELER_DEFAULT_COPY_MOVE(LocalOffloadInterface);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
