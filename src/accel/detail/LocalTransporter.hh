//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LocalTransporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Track.hh>

#include "corecel/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"

#include "../SetupOptions.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Transport a set of primaries to completion.
 */
class LocalTransporter
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstOptions = std::shared_ptr<const SetupOptions>;
    using SPConstParams  = std::shared_ptr<const CoreParams>;
    //!@}

  public:
    // Construct in an invalid state
    LocalTransporter() = default;

    // Construct with shared (MT) params
    LocalTransporter(SPConstOptions, SPConstParams);

    // Convert a Geant4 track to a Celeritas primary and add to buffer
    void add(const G4Track&);

    // Transport all buffered tracks to completion
    void flush();

    // Set the event ID
    void set_event(EventId);

    // Number of buffered tracks
    size_type buffer_size() const { return buffer_.size(); }

    // Whether the transporter is valid
    explicit operator bool() const { return opts_ && params_ && step_; }

  private:
    SPConstOptions                    opts_;
    SPConstParams                     params_;
    std::shared_ptr<StepperInterface> step_;
    std::vector<Primary>              buffer_;
    EventId                           event_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
