//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LocalTransporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"

#include "../SetupOptions.hh"
#include "G4Track.hh"

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
    using SPCOptions = std::shared_ptr<const SetupOptions>;
    using SPCParams  = std::shared_ptr<const CoreParams>;
    //!@}

  public:
    // Construct with shared (MT) params
    LocalTransporter(SPCParams, SPCOptions);

    // Convert a Geant4 track to a Celeritas primary and add to buffer
    void add(const G4Track&);

    // Transport all buffered tracks to completion
    void flush();

  private:
    SPCParams                         params_;
    SPCOptions                        opts_;
    std::shared_ptr<StepperInterface> step_;
    std::vector<Primary>              buffer_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
