//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/EventAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4UserEventAction.hh>

#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage begin- and end-of-event setup.
 *
 * This class should be local to a thread/task/stream.
 */
class EventAction final : public G4UserEventAction
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

  public:
    EventAction();

    void BeginOfEventAction(const G4Event* event) final;
    void EndOfEventAction(const G4Event* event) final;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
