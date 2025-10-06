//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/channel/MuDecayChannel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/inp/Physics.hh"

#include "DecayChannel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch an action to model muon decay kinematics.
 */
class MuDecayChannel : public DecayChannel, public StaticConcreteAction
{
  public:
    // Construct from action ID and imported data
    MuDecayChannel(ActionId, inp::DecayPhysics const&);

    //!@{
    //! \name StepAction interface

    // Apply the interaction kernel on host
    void step(CoreParams const&, CoreStateHost&) const final;
    // Apply the interaction kernel on device
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
