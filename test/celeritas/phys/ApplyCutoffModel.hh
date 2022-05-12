//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ApplyCutoffModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Types.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/generated/ApplyCutoffInteract.hh"

#include "data/ApplyCutoffData.hh"

namespace celeritas
{
class CutoffParams;
}

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Debugging class to kill particles below the energy cutoff.
 */
class ApplyCutoffModel final : public celeritas::Model
{
  public:
    //!@{
    //! Type aliases
    using ActionId      = celeritas::ActionId;
    using SPConstCutoff = std::shared_ptr<const celeritas::CutoffParams>;
    //!@}

  public:
    // Construct from model ID and list of particle IDs to kill
    ApplyCutoffModel(ActionId id, SPConstCutoff cutoffs);

    // Particle types and energy ranges that this model applies to
    SetApplicability applicability() const final;

    //! Apply the interaction kernel to host data
    void execute(CoreHostRef const&) const final;

    // Apply the interaction kernel to device data
    void execute(CoreDeviceRef const&) const final;

    //! ID of the model
    ActionId action_id() const final { return data_.ids.action; }

    //! Short name for the interaction kernel
    std::string label() const final { return "abs-cutoff"; }

    //! Name of the model, for user interaction
    std::string description() const final { return "Absorb low-E tracks"; }

  private:
    SPConstCutoff   cutoffs_;
    ApplyCutoffData data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
