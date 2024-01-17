//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/global/ActionInterface.hh"

#include "MockInteractData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Change the track state and allocate secondaries.
 */
class MockInteractAction final : public ExplicitActionInterface
{
  public:
    // Construct with number of secondaries and post-interact state
    MockInteractAction(ActionId id,
                       std::vector<size_type> const& num_secondaries,
                       std::vector<bool> const& alive);

    // Run on host
    void execute(CoreParams const&, CoreStateHost&) const final;
    // Run on device
    void execute(CoreParams const&, CoreStateDevice&) const final;

    ActionId action_id() const final { return id_; }
    std::string label() const final { return "mock-interact"; }
    std::string description() const final { return "mock interact kernel"; }
    ActionOrder order() const final { return ActionOrder::post; }

    // Get the number of secondaries
    Span<size_type const> num_secondaries() const;
    // Get true/false values for the pending alive
    Span<char const> alive() const;

  private:
    ActionId id_;
    CollectionMirror<MockInteractData> data_;
};

#if !CELER_USE_DEVICE
inline void
MockInteractAction::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
