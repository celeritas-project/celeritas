//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/data/ParamsDataStore.hh"
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
class MockInteractAction final : public CoreStepActionInterface
{
  public:
    // Construct with number of secondaries and post-interact state
    MockInteractAction(ActionId id,
                       std::vector<size_type> const& num_secondaries,
                       std::vector<bool> const& alive);

    // Run on host
    void step(CoreParams const&, CoreStateHost&) const final;
    // Run on device
    void step(CoreParams const&, CoreStateDevice&) const final;

    ActionId action_id() const final { return id_; }
    std::string_view label() const final { return "mock-interact"; }
    std::string_view description() const final
    {
        return "mock interact kernel";
    }
    StepActionOrder order() const final { return StepActionOrder::post; }

    // Get the number of secondaries
    Span<size_type const> num_secondaries() const;
    // Get true/false values for the pending alive
    Span<char const> alive() const;

  private:
    ActionId id_;
    ParamsDataStore<MockInteractData> data_;
};

#if !CELER_USE_DEVICE
inline void MockInteractAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
