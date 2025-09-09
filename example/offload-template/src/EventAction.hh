//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/EventAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserEventAction.hh>

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
class StepDiagnostic;

//---------------------------------------------------------------------------//
/*!
 * Print step statistics at the end of every event.
 */
class EventAction final : public G4UserEventAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPStepDiagnostic = std::shared_ptr<StepDiagnostic>;
    //!@}

  public:
    // From MakeCelerOptions during setup on master, set the step diagnostic
    static void SetStepDiagnostic(SPStepDiagnostic&&);
    // During problem destruction, clear the diagnostic
    static void ClearStepDiagnostic();

    EventAction() = default;

    void BeginOfEventAction(G4Event const*) final {};
    void EndOfEventAction(G4Event const* event) final;

  private:
    SPStepDiagnostic params_;
};

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
