//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/ScoringParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/inp/Scoring.hh"

namespace celeritas
{
class ActionRegistry;

namespace optical
{
class DetectorAction;
//---------------------------------------------------------------------------//
/*!
 * Manages user callback for optical detectors.
 *
 * Constructs the \c DetectorAction used by sensitive detectors to send hits
 * back to the user provided callback function. If the callback function is not
 * provided, then no \c DetectorAction is created.
 */
class ScoringParams final
{
  public:
    //!@{
    //! \name Type aliases
    using HitCallbackFunc = inp::OpticalScoring::HitCallbackFunc;
    //!@}

  public:
    // Construct from optical scoring input
    ScoringParams(ActionRegistry*, inp::OpticalScoring);

    // Send hits to user callback function
    void process_hits(Span<DetectorHit> const&) const;

  private:
    std::optional<HitCallbackFunc> detector_callback_;
    std::shared_ptr<DetectorAction> detector_action_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
