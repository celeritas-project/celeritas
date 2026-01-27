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
 */
class ScoringParams final
{
  public:
    //!@{
    //! \name Type aliases
    using HitCallbackFunc = inp::OpticalScoring::HitCallbackFunc;
    //!@}

  public:
    ScoringParams(ActionRegistry*, inp::OpticalScoring);

    void process_hits(Span<DetectorHit> const&) const;

  private:
    std::optional<HitCallbackFunc> detector_callback_;
    std::shared_ptr<DetectorAction> detector_action_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
