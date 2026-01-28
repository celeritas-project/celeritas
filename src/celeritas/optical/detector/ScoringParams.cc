//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/ScoringParams.cc
//---------------------------------------------------------------------------//
#include "ScoringParams.hh"

#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"

#include "DetectorAction.hh"
#include "DetectorData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct scoring parameters from input.
 *
 * Registers the \c DetectorAction as a post-step action and stores the
 * callback function for the \c DetectorAction to send hits to.
 *
 * If no callback function is provided, then \c DetectorAction is not
 * registered.
 */
ScoringParams::ScoringParams(ActionRegistry* action_reg,
                             inp::OpticalScoring input)
    : detector_callback_(std::move(input.detector_callback))
{
    CELER_EXPECT(action_reg);

    if (detector_callback_)
    {
        detector_action_
            = std::make_shared<DetectorAction>(action_reg->next_id());
        CELER_ASSERT(detector_action_);
        action_reg->insert(detector_action_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Send hits to the user provided callback function.
 *
 * Must be built with a user callback function.
 */
void ScoringParams::process_hits(Span<DetectorHit> const& hits) const
{
    CELER_EXPECT(detector_callback_);

    (*detector_callback_)(hits);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
