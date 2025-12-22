//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/AlongStepNeutralImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"

#include "../AlongStep.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreTrackView;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for along-step kernel when no MSC is in use.
 */
struct NoMsc
{
    //! MSC never applies to the current track
    CELER_FUNCTION bool
    is_applicable(CoreTrackView const&, real_type step) const
    {
        CELER_ASSERT(step > 0);
        return false;
    }

    //! No updates needed to the physical and geometric step lengths
    CELER_FUNCTION void limit_step(CoreTrackView const&) const {}

    //! MSC is never applied
    CELER_FUNCTION void apply_step(CoreTrackView const&) const {}
};

//---------------------------------------------------------------------------//
/*!
 * Class that returns zero energy loss.
 */
struct NoELoss
{
    //! No energy loss
    CELER_FUNCTION auto operator()(CoreTrackView const&) const -> decltype(auto)
    {
        return zero_quantity();
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
