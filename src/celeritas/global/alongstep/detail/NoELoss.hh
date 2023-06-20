//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/NoELoss.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/Quantity.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreTrackView;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Class that returns zero energy loss.
 */
struct NoELoss
{
    //! This calculator never returns energy loss
    CELER_CONSTEXPR_FUNCTION bool is_applicable(CoreTrackView const&)
    {
        return false;
    }

    //! No energy loss
    CELER_FUNCTION auto calc_eloss(CoreTrackView const&, real_type, bool) const
        -> decltype(auto)
    {
        return zero_quantity();
    }

    //! No slowing down
    static CELER_CONSTEXPR_FUNCTION bool imprecise_range() { return false; }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
