//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "Surfaces.hh"
#include "detail/SurfaceAction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper function for creating a SurfaceAction instance.
 */
template<class F>
inline CELER_FUNCTION detail::SurfaceAction<F>
                      make_surface_action(const Surfaces& surfaces, F&& action)
{
    return detail::SurfaceAction<F>{surfaces, std::forward<F>(action)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
