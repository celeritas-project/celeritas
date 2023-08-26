//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"

#include "detail/SurfaceActionImpl.hh"

namespace celeritas
{
class Surfaces;

//---------------------------------------------------------------------------//
/*!
 * Helper function for creating a SurfaceAction instance.
 *
 * The function argument must have an \c operator() that takes a surface class.
 */
template<class F>
inline CELER_FUNCTION detail::SurfaceAction<F>
make_surface_action(Surfaces const& surfaces, F&& action)
{
    return detail::SurfaceAction<F>{surfaces, ::celeritas::forward<F>(action)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
