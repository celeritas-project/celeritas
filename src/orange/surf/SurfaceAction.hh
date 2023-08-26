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
 * The function argument must have an \c operator() that takes a surface type.
 */
template<class F>
inline CELER_FUNCTION detail::SurfaceAction<F>
make_surface_action(Surfaces const& surfaces, F&& action)
{
    return detail::SurfaceAction<F>{surfaces, ::celeritas::forward<F>(action)};
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for creating a StaticSurfaceAction instance.
 *
 * The template parameter must be a class templated on surface type that
 * has an \c operator() for returning the desired value.
 *
 * The result takes a SurfaceType enum as an argument and returns the traits
 * value for the given type.
 */
template<template<class> class T>
inline CELER_FUNCTION detail::StaticSurfaceAction<T>
make_static_surface_action()
{
    return detail::StaticSurfaceAction<T>{};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
