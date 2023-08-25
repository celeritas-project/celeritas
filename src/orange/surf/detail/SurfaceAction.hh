//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

#include "../SurfaceTypeTraits.hh"
#include "../Surfaces.hh"
#include "AllSurfaces.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for applying an action functor to a generic surface.
 *
 * The function-like instance of \c F must accept any surface type as an
 * argument: this should always just be a templated \c operator() on the
 * surface class. The result type should be the same regardless of the surface
 * type.
 */
template<class F>
class SurfaceAction
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = decltype(std::declval<F>()(std::declval<Sphere>()));
    //!@}

  public:
    // Construct from surfaces and action
    inline CELER_FUNCTION SurfaceAction(Surfaces const& surfaces, F&& action);

    // Apply to the surface specified by a surface ID
    inline CELER_FUNCTION result_type operator()(LocalSurfaceId id);

    //! Access the resulting action
    CELER_FUNCTION F const& action() const { return action_; }

  private:
    //// DATA ////
    Surfaces surfaces_;
    F action_;
};

//---------------------------------------------------------------------------//
/*!
 * Convert a surface type to a class property via a traits class.
 *
 * The traits class \c T must be templated on surface type, and (like \c
 * std::integral_constant ) have a \verbatim
      constexpr value_type operator()() const noexcept
 * \endverbatim
 * member function for extracting the desired value.
 */
template<template<class> class T>
struct StaticSurfaceAction
{
    // Apply to the surface specified by a surface ID
    inline CELER_FUNCTION decltype(auto) operator()(SurfaceType type) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to surfaces and action to apply.
 */
template<class F>
CELER_FUNCTION
SurfaceAction<F>::SurfaceAction(Surfaces const& surfaces, F&& action)
    : surfaces_(surfaces), action_(::celeritas::forward<F>(action))
{
}

//---------------------------------------------------------------------------//
/*!
 * Apply to the surface specified by the given surface ID.
 */
template<class F>
CELER_FUNCTION auto SurfaceAction<F>::operator()(LocalSurfaceId id)
    -> result_type
{
    CELER_EXPECT(id < surfaces_.num_surfaces());

    return visit_surface_type(
        [this, id](auto st_traits) -> result_type {
            using S = typename decltype(st_traits)::type;
            return action_(surfaces_.make_surface<S>(id));
        },
        surfaces_.surface_type(id));
}

//---------------------------------------------------------------------------//
/*!
 * Apply to the surface specified by the given surface ID.
 */
template<template<class> class T>
CELER_FUNCTION decltype(auto)
StaticSurfaceAction<T>::operator()(SurfaceType type) const
{
    return visit_surface_type(
        [](auto st_traits) {
            using S = typename decltype(st_traits)::type;
            return T<S>{}();
        },
        type);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
