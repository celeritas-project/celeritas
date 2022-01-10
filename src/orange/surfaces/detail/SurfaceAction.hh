//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "base/Algorithms.hh"
#include "base/Macros.hh"
#include "orange/Types.hh"
#include "../PlaneAligned.hh"
#include "../CylCentered.hh"
#include "../GeneralQuadric.hh"
#include "../Sphere.hh"
#include "../SurfaceTypeTraits.hh"
#include "../Surfaces.hh"

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
    // Construct from surfaces and action
    inline CELER_FUNCTION SurfaceAction(const Surfaces& surfaces, F&& action);

    // Apply to the surface specified by a surface ID
    inline CELER_FUNCTION decltype(auto) operator()(SurfaceId id);

    //! Access the resulting action
    CELER_FUNCTION const F& action() const { return action_; }

  private:
    //// DATA ////
    Surfaces surfaces_;
    F        action_;
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
// PRIVATE MACRO DEFINITIONS
//---------------------------------------------------------------------------//

#define ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, TYPE) \
    case SurfaceType::TYPE:                        \
        FUNC(typename SurfaceTypeTraits<SurfaceType::TYPE>::type) break

#define ORANGE_SURF_DISPATCH_IMPL(FUNC, ST)            \
    do                                                 \
    {                                                  \
        switch (ST)                                    \
        {                                              \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, px);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, py);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, pz);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, cxc); \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, cyc); \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, czc); \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, sc); */ \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, cx); */ \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, cy); */ \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, cz); */ \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, p);  */ \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, s);   \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, kx); */ \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, ky); */ \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, kz); */ \
            /* ORANGE_SURF_DISPATCH_CASE(FUNC, sq); */ \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, gq);  \
            case SurfaceType::size_:                   \
                CELER_ASSERT_UNREACHABLE();            \
        }                                              \
    } while (0)

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to surfaces and action to apply.
 */
template<class F>
CELER_FUNCTION
SurfaceAction<F>::SurfaceAction(const Surfaces& surfaces, F&& action)
    : surfaces_(surfaces), action_(::celeritas::forward<F>(action))
{
}

//---------------------------------------------------------------------------//
/*!
 * Apply to the surface specified by the given surface ID.
 */
template<class F>
CELER_FUNCTION auto SurfaceAction<F>::operator()(SurfaceId id)
    -> decltype(auto)
{
    CELER_EXPECT(id < surfaces_.num_surfaces());
#define ORANGE_SA_APPLY_IMPL(SURFACE) \
    return action_(surfaces_.make_surface<SURFACE>(id));

    ORANGE_SURF_DISPATCH_IMPL(ORANGE_SA_APPLY_IMPL, surfaces_.surface_type(id));
#undef ORANGE_SA_APPLY_IMPL
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Apply to the surface specified by the given surface ID.
 */
template<template<class> class T>
CELER_FUNCTION decltype(auto)
StaticSurfaceAction<T>::operator()(SurfaceType type) const
{
#define ORANGE_SSA_GET(SURFACE) return T<SURFACE>()();
    ORANGE_SURF_DISPATCH_IMPL(ORANGE_SSA_GET, type);
#undef ORANGE_SSA_GET
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
#undef ORANGE_SURF_DISPATCH_CASE_IMPL
#undef ORANGE_SURF_DISPATCH_IMPL
} // namespace detail
} // namespace celeritas
