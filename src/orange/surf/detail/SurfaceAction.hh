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

#include "../ConeAligned.hh"
#include "../CylAligned.hh"
#include "../CylCentered.hh"
#include "../GeneralQuadric.hh"
#include "../Plane.hh"
#include "../PlaneAligned.hh"
#include "../SimpleQuadric.hh"
#include "../Sphere.hh"
#include "../SphereCentered.hh"
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
    inline CELER_FUNCTION SurfaceAction(Surfaces const& surfaces, F&& action);

    // Apply to the surface specified by a surface ID
    inline CELER_FUNCTION decltype(auto) operator()(LocalSurfaceId id);

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
// PRIVATE MACRO DEFINITIONS
//---------------------------------------------------------------------------//

//! Call a function for the given type, using the type as the argument.
#define ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, TYPE) \
    case SurfaceType::TYPE:                        \
        FUNC(typename SurfaceTypeTraits<SurfaceType::TYPE>::type)

/*!
 * Expand a macro to a switch statement over all possible surface types.
 *
 * The \c FUNC argument should be a macro that:
 * - takes a single argument which is a surface class identifier (e.g.
 *   \c GeneralQuadric)
 * - calls \c return or \c break
 *
 * The \c ST argument must be a value of type \c SurfaceType.
 */
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
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, sc);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, cx);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, cy);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, cz);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, p);   \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, s);   \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, kx);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, ky);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, kz);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, sq);  \
            ORANGE_SURF_DISPATCH_CASE_IMPL(FUNC, gq);  \
            case SurfaceType::size_:                   \
                CELER_ASSERT_UNREACHABLE();            \
        }                                              \
    } while (0)

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
}  // namespace detail
}  // namespace celeritas
