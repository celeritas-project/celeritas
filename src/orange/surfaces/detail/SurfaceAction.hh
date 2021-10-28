//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

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
 * The templated operator() of the given functor F must be a Surface class. The
 * `result_type` type alias here uses GeneralQuadric to represent the "most
 * generic" type the functor accepts.
 */
template<class F>
class SurfaceAction
{
  public:
    //@{
    //! Type aliases
    using result_type
        = decltype(std::declval<F>()(std::declval<GeneralQuadric>()));
    //@}

  public:
    // Construct from surfaces and action
    inline CELER_FUNCTION SurfaceAction(const Surfaces& surfaces, F&& action);

    // Apply to the surface specified by a surface ID
    inline CELER_FUNCTION result_type operator()(SurfaceId id);

    //! Access the resulting action
    CELER_FUNCTION const F& action() const { return action_; }

  private:
    //// DATA ////
    Surfaces surfaces_;
    F        action_;

    //// METHODS ////
    template<SurfaceType ST>
    inline CELER_FUNCTION result_type apply_impl(SurfaceId id);
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to surfaces and action to apply.
 */
template<class F>
CELER_FUNCTION
SurfaceAction<F>::SurfaceAction(const Surfaces& surfaces, F&& action)
    : surfaces_(surfaces), action_(std::forward<F>(action))
{
}

//---------------------------------------------------------------------------//
/*!
 * Apply to the surface specified by the given surface ID.
 */
template<class F>
CELER_FUNCTION auto SurfaceAction<F>::operator()(SurfaceId id) -> result_type
{
    CELER_EXPECT(id < surfaces_.num_surfaces());
#define ORANGE_SURF_APPLY_IMPL(TYPE) \
    case (SurfaceType::TYPE):        \
        return this->apply_impl<SurfaceType::TYPE>(id)

    switch (surfaces_.surface_type(id))
    {
        ORANGE_SURF_APPLY_IMPL(px);
        ORANGE_SURF_APPLY_IMPL(py);
        ORANGE_SURF_APPLY_IMPL(pz);
        ORANGE_SURF_APPLY_IMPL(cxc);
        ORANGE_SURF_APPLY_IMPL(cyc);
        ORANGE_SURF_APPLY_IMPL(czc);
#if 0
        ORANGE_SURF_APPLY_IMPL(sc);
        ORANGE_SURF_APPLY_IMPL(cx);
        ORANGE_SURF_APPLY_IMPL(cy);
        ORANGE_SURF_APPLY_IMPL(cz);
        ORANGE_SURF_APPLY_IMPL(p);
#endif
        ORANGE_SURF_APPLY_IMPL(s);
#if 0
        ORANGE_SURF_APPLY_IMPL(kx);
        ORANGE_SURF_APPLY_IMPL(ky);
        ORANGE_SURF_APPLY_IMPL(kz);
        ORANGE_SURF_APPLY_IMPL(sq);
#endif
        ORANGE_SURF_APPLY_IMPL(gq);
        case SurfaceType::size_:
            CELER_ASSERT_UNREACHABLE();
    }
#undef ORANGE_SURF_APPLY_IMPL
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
// PRIVATE INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Apply to the surface specified by a surface ID.
 */
template<class F>
template<SurfaceType ST>
CELER_FUNCTION auto SurfaceAction<F>::apply_impl(SurfaceId id) -> result_type
{
    using Surface_t = typename SurfaceTypeTraits<ST>::type;
    return action_(surfaces_.make_surface<Surface_t>(id));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
