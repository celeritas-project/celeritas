//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateView.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGStateView_hh
#define geometry_VGStateView_hh

#include <VecGeom/navigation/NavigationState.h>
#include "base/Array.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * View to a vector of VecGeom state information.
 *
 * This "view" is expected to be an argument to a geometry-related kernel
 * launch.
 *
 * The \c vgstate and \c vgnext arguments must be the result of
 * vecgeom::NavStateContainer::GetGPUPointer; and they are only meaningful with
 * the corresponding \c vgmaxdepth, the result of \c GeoManager::getMaxDepth .
 */
class VGStateView
{
  public:
    //@{
    //! Type aliases
    using NavState       = vecgeom::NavigationState;
    using Volume         = vecgeom::VPlacedVolume;
    using ConstPtrVolume = const Volume*;
    //@}

    //! Construction parameters
    struct Params
    {
        size_type size       = 0;
        size_type vgmaxdepth = 0;
        void*     vgstate    = nullptr;
        void*     vgnext     = nullptr;

        Real3*     pos       = nullptr;
        Real3*     dir       = nullptr;
        real_type* next_step = nullptr;
    };

    //! Reference to a single element of the state view
    struct Ref
    {
        NavState* vgstate;
        NavState* vgnext;

        Real3*     pos;
        Real3*     dir;
        real_type* next_step;
    };

  public:
    // Construct with host-managed data
    explicit VGStateView(const Params& params);

    //! Number of states
    CELER_INLINE_FUNCTION size_type size() const { return data_.size; }

    // Get a reference to the mutable thread-local state
    CELER_INLINE_FUNCTION Ref operator[](ThreadId id) const;

  private:
    CELER_INLINE_FUNCTION NavState*
                          get_nav_state(void* state, size_type idx) const;

  private:
    Params data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "VGStateView.i.hh"

#endif // geometry_VGStateView_hh
