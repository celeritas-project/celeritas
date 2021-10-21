//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "MaterialData.hh"
#include "MaterialData.hh"
#include "MaterialView.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read/write view to the material properties of a single particle track.
 *
 * These functions should be used in each physics Process or Interactor or
 * anything else that needs to access particle properties. Assume that all
 * these functions are expensive: when using them as accessors, locally store
 * the results rather than calling the function repeatedly. If any of the
 * calculations prove to be hot spots we will experiment with cacheing some of
 * the variables.
 *
 * The element scratch space is "thread-private" data with a fixed size
 * *greater than or equal to* the number of elemental components in the current
 * material.
 */
class MaterialTrackView
{
  public:
    //!@{
    //! Type aliases
    using Initializer_t = MaterialTrackState;
    using MaterialParamsData
        = MaterialParamsData<Ownership::const_reference, MemSpace::native>;
    using MaterialStateData
        = MaterialStateData<Ownership::reference, MemSpace::native>;
    //!@}

  public:
    // Construct from "static" parameters and "dynamic" state
    inline CELER_FUNCTION MaterialTrackView(const MaterialParamsData& params,
                                            const MaterialStateData&  states,
                                            ThreadId                  tid);

    // Initialize the particle
    inline CELER_FUNCTION MaterialTrackView&
                          operator=(const Initializer_t& other);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Current material identifier
    inline CELER_FUNCTION MaterialId material_id() const;

    //// STATIC PROPERTIES ////

    // Get a view to material properties
    inline CELER_FUNCTION MaterialView material_view() const;

    // Access scratch space with at least one real per element component
    inline CELER_FUNCTION Span<real_type> element_scratch();

  private:
    const MaterialParamsData&     params_;
    const MaterialStateData&      states_;
    const ThreadId                thread_;

    inline CELER_FUNCTION MaterialTrackState& state() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MaterialTrackView.i.hh"
