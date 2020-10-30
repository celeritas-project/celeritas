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
#include "MaterialStatePointers.hh"
#include "MaterialParamsPointers.hh"
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
 */
class MaterialTrackView
{
  public:
    //@{
    //! Type aliases
    using Initializer_t = MaterialTrackState;
    //@}

  public:
    // Construct from "static" parameters and "dynamic" state
    inline CELER_FUNCTION
    MaterialTrackView(const MaterialParamsPointers& params,
                      const MaterialStatePointers&  states,
                      ThreadId                      id);

    // Initialize the particle
    inline CELER_FUNCTION MaterialTrackView&
                          operator=(const Initializer_t& other);

    //! Current material identifier
    CELER_FUNCTION MaterialDefId def_id() const { return state_.def_id; }

    // Get a view to material properties
    inline CELER_FUNCTION MaterialView material_view() const;

  private:
    const MaterialParamsPointers& params_;
    MaterialTrackState&           state_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MaterialTrackView.i.hh"
