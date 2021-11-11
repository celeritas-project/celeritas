//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CellInitializer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../VolumeView.hh"
#include "LogicEvaluator.hh"
#include "SenseCalculator.hh"
#include "Types.hh"

namespace celeritas
{
class Surfaces;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * See if a position is 'inside' a cell.
 *
 * This both *calculates* and *evaluates* senses. It's assumed that the
 * position is fixed but different cells and senses are being tested.
 *
 * TODO: this class has been continually reduced in functionality throughout
 * its evolution, so we may want to just "inline" it into the caller.
 */
class CellInitializer
{
  public:
    //!@{
    //! Type aliases
    using VolumeDataRef
        = VolumeData<Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct with defaults
    inline CELER_FUNCTION
    CellInitializer(const Surfaces& surfaces, const LocalState& state);

    // Test the given cell on the given surface with the given sense
    inline CELER_FUNCTION FoundFace operator()(const VolumeView& vol);

  private:
    //// DATA ////

    SenseCalculator calc_senses_;

    //! Local surface
    OnSurface on_surface_;
};

//---------------------------------------------------------------------------//
/*!
 * Constructor initializes surface calculator and saves surface.
 */
CELER_FUNCTION CellInitializer::CellInitializer(const Surfaces&   surfaces,
                                                const LocalState& state)
    : calc_senses_(surfaces, state.pos, state.temp_senses)
    , on_surface_(state.surface)
{
}

//---------------------------------------------------------------------------//
/*!
 * Test the given cell.
 *
 * \return Whether our stored point is inside the cell.
 */
CELER_FUNCTION auto CellInitializer::operator()(const VolumeView& vol)
    -> FoundFace
{
    // Try to find the local face in our surface list, and save initial sense
    auto result = calc_senses_(vol,
                               OnFace{vol.find_face(on_surface_.id()),
                                      on_surface_.unchecked_sense()});

    LogicEvaluator is_inside(vol.logic());
    bool           found = is_inside(result.senses);
    return FoundFace{found, result.on_face};
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
