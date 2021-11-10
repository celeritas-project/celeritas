//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SenseCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Range.hh"
#include "base/Span.hh"
#include "orange/surfaces/SurfaceAction.hh"
#include "orange/surfaces/Surfaces.hh"
#include "orange/universes/VolumeView.hh"
#include "SurfaceFunctors.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate senses with a fixed particle position.
 *
 * This is an implementation detail for CellInitializer but is also used by
 * complex intersection methods.
 */
class SenseCalculator
{
  public:
    //! Return result
    struct result_type
    {
        Span<Sense> senses; //!< Calculated senses for the volume
        FaceId      face;   //!< The first face encountered that we are "on"
    };

  public:
    // Construct from persistent, current, and temporary data
    inline CELER_FUNCTION SenseCalculator(const Surfaces& surfaces,
                                          const Real3&    pos,
                                          Span<Sense>     storage);

    // Calculate senses for the given volume
    inline CELER_FUNCTION result_type operator()(const VolumeView& vol,
                                                 FaceId            face = {});

  private:
    //! Compressed vector of surface definitions
    Surfaces surfaces_;

    //! Local position
    Real3 pos_;

    //! Temporary senses
    Span<Sense> sense_storage_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
CELER_FUNCTION SenseCalculator::SenseCalculator(const Surfaces& surfaces,
                                                const Real3&    pos,
                                                Span<Sense>     storage)
    : surfaces_(surfaces), pos_(pos), sense_storage_(storage)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate senses for the given volume.
 *
 * If the point is exactly on one of the volume's surfaces, the \c face value
 * of the return will be set.
 */
CELER_FUNCTION auto
SenseCalculator::operator()(const VolumeView& vol, FaceId face) -> result_type
{
    CELER_EXPECT(vol.num_faces() <= sense_storage_.size());
    // Resulting senses are a subset of the storage; and the face is preserved
    result_type result;
    result.senses = sense_storage_.first(vol.num_faces());
    result.face   = face;

    // Build a functor to calculate the sense of a surface ID given the current
    // state position
    auto calc_sense = make_surface_action(surfaces_, CalcSense{pos_});

    // Fill the temp logic vector with values for all surfaces in the cell
    for (FaceId face_id : range(FaceId{vol.num_faces()}))
    {
        // Calculate sense
        SignedSense ss = calc_sense(vol.get_surface(face_id));
        // Save to result scratch space
        result.senses[face_id.unchecked_get()] = to_sense(ss);
        if (!result.face && ss == SignedSense::on)
        {
            // This is the first face that we're exactly on: save it
            result.face = face_id;
        }
    }

    CELER_ENSURE(!result.face || result.face < vol.num_faces());
    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
