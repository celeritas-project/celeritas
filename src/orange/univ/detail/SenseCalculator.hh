//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SenseCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"

#include "SurfaceFunctors.hh"
#include "../VolumeView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate senses with a fixed particle position.
 *
 * This is an implementation detail used in initialization *and* complex
 * intersection.
 */
class SenseCalculator
{
  public:
    //! Return result
    struct result_type
    {
        Span<SenseValue> senses;  //!< Calculated senses for the volume
        OnFace face;  //!< The first face encountered that we are "on"
    };

  public:
    // Construct from persistent, current, and temporary data
    inline CELER_FUNCTION SenseCalculator(LocalSurfaceVisitor const& visit,
                                          Real3 const& pos,
                                          Span<SenseValue> storage);

    // Calculate senses for the given volume, possibly on a face
    inline CELER_FUNCTION result_type operator()(VolumeView const& vol,
                                                 OnFace face = {});

  private:
    //! Apply a function to a local surface
    LocalSurfaceVisitor visit_;

    //! Local position
    Real3 pos_;

    //! Temporary senses
    Span<SenseValue> sense_storage_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
CELER_FUNCTION
SenseCalculator::SenseCalculator(LocalSurfaceVisitor const& visit,
                                 Real3 const& pos,
                                 Span<SenseValue> storage)
    : visit_{visit}, pos_(pos), sense_storage_(storage)
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
SenseCalculator::operator()(VolumeView const& vol, OnFace face) -> result_type
{
    CELER_EXPECT(vol.num_faces() <= sense_storage_.size());
    CELER_EXPECT(!face || face.id() < vol.num_faces());

    // Resulting senses are a subset of the storage; and the face is preserved
    result_type result;
    result.senses = sense_storage_.first(vol.num_faces());
    result.face = face;

    // Fill the temp logic vector with values for all surfaces in the volume
    for (FaceId cur_face : range(FaceId{vol.num_faces()}))
    {
        Sense cur_sense;
        if (cur_face != face.id())
        {
            // Calculate sense
            SignedSense ss = visit_(CalcSense{pos_}, vol.get_surface(cur_face));
            cur_sense = to_sense(ss);
            if (!result.face && ss == SignedSense::on)
            {
                // This is the first face that we're exactly on: save it
                result.face = {cur_face, cur_sense};
            }
        }
        else
        {
            // Sense is known a priori
            cur_sense = face.sense();
        }
        // Save sense to result scratch space
        result.senses[cur_face.unchecked_get()] = cur_sense;
    }

    CELER_ENSURE(!result.face || result.face.id() < result.senses.size());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
