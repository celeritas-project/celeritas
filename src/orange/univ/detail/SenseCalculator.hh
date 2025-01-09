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
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"
#include "orange/univ/detail/Types.hh"

#include "LazySenseCalculator.hh"
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
    // Construct from persistent, current, and temporary data
    inline CELER_FUNCTION SenseCalculator(LocalSurfaceVisitor const& visit,
                                          VolumeView const& vol,
                                          Real3 const& pos,
                                          Span<SenseValue> storage,
                                          OnFace& face);

    // Calculate senses for the given volume, possibly on a face
    inline CELER_FUNCTION Sense operator()(FaceId face_id);

    //! Flip the sense of a face
    CELER_FUNCTION void flip_sense(FaceId face_id)
    {
        sense_storage_[face_id.get()] = celeritas::flip_sense((*this)(face_id));
    }

  private:
    //! Apply a function to a local surface
    LocalSurfaceVisitor visit_;

    VolumeView vol_;

    //! Local position
    Real3 pos_;

    //! Temporary senses
    Span<SenseValue> sense_storage_;

    OnFace& face_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
CELER_FUNCTION
SenseCalculator::SenseCalculator(LocalSurfaceVisitor const& visit,
                                 VolumeView const& vol,
                                 Real3 const& pos,
                                 Span<SenseValue> storage,
                                 OnFace& face)
    : visit_{visit}, vol_{vol}, pos_{pos}, sense_storage_{storage}, face_{face}
{
    CELER_EXPECT(vol_.num_faces() <= sense_storage_.size());
    LazySenseCalculator lazy_sense_calculator(visit_, vol_, pos_, face_);
    // Fill the temp logic vector with values for all surfaces in the volume
    auto senses = sense_storage_.first(vol_.num_faces());
    for (FaceId cur_face : range(FaceId{vol_.num_faces()}))
    {
        senses[cur_face.unchecked_get()] = lazy_sense_calculator(cur_face);
    }

    CELER_ENSURE(!face_ || face_.id() < senses.size());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate senses for the given volume.
 *
 * If the point is exactly on one of the volume's surfaces, the \c face
 * reference passed during instance construction will be set.
 */
CELER_FUNCTION Sense SenseCalculator::operator()(FaceId face_id)
{
    CELER_EXPECT(face_id < sense_storage_.size());

    return sense_storage_[face_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
