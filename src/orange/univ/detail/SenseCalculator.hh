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
 * intersection. Senses are eagerly calculated for all faces in the volume at
 * construction.
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
                                 VolumeView const& vol,
                                 Real3 const& pos,
                                 Span<SenseValue> storage,
                                 OnFace& face)
    : sense_storage_{storage.first(vol.num_faces())}
{
    CELER_EXPECT(vol.num_faces() <= storage.size());
    LazySenseCalculator lazy_sense_calculator(visit, vol, pos, face);
    // Fill the temp logic vector with values for all surfaces in the volume
    for (FaceId cur_face : range(FaceId{vol.num_faces()}))
    {
        sense_storage_[cur_face.unchecked_get()]
            = lazy_sense_calculator(cur_face);
    }

    CELER_ENSURE(!face || face.id() < sense_storage_.size());
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
