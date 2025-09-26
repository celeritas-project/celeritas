//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/CachedLazySenseCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"

#include "LazySenseCalculator.hh"
#include "Types.hh"
#include "../VolumeView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate senses with a fixed particle position.
 *
 * This is an implementation detail used in initialization, boundary crossing,
 * simple *and* complex intersection. Instances of this class are specific to a
 * volume, and a position. Calling an instance evaluates the sense of a
 * volume's face with respect to the given position. This class is used to
 * lazily calculate sense during evaluation of a logic expression, caching
 * previously calculated senses.
 *
 * The OnFace constructor's parameter is used to store the first face that we
 * are "on".
 */
class CachedLazySenseCalculator
{
  public:
    // Construct from persistent, current, and temporary data
    inline CELER_FUNCTION
    CachedLazySenseCalculator(LocalSurfaceVisitor const& visit,
                              VolumeView const& vol,
                              Real3 const& pos,
                              Span<SenseValue> sense_cache,
                              OnFace& face);

    // Calculate senses for a single face of the given volume, possibly on a
    // face
    inline CELER_FUNCTION Sense operator()(FaceId face_id);

    //! Flip the sense of a face
    CELER_FUNCTION void flip_sense(FaceId face_id)
    {
        sense_cache_[face_id.get()] = celeritas::flip_sense((*this)(face_id));
    }

  private:
    //! Sense calculator for the volume
    LazySenseCalculator lazy_sense_calculator_;

    //! Temporary senses
    Span<SenseValue> sense_cache_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
CELER_FUNCTION
CachedLazySenseCalculator::CachedLazySenseCalculator(
    LocalSurfaceVisitor const& visit,
    VolumeView const& vol,
    Real3 const& pos,
    Span<SenseValue> sense_cache,
    OnFace& face)
    : lazy_sense_calculator_{visit, vol, pos, face}
    , sense_cache_{sense_cache.first(vol.num_faces())}
{
    for (auto& sense : sense_cache_)
    {
        sense.reset();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate senses for the given volume.
 *
 * If the point is exactly on one of the volume's surfaces, the \c face
 * reference passed during instance construction will be set.
 */
CELER_FUNCTION auto CachedLazySenseCalculator::operator()(FaceId face_id)
    -> Sense
{
    CELER_EXPECT(face_id < sense_cache_.size());
    auto& cached_sense = sense_cache_[face_id.get()];
    if (!cached_sense.is_assigned())
    {
        cached_sense = lazy_sense_calculator_(face_id);
    }
    return cached_sense;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
