//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/LazySenseCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"
#include "orange/univ/detail/Types.hh"

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
class LazySenseCalculator
{
  public:
    // Construct from persistent, current, and temporary data
    inline CELER_FUNCTION LazySenseCalculator(LocalSurfaceVisitor const& visit,
                                              VolumeView const& vol,
                                              Real3 const& pos,
                                              Span<Sense> sense_cache,
                                              Span<SenseModFlags> sense_flags,
                                              OnFace& face);

    // Calculate senses for a single face of the given volume, possibly on a
    // face
    inline CELER_FUNCTION Sense operator()(FaceId face_id);

    //! Flip the sense of a face
    CELER_FUNCTION void flip_sense(FaceId face_id)
    {
        sense_flags_[face_id.get()]
            = flip_sense_mod(SenseMod::flipped, sense_flags_[face_id.get()]);

        // If the sense is cached, flip it, otherwise it will be flipped when
        // we calculate it
        if (is_sense_mod_set(SenseMod::cached, sense_flags_[face_id.get()]))
        {
            sense_cache_[face_id.get()]
                = celeritas::flip_sense(sense_cache_[face_id.get()]);
        }
    }

  private:
    //! Apply a function to a local surface
    LocalSurfaceVisitor visit_;

    //! Volume to calculate senses for
    VolumeView const& vol_;

    //! Local position
    Real3 pos_;

    //! Temporary senses
    Span<Sense> sense_cache_;
    Span<SenseModFlags> sense_flags_;

    //! The first face encountered that we are "on"
    OnFace& face_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
CELER_FUNCTION
LazySenseCalculator::LazySenseCalculator(LocalSurfaceVisitor const& visit,
                                         VolumeView const& vol,
                                         Real3 const& pos,
                                         Span<Sense> sense_cache,
                                         Span<SenseModFlags> sense_flags,
                                         OnFace& face)
    : visit_{visit}
    , vol_(vol)
    , pos_{pos}
    , sense_cache_{sense_cache}
    , sense_flags_{sense_flags}
    , face_(face)
{
    for (auto& sense : sense_flags_)
    {
        sense = static_cast<SenseModFlags>(SenseMod::normal);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate senses for the given volume.
 *
 * If the point is exactly on one of the volume's surfaces, the \c face value
 * of the return will be set.
 */
CELER_FUNCTION auto LazySenseCalculator::operator()(FaceId face_id) -> Sense
{
    CELER_EXPECT(!face_ || face_.id() < vol_.num_faces());

    if (is_sense_mod_set(SenseMod::cached, sense_flags_[face_id.get()]))
    {
        return sense_cache_[face_id.get()];
    }

    Sense sense;
    if (face_id != face_.id())
    {
        // Calculate sense
        SignedSense ss = visit_(CalcSense{pos_}, vol_.get_surface(face_id));
        sense = to_sense(ss);
        if (ss == SignedSense::on && !face_)
        {
            // This is the first face that we're exactly on: save it
            face_ = {face_id, sense};
        }
    }
    else
    {
        // Sense is known a priori
        sense = face_.sense();
    }
    if (is_sense_mod_set(SenseMod::flipped, sense_flags_[face_id.get()]))
    {
        sense = celeritas::flip_sense(sense);
    }

    sense_flags_[face_id.get()]
        = set_sense_mod(SenseMod::cached, sense_flags_[face_id.get()]);
    sense_cache_[face_id.get()] = sense;
    return sense;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
