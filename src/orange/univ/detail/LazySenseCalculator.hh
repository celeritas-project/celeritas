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
                                              Real3 const& pos,
                                              Span<SenseModFlags> sense_mod);

    // Calculate senses for a single face of the given volume, possibly on a
    // face
    inline CELER_FUNCTION Sense operator()(VolumeView const& vol,
                                           FaceId face_id,
                                           OnFace face = {});

    OnFace& on_face() { return face_; }

  private:
    //! The first face encountered that we are "on"
    OnFace face_;

    //! Apply a function to a local surface
    LocalSurfaceVisitor visit_;

    //! Local position
    Real3 pos_;

    //! Temporary senses
    Span<SenseModFlags> sense_storage_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
CELER_FUNCTION
LazySenseCalculator::LazySenseCalculator(LocalSurfaceVisitor const& visit,
                                         Real3 const& pos,
                                         Span<SenseModFlags> sense_mod)
    : visit_{visit}, pos_(pos), sense_storage_{sense_mod}
{
    for (auto& sense : sense_storage_)
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
CELER_FUNCTION auto LazySenseCalculator::operator()(VolumeView const& vol,
                                                    FaceId face_id,
                                                    OnFace face) -> Sense
{
    CELER_EXPECT(!face || face.id() < vol.num_faces());

    if (!face_ && face)
    {
        face_ = face;
    }

    Sense sense;
    if (face_id != face.id())
    {
        // Calculate sense
        SignedSense ss = visit_(CalcSense{pos_}, vol.get_surface(face_id));
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
        sense = face.sense();
    }
    if (is_sense_mod_set(SenseMod::flipped, sense_storage_[face_id.get()]))
    {
        sense = flip_sense(sense);
    }

    return sense;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
