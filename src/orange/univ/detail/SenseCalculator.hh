//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SenseCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"

#include "LocalVolumeView.hh"
#include "SurfaceFunctors.hh"
#include "Types.hh"

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
 * volume's face with respect to the given position. This class calculates
 * sense on-the-fly during evaluation of a logic expression without caching
 * previously requested senses. The advantage of not caching senses is that we
 * eliminate global memory usage. With infix evaluation, and short-circuiting
 * logic, the cost of recomputing senses should be small.
 *
 * The OnFace constructor's parameter is used to store the first face that we
 * are "on".
 *
 * When constructed with a direction (used only during initialization), a
 * position that lies exactly on a face is assigned the sense that the track
 * is \em moving toward: if \f$ \hat{n} \cdot \hat{\Omega} > 0 \f$, where
 * \f$ \hat{n} \f$ is the outward surface normal and \f$ \hat{\Omega} \f$ is
 * the local direction, the sense is \c outside; if it is negative, the sense
 * is \c inside. The face is still recorded so that the caller can logically
 * place the track on that surface. If the direction is exactly tangent to the
 * surface, the sense is undetermined and the conventional (position-only)
 * sense is used.
 */
class SenseCalculator
{
  public:
    // Construct from persistent, current, and temporary data
    inline CELER_FUNCTION SenseCalculator(LocalSurfaceVisitor const& visit,
                                          LocalVolumeView const& vol,
                                          Real3 const& pos,
                                          OnFace& face);

    // Construct with a direction to resolve senses exactly on a face
    inline CELER_FUNCTION SenseCalculator(LocalSurfaceVisitor const& visit,
                                          LocalVolumeView const& vol,
                                          Real3 const& pos,
                                          Real3 const& dir,
                                          OnFace& face);

    // Calculate senses for a single face of the given volume, possibly on a
    // face
    inline CELER_FUNCTION Sense operator()(FaceId face_id);

  private:
    // Calculate the sense on a surface along the direction of travel
    inline CELER_FUNCTION SignedSense calc_dir_sense(LocalSurfaceId surf);

    //! Apply a function to a local surface
    LocalSurfaceVisitor visit_;

    //! Volume to calculate senses for
    LocalVolumeView const& vol_;

    //! Local position
    Real3 const& pos_;

    //! Local direction (optional: null if not resolving on-face senses)
    Real3 const* dir_{nullptr};

    //! The first face encountered that we are "on"
    OnFace& face_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
CELER_FUNCTION SenseCalculator::SenseCalculator(
    LocalSurfaceVisitor const& visit,
    LocalVolumeView const& vol,
    Real3 const& pos,
    OnFace& face)
    : visit_{visit}, vol_{vol}, pos_{pos}, face_{face}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a direction to resolve senses exactly on a face.
 *
 * The direction must remain valid for the lifetime of this instance.
 */
CELER_FUNCTION SenseCalculator::SenseCalculator(
    LocalSurfaceVisitor const& visit,
    LocalVolumeView const& vol,
    Real3 const& pos,
    Real3 const& dir,
    OnFace& face)
    : visit_{visit}, vol_{vol}, pos_{pos}, dir_{&dir}, face_{face}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate senses for the given volume.
 *
 * If the point is exactly on one of the volume's surfaces, the \c face
 * reference passed during instance construction will be set. If a direction
 * was given, the sense of any face the point is on is that of the side the
 * track is moving toward.
 */
CELER_FUNCTION auto SenseCalculator::operator()(FaceId face_id) -> Sense
{
    CELER_EXPECT(face_id < vol_.num_faces());
    CELER_EXPECT(!face_ || face_.id() < vol_.num_faces());

    Sense sense;
    if (face_id != face_.id())
    {
        // Calculate sense
        LocalSurfaceId surf = vol_.get_surface(face_id);
        SignedSense ss = visit_(CalcSense{pos_}, surf);
        if (ss == SignedSense::on)
        {
            if (dir_)
            {
                // Use the direction of travel to choose the side
                ss = this->calc_dir_sense(surf);
            }
            sense = to_sense(ss);
            if (!face_)
            {
                // This is the first face that we're exactly on: save it
                face_ = {face_id, sense};
            }
        }
        else
        {
            sense = to_sense(ss);
        }
    }
    else
    {
        // Sense is known a priori
        sense = face_.sense();
    }

    CELER_ENSURE(!face_ || face_.id() < vol_.num_faces());
    return sense;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the sense on a surface along the direction of travel.
 *
 * The result is \c on if the direction is tangent to the surface.
 */
CELER_FUNCTION SignedSense SenseCalculator::calc_dir_sense(LocalSurfaceId surf)
{
    CELER_EXPECT(dir_);
    real_type n_dot_dir = dot_product(visit_(CalcNormal{pos_}, surf), *dir_);
    if (n_dot_dir > 0)
    {
        return SignedSense::outside;
    }
    if (n_dot_dir < 0)
    {
        return SignedSense::inside;
    }
    return SignedSense::on;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
