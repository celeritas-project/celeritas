//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SenseCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

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
 *
 * The template parameter is the type of the stored direction: \c
 * std::nullptr_t (the default, selected by the constructor without a
 * direction) or <code>Real3 const*</code> (selected by the constructor with a
 * direction). Resolving on-face senses with the direction is thus enabled or
 * eliminated at compile time, so the common position-only case incurs no
 * runtime check.
 */
template<class DirT = std::nullptr_t>
class SenseCalculator
{
    static_assert(std::is_same_v<DirT, std::nullptr_t>
                  || std::is_same_v<DirT, Real3 const*>);

  public:
    //! Whether a direction is used to resolve senses exactly on a face
    static constexpr bool use_direction = !std::is_null_pointer_v<DirT>;

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
    //! Apply a function to a local surface
    LocalSurfaceVisitor visit_;

    //! Volume to calculate senses for
    LocalVolumeView const& vol_;

    //! Local position
    Real3 const& pos_;

    //! The first face encountered that we are "on"
    OnFace& face_;

    //! Local direction (if used to resolve on-face senses)
    DirT dir_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

SenseCalculator(
    LocalSurfaceVisitor const&, LocalVolumeView const&, Real3 const&, OnFace&)
    -> SenseCalculator<std::nullptr_t>;

SenseCalculator(LocalSurfaceVisitor const&,
                LocalVolumeView const&,
                Real3 const&,
                Real3 const&,
                OnFace&) -> SenseCalculator<Real3 const*>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent, current, and temporary data.
 */
template<class DirT>
CELER_FUNCTION SenseCalculator<DirT>::SenseCalculator(
    LocalSurfaceVisitor const& visit,
    LocalVolumeView const& vol,
    Real3 const& pos,
    OnFace& face)
    : visit_{visit}, vol_{vol}, pos_{pos}, face_{face}, dir_{nullptr}
{
    static_assert(!use_direction, "direction is required");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a direction to resolve senses exactly on a face.
 *
 * The direction must remain valid for the lifetime of this instance.
 */
template<class DirT>
CELER_FUNCTION SenseCalculator<DirT>::SenseCalculator(
    LocalSurfaceVisitor const& visit,
    LocalVolumeView const& vol,
    Real3 const& pos,
    Real3 const& dir,
    OnFace& face)
    : visit_{visit}, vol_{vol}, pos_{pos}, face_{face}, dir_{&dir}
{
    static_assert(use_direction, "direction is unused");
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
template<class DirT>
CELER_FUNCTION auto SenseCalculator<DirT>::operator()(FaceId face_id) -> Sense
{
    CELER_EXPECT(face_id < vol_.num_faces());
    CELER_EXPECT(!face_ || face_.id() < vol_.num_faces());

    Sense sense;
    if (face_id != face_.id())
    {
        // Calculate sense
        LocalSurfaceId surf = vol_.get_surface(face_id);
        SignedSense ss = visit_(CalcSense{pos_}, surf);
        bool const on_face = (ss == SignedSense::on);
        if constexpr (use_direction)
        {
            if (on_face)
            {
                // Use the direction of travel to choose the side: "on" if
                // the direction is tangent to the surface
                ss = real_to_sense(
                    dot_product(visit_(CalcNormal{pos_}, surf), *dir_));
            }
        }
        sense = to_sense(ss);
        if (on_face && !face_)
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

    CELER_ENSURE(!face_ || face_.id() < vol_.num_faces());
    return sense;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
