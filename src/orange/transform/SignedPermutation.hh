//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/SignedPermutation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply a rotation that remaps and possibly flips signs.
 *
 * The daughter-to-parent rotation matrix in this special case:
 * \f[
 \mathbf{R} = \begin{bmatrix}
  \mathbf{e}_x \\ \hline
  \mathbf{e}_y \\ \hline
  \mathbf{e}_z
  \end{bmatrix}
  \f]
 * where \f$ \mathbf{e}_u \f$ is has exactly one entry with a value \f$ \pm 1
 \f$ and the other entries being zero.
 *
 * TODO: implement error checking to catch reflection.
 *
 * The underlying storage are a compressed series of bits in little-endian form
 * that indicate the positions of the nonzero entry followed by the sign:
 * \verbatim
   [flip z'][z' axis][flip y'][y' axis][flip x'][x' axis]
         11  10 9  8        7  6  5  4        3  2  1  0  bit position
 * \endverbatim
 * This allows the "rotate up" to simply copy one value at a time into a new
 * position, and optionally flip the sign of the result.
 */
class SignedPermutation
{
  public:
    enum Sign
    {
        pos,
        neg
    };

    //!@{
    //! \name Type aliases
    using SignedAxis = std::pair<Sign, Axis>;
    using SignedAxes = Array<SignedAxis, 3>
        //!@}

        //// ACCESSORS ////

        //! Rotation matrix
        CELER_FORCEINLINE_FUNCTION Mat3 const& rotation() const
    {
        return rot_;
    }

    //! Translation vector
    CELER_FORCEINLINE_FUNCTION Real3 const& translation() const
    {
        return tra_;
    }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&rot_[0][0], 12}; }

    //// CALCULATION ////

    // Transform from daughter to parent
    [[nodiscard]] inline CELER_FUNCTION Real3
    transform_up(Real3 const& pos) const;

    // Transform from parent to daughter
    [[nodiscard]] inline CELER_FUNCTION Real3
    transform_down(Real3 const& parent_pos) const;

    // Rotate from daughter to parent
    [[nodiscard]] inline CELER_FUNCTION Real3 rotate_up(Real3 const& dir) const;

    // Rotate from parent to daughter
    [[nodiscard]] inline CELER_FUNCTION Real3
    rotate_down(Real3 const& parent_dir) const;

  private:
    unsigned int compressed_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
