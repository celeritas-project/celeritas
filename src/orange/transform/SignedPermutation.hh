//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/SignedPermutation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Turn.hh"
#include "orange/OrangeTypes.hh"
#include "orange/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply a rotation that remaps and possibly flips signs.
 *
 * A signed permutation matrix is a special matrix that has only one entry with
 * the value of \f$\pm1\f$ in each row and each column. This is a specialized
 * rotation matrix that, when applied to a vector, simply exchanges the
 * locations and/or flips the signs of the vector entries.
 *
 * This class stores a special version of the daughter-to-parent rotation
 * matrix:
 * \f[
 \mathbf{R} = \begin{bmatrix}
  \mathbf{e}_x \\ \hline
  \mathbf{e}_y \\ \hline
  \mathbf{e}_z
  \end{bmatrix}
  \f]
 * where \f$ \mathbf{e}_u \f$ has exactly one entry with a value \f$ \pm 1
 \f$ and the other entries are zero.
 *
 * The underlying storage are a compressed series of bits in little-endian form
 * that indicate the positions of the nonzero entry followed by the sign:
 * \verbatim
   [flip z'][z' axis][flip y'][y' axis][flip x'][x' axis]
         8   7    6        5   4     3        2   1   0  bit position
 * \endverbatim
 * This allows the "rotate up" to simply copy one value at a time into a new
 * position, and optionally flip the sign of the result.
 *
 * Construction of this class takes a length 3 array of \c SignedAxis values.
 * The sign is a '+' or '-' character and the axis is the position of the
 * nonzero component in that row.
 */
class SignedPermutation
{
  public:
    //!@{
    //! \name Type aliases
    using SignedAxis = std::pair<char, Axis>;
    using SignedAxes = EnumArray<Axis, SignedAxis>;
    using StorageSpan = Span<real_type const, 1>;
    using DataArray = Array<real_type, 1>;
    //!@}

  public:
    // Construct with an identity permutation
    SignedPermutation();

    // Construct with the permutation vector
    explicit SignedPermutation(SignedAxes permutation);

    // Construct inline from storage
    explicit inline CELER_FUNCTION SignedPermutation(StorageSpan s);

    //// ACCESSORS ////

    // Reconstruct the permutation
    SignedAxes permutation() const;

    // Get a view to the data for type-deleted storage
    DataArray data() const;

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
    // At least 16 bits: more than enough to round trip through a float
    using UIntT = short unsigned int;

    //// DATA ////

    UIntT compressed_;

    //// FUNCTIONS ////

    // Maximum compressed integer value used for bounds checking
    static CELER_CONSTEXPR_FUNCTION UIntT max_value() { return (1 << 9) - 1; }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Make a permutation by rotating about the given axis
SignedPermutation make_permutation(Axis ax, QuarterTurn qtr);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct inline from storage.
 */
CELER_FUNCTION SignedPermutation::SignedPermutation(StorageSpan s)
    : compressed_{static_cast<UIntT>(s[0])}
{
    CELER_EXPECT(s[0] >= 0 && s[0] <= static_cast<real_type>(max_value()));
}

//---------------------------------------------------------------------------//
/*!
 * Transform from daughter to parent.
 */
CELER_FORCEINLINE_FUNCTION Real3
SignedPermutation::transform_up(Real3 const& pos) const
{
    return this->rotate_up(pos);
}

//---------------------------------------------------------------------------//
/*!
 * Transform from parent to daughter.
 */
CELER_FORCEINLINE_FUNCTION Real3
SignedPermutation::transform_down(Real3 const& parent_pos) const
{
    return this->rotate_down(parent_pos);
}

//---------------------------------------------------------------------------//
/*!
 * Rotate from daughter to parent.
 */
CELER_FORCEINLINE_FUNCTION Real3 SignedPermutation::rotate_up(Real3 const& d) const
{
    Real3 result;

    UIntT temp = compressed_;
    for (auto ax : range(Axis::size_))
    {
        // Copy to new axis
        unsigned int new_ax = temp & 0b11;
        result[to_int(ax)] = d[new_ax];
        temp >>= 2;
        if (temp & 0b1)
        {
            // Flip the bit, avoiding signed zero
            result[to_int(ax)] = negate(result[to_int(ax)]);
        }
        temp >>= 1;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Rotate from parent to daughter.
 */
CELER_FORCEINLINE_FUNCTION Real3
SignedPermutation::rotate_down(Real3 const& d) const
{
    Real3 result;

    UIntT temp = compressed_;
    for (auto ax : range(Axis::size_))
    {
        // Copy to new axis
        unsigned int new_ax = temp & 0b11;
        result[new_ax] = d[to_int(ax)];
        temp >>= 2;
        if (temp & 0b1)
        {
            // Flip the bit, avoiding signed zero
            result[new_ax] = negate(result[new_ax]);
        }
        temp >>= 1;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
