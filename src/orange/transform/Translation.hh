//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Translator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Storage implementation for transformations with rotation.
 *
 * The input translation and rotation are specified by rotating the daughter's
 * coordinate system internally, then translating into the parent. The input
 * rotation must be orthogonal (real and unitary) and will be checked on
 * construction.
 */
class Translation
{
  public:
    //@{
    //! \name Type aliases
    using StorageSpan = Span<const real_type, 3>;
    //@}

    //! Transform type identifier
    static CELER_CONSTEXPR_FUNCTION TransformType transform_type()
    {
        return TransformType::translation;
    }

  public:
    //! Construct with an identity translation
    explicit CELER_FUNCTION Translation() : tra_{0, 0, 0} {}

    //! Construct with the translation vector
    explicit CELER_FUNCTION Translation(Real3 const& trans) : tra_(trans) {}

    // Construct inline from storage
    explicit inline CELER_FUNCTION Translation(StorageSpan s);

    //// ACCESSORS ////

    //! Translation vector
    Real3 const& translation() const { return tra_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&tra_[0], 3}; }

    //// CALCULATION ////

    // Transform from daughter to parent
    inline Real3 transform_up(Real3 const& pos) const;

    // Transform from parent to daughter
    inline Real3 transform_down(Real3 const& parent_pos) const;

    // Rotate from daughter to parent (identity)
    inline Real3 const& rotate_up(Real3 const& d) const;

    //! Rotate from parent to daughter (identity)
    inline Real3 const& rotate_down(Real3 const& d) const;

  private:
    Real3 tra_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct inline from storage.
 */
CELER_FUNCTION Translation::Translation(StorageSpan s) : tra_{s[0], s[1], s[2]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Transform from daughter to parent.
 */
CELER_FORCEINLINE_FUNCTION Real3 Translation::transform_up(Real3 const& pos) const
{
    return pos + tra_;
}

//---------------------------------------------------------------------------//
/*!
 * Transform from parent to daughter.
 */
CELER_FORCEINLINE_FUNCTION Real3
Translation::transform_down(Real3 const& parent_pos) const
{
    return parent_pos - tra_;
}

//---------------------------------------------------------------------------//
/*!
 * Rotate from daughter to parent (identity).
 */
CELER_FORCEINLINE_FUNCTION Real3 const&
Translation::rotate_up(Real3 const& d) const
{
    return d;
}

//---------------------------------------------------------------------------//
/*!
 * Rotate from parent to daughter (identity).
 */
CELER_FORCEINLINE_FUNCTION Real3 const&
Translation::rotate_down(Real3 const& d) const
{
    return d;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
