//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/NoTransformation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"

#include "../OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply an identity transformation.
 *
 * This trivial class has the Transformation interface but has no storage
 * requirements and simply passes through all its data.
 */
class NoTransformation
{
  public:
    //@{
    //! \name Type aliases
    using StorageSpan = Span<const real_type, 0>;
    //@}

    //! Transform type identifier
    static CELER_CONSTEXPR_FUNCTION TransformType transform_type()
    {
        return TransformType::no_transformation;
    }

  public:
    //! Construct with an identity NoTransformation
    NoTransformation() = default;

    //! Construct inline from storage
    explicit CELER_FUNCTION NoTransformation(StorageSpan) {}

    //// ACCESSORS ////

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {}; }

    //// CALCULATION ////

    //! Transform from daughter to parent (identity)
    CELER_FUNCTION Real3 const& transform_up(Real3 const& d) const
    {
        return d;
    }

    //! Transform from parent to daughter (identity)
    CELER_FUNCTION Real3 const& transform_down(Real3 const& d) const
    {
        return d;
    }

    //! Rotate from daughter to parent (identity)
    CELER_FUNCTION Real3 const& rotate_up(Real3 const& d) const { return d; }

    //! Rotate from parent to daughter (identity)
    CELER_FUNCTION Real3 const& rotate_down(Real3 const& d) const { return d; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
