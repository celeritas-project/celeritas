//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Next face ID and the distance to it.
 *
 * We may want to restructuer this if we store a vector of face/distance rather
 * than two separate vectors.
 */
struct TempNextFace
{
    FaceId*    face{nullptr};
    real_type* distance{nullptr};
    size_type  num_faces{0}; //!< "constant" in params

    explicit CELER_FORCEINLINE_FUNCTION operator bool() const
    {
        return static_cast<bool>(face);
    }
    CELER_FORCEINLINE_FUNCTION size_type size() const { return num_faces; }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
