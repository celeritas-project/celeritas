//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParamsPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "ElementDef.hh"
#include "MaterialDef.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access material properties on the device.
 *
 * This view is created from \c MaterialParams. The size of the \c defs data
 * member is the number of particle types (accessed by \c MaterialDefId).
 *
 * \sa MaterialParams (owns the pointed-to data)
 * \sa MaterialTrackView (uses the pointed-to data in a kernel)
 */
struct MaterialParamsPointers
{
    span<const ElementDef>  elements;
    span<const MaterialDef> materials;

    //! Check whether the interface is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !materials.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
