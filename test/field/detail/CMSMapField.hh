//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSMapField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FieldMapInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a volume-based CMS field map
 * excerpted from the CMS detector simulation (CMSSW)
 */
class CMSMapField
{
    using FieldMapRef = detail::FieldMapNativeRef;

  public:
    // Construct with the CMS FieldMap
    CELER_FUNCTION
    CMSMapField(const FieldMapRef& shared);

    // Evaluate the magnetic field for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 pos) const;

  private:
    // Shared constant field map
    const FieldMapRef& shared_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "CMSMapField.i.hh"
