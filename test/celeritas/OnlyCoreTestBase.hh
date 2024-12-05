//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/OnlyCoreTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"

#include "GlobalTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mixin class providing "unreachable" implementations for optical params
 * construction.
 */
class OnlyCoreTestBase : virtual public GlobalTestBase
{
  public:
    SPConstCherenkov build_cherenkov() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstOpticalMaterial build_optical_material() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstScintillation build_scintillation() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
};
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
