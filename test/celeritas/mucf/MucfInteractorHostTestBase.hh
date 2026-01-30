//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/MucfInteractorHostTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/phys/InteractorHostTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for MuCF interactors.
 *
 * This sets up particle and material parameters suitable for muon-catalyzed
 * fusion tests.
 */
class MucfInteractorHostBase : public InteractorHostBase
{
  public:
    //!@{
    //! Initialize and destroy
    MucfInteractorHostBase();
    ~MucfInteractorHostBase() = default;
    //!@}
};

class MucfInteractorHostTestBase : public MucfInteractorHostBase, public Test
{
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
