//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/MucfInteractorHostTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/mucf/data/DTMixMucfData.hh"
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

    // Construct MuCF data from test values for interactors
    HostVal<DTMixMucfData> make_host_data();
};

class MucfInteractorHostTestBase : public MucfInteractorHostBase, public Test
{
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
