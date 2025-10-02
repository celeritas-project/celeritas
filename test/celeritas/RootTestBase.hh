//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/RootTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ImportedDataTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for loading problem data from a ROOT file
 */
class RootTestBase : public ImportedDataTestBase
{
  protected:
    // Access lazily loaded static ROOT data
    ImportData const& imported_data() const final;

    //! Daughter class can modify data after import
    virtual void fixup_data(ImportData&) const {}

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

  private:
    struct ImportHelper;
    class CleanupGeantEnvironment;

    static ImportHelper& import_helper();
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
