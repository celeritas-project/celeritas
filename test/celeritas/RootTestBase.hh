//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

  private:
    struct ImportHelper;
    class CleanupGeantEnvironment;

    static ImportHelper& import_helper();
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
