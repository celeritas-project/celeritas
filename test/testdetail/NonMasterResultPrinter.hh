//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/NonMasterResultPrinter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Print test results on non-rank-zero processes.
 */
class NonMasterResultPrinter : public ::testing::EmptyTestEventListener
{
  public:
    // Construct with MPI rank
    explicit NonMasterResultPrinter(int rank);

    void OnTestPartResult(::testing::TestPartResult const& result) override;

  private:
    int rank_;
};

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
