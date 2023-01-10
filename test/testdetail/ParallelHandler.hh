//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/ParallelHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>

namespace celeritas
{
class MpiCommunicator;
}

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Integrate google test with MPI.
 */
class ParallelHandler : public ::testing::EmptyTestEventListener
{
  public:
    using Comm = MpiCommunicator;

    // Construct with communicator
    explicit ParallelHandler(Comm const& comm);

    void OnTestProgramStart(::testing::UnitTest const&) override;
    void OnTestProgramEnd(::testing::UnitTest const&) override;
    void OnTestStart(::testing::TestInfo const&) override;
    void OnTestEnd(::testing::TestInfo const&) override;

  private:
    Comm const& comm_;
};

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
