//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file detail/ParallelHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>

namespace celeritas
{
class MpiCommunicator;
}

namespace celeritas
{
namespace test
{
namespace detail
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
    explicit ParallelHandler(const Comm& comm);

    void OnTestProgramStart(const ::testing::UnitTest&) override;
    void OnTestProgramEnd(const ::testing::UnitTest&) override;
    void OnTestStart(const ::testing::TestInfo&) override;
    void OnTestEnd(const ::testing::TestInfo&) override;

  private:
    const Comm& comm_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace test
} // namespace celeritas
