//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParallelHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>

namespace celeritas
{
class Communicator;
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Integrate google test with MPI.
 */
class ParallelHandler : public ::testing::EmptyTestEventListener
{
  public:
    // Construct with communicator
    explicit ParallelHandler(const Communicator& comm);

    void OnTestProgramStart(const ::testing::UnitTest&) override;
    void OnTestProgramEnd(const ::testing::UnitTest&) override;
    void OnTestStart(const ::testing::TestInfo&) override;
    void OnTestEnd(const ::testing::TestInfo&) override;

  private:
    const Communicator& comm_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
