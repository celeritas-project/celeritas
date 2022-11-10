//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/StepCollector.hh"

#include "../MiniStepTestBase.hh"
#include "../SimpleTestBase.hh"
#include "ExampleStepCollector.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class StepCollectorTestBase : public MiniStepTestBase
{
  public:
    struct RunResult
    {
        void print_expected() const;
    };

    RunResult run(const Input&, size_type num_tracks = 1);

  protected:
    std::shared_ptr<ExampleStepCollector> example_steps;
};

class KnStepCollectorTest : public SimpleTestBase, public StepCollectorTestBase
{
};

//---------------------------------------------------------------------------//
void StepCollectorTestBase::RunResult::print_expected() const {}

//---------------------------------------------------------------------------//

TEST_F(KnStepCollectorTest, host) {}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
