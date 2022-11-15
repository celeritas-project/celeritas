//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollectorTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/io/Repr.hh"
#include "celeritas/phys/Primary.hh"

#include "../GlobalTestBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class StepCollector;

namespace test
{
//---------------------------------------------------------------------------//
class ExampleStepCallback;

class StepCollectorTestBase : virtual public GlobalTestBase
{
  public:
    using VecPrimary = std::vector<Primary>;

    struct RunResult
    {
        std::vector<int>    event;
        std::vector<int>    track;
        std::vector<int>    step;
        std::vector<int>    volume;
        std::vector<double> pos;
        std::vector<double> dir;

        void print_expected() const;
    };

  public:
    // Default destructor
    ~StepCollectorTestBase();

    void SetUp() override;

    virtual VecPrimary make_primaries(size_type count) = 0;

    RunResult run(size_type num_tracks, size_type num_steps);

  protected:
    std::shared_ptr<ExampleStepCallback> example_steps_;
    std::shared_ptr<StepCollector>       collector_;
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
