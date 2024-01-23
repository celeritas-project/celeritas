//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/MctruthTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/io/Repr.hh"
#include "celeritas/phys/Primary.hh"

#include "StepCollectorTestBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class StepCollector;

namespace test
{
//---------------------------------------------------------------------------//
class ExampleMctruth;

class MctruthTestBase : virtual public StepCollectorTestBase
{
  public:
    using VecPrimary = std::vector<Primary>;

    struct RunResult
    {
        std::vector<int> event;
        std::vector<int> track;
        std::vector<int> step;
        std::vector<int> volume;
        std::vector<double> pos;
        std::vector<double> dir;

        void print_expected() const;
    };

  public:
    // Default destructor
    ~MctruthTestBase();

    void SetUp() override;

    RunResult run(size_type num_tracks, size_type num_steps);

  protected:
    std::shared_ptr<ExampleMctruth> example_mctruth_;
    std::shared_ptr<StepCollector> collector_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
