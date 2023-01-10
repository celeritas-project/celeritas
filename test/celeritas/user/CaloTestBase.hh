//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/CaloTestBase.hh
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
class ExampleCalorimeters;

class CaloTestBase : virtual public StepCollectorTestBase
{
  public:
    using VecPrimary = std::vector<Primary>;
    using VecString = std::vector<std::string>;

    struct RunResult
    {
        std::vector<double> edep;

        void print_expected() const;
    };

  public:
    // Default destructor
    ~CaloTestBase();

    void SetUp() override;

    virtual VecString get_detector_names() const = 0;

    RunResult run(size_type num_tracks, size_type num_steps);

  protected:
    std::shared_ptr<ExampleCalorimeters> example_calos_;
    std::shared_ptr<StepCollector> collector_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
