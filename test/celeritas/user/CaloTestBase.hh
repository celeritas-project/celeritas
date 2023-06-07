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

#include "corecel/io/Label.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/phys/Primary.hh"

#include "StepCollectorTestBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class OutputRegistry;
class StepCollector;
class SimpleCalo;

namespace test
{
//---------------------------------------------------------------------------//
class CaloTestBase : virtual public StepCollectorTestBase
{
  public:
    using VecPrimary = std::vector<Primary>;
    using VecString = std::vector<std::string>;

    struct RunResult
    {
        std::vector<double> edep;
        std::vector<double> edep_err;

        void print_expected() const;
    };

  public:
    // Default destructor
    ~CaloTestBase();

    void SetUp() override;

    virtual VecString get_detector_names() const = 0;

    template<MemSpace M>
    RunResult
    run(size_type num_tracks, size_type num_steps, size_type num_batches = 1);

    // Get JSON output from the simple calo interface
    std::string output() const;

  protected:
    virtual void gather_batch_results();
    virtual void initalize();
    virtual void finalize();

    std::shared_ptr<SimpleCalo> calo_;
    std::shared_ptr<StepCollector> collector_;
    std::shared_ptr<OutputRegistry> output_;

  private:
    RunResult result;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
