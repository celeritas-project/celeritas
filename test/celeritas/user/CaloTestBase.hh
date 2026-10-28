//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/CaloTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <vector>

#include "corecel/io/Label.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/phys/Primary.hh"

#include "SimpleLoopTestBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class OutputRegistry;
class StepCollector;
class SimpleCalo;

namespace test
{
class AssertionHelper;

//---------------------------------------------------------------------------//
class CaloTestBase : virtual public SimpleLoopTestBase
{
  public:
    using VecPrimary = std::vector<Primary>;
    using VecString = std::vector<std::string>;

    struct RunResult
    {
        struct Tolerance
        {
            double edep{Test::fine_eps};
        };

        std::vector<double> edep;

        void print_expected() const;

        friend std::ostream& operator<<(std::ostream&, RunResult const&);
        friend void check_ref_eq(AssertionHelper&,
                                 RunResult const&,
                                 RunResult const&,
                                 std::nullptr_t);
        friend void check_ref_eq(AssertionHelper&,
                                 RunResult const&,
                                 RunResult const&,
                                 Tolerance const&);
    };

  public:
    // Default destructor
    ~CaloTestBase();

    void SetUp() override;

    virtual VecString get_detector_names() const = 0;

    template<MemSpace M>
    RunResult run(size_type num_tracks, size_type num_steps);

    // Get JSON output from the simple calo interface
    std::string output() const;

  protected:
    std::shared_ptr<SimpleCalo> calo_;
    std::shared_ptr<StepCollector> collector_;
    std::shared_ptr<OutputRegistry> output_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
