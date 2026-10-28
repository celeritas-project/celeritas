//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/MctruthTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <vector>

#include "celeritas/phys/Primary.hh"

#include "AssertionHelper.hh"
#include "SimpleLoopTestBase.hh"
#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class StepCollector;

namespace test
{
//---------------------------------------------------------------------------//
class ExampleMctruth;

class MctruthTestBase : virtual public SimpleLoopTestBase
{
  public:
    using VecPrimary = std::vector<Primary>;

    struct RunResult
    {
        static constexpr char class_str[] = "MctruthTestBase::RunResult";

        struct Tolerance
        {
            double pos{Test::fine_eps};
            double dir{Test::fine_eps};
        };

        std::vector<int> event;
        std::vector<int> track;
        std::vector<int> step;
        std::vector<int> volume;
        std::vector<Real3> pos;
        std::vector<Real3> dir;

        friend std::ostream& operator<<(std::ostream&, RunResult const&);
        friend void print_expected(RunResult const&);
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
